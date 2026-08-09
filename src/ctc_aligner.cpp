#include "ctc_aligner.h"

#include <ggml-cpu.h>
#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

#define QWEN3_CTC_MAX_NODES 8192

namespace qwen3_asr {

namespace {

// Applies LayerNorm across ne[0] and then the learned scale/shift, which ggml
// leaves as separate ops.
ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * w, ggml_tensor * b, float eps) {
    cur = ggml_norm(ctx, cur, eps);
    cur = ggml_mul(ctx, cur, w);
    return ggml_add(ctx, cur, b);
}

// Adds a per-channel bias to a [time, channels] tensor. The bias is 1-D, so it
// has to be reshaped to broadcast along the time axis rather than the channels.
ggml_tensor * add_channel_bias(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * bias) {
    return ggml_add(ctx, cur, ggml_reshape_2d(ctx, bias, 1, bias->ne[0]));
}

} // namespace

CtcAligner::CtcAligner() = default;

CtcAligner::~CtcAligner() {
    if (sched_) ggml_backend_sched_free(sched_);
    if (model_buffer_) ggml_backend_buffer_free(model_buffer_);
    if (model_ctx_) ggml_free(model_ctx_);
    if (backend_gpu_) ggml_backend_free(backend_gpu_);
    if (backend_cpu_) ggml_backend_free(backend_cpu_);
}

void CtcAligner::set_n_threads(int n_threads) {
    n_threads_ = std::max(1, n_threads);
    if (backend_cpu_) {
        ggml_backend_cpu_set_n_threads(backend_cpu_, n_threads_);
    }
}

bool CtcAligner::parse_hparams(gguf_context * ctx) {
    const std::string arch = "wav2vec2-ctc";

    auto get_u32 = [&](const char * suffix, int32_t def) -> int32_t {
        int64_t idx = gguf_find_key(ctx, (arch + "." + suffix).c_str());
        return idx < 0 ? def : (int32_t) gguf_get_val_u32(ctx, idx);
    };
    auto get_f32 = [&](const char * suffix, float def) -> float {
        int64_t idx = gguf_find_key(ctx, (arch + "." + suffix).c_str());
        return idx < 0 ? def : gguf_get_val_f32(ctx, idx);
    };
    auto get_i32_array = [&](const char * suffix, std::vector<int32_t> & out) -> bool {
        int64_t idx = gguf_find_key(ctx, (arch + "." + suffix).c_str());
        if (idx < 0 || gguf_get_kv_type(ctx, idx) != GGUF_TYPE_ARRAY) {
            return false;
        }
        const int32_t * data = (const int32_t *) gguf_get_arr_data(ctx, idx);
        size_t n = gguf_get_arr_n(ctx, idx);
        out.assign(data, data + n);
        return true;
    };

    auto & hp = hparams_;
    hp.hidden_size    = get_u32("embedding_length", 1024);
    hp.n_layers       = get_u32("block_count", 24);
    hp.n_heads        = get_u32("attention.head_count", 16);
    hp.ffn_dim        = get_u32("feed_forward_length", 4096);
    hp.layer_norm_eps = get_f32("attention.layer_norm_epsilon", 1e-5f);
    hp.vocab_size     = get_u32("vocab_size", 2341);
    hp.pos_conv_kernel = get_u32("pos_conv.kernel", 128);
    hp.pos_conv_groups = get_u32("pos_conv.groups", 16);
    hp.sample_rate     = get_u32("sample_rate", 16000);
    hp.blank_id        = get_u32("blank_token_id", 0);

    if (!get_i32_array("conv.dim", hp.conv_dim) ||
        !get_i32_array("conv.stride", hp.conv_stride) ||
        !get_i32_array("conv.kernel", hp.conv_kernel)) {
        error_msg_ = "CTC model is missing the conv.dim/stride/kernel arrays";
        return false;
    }
    if (hp.conv_dim.size() != hp.conv_stride.size() || hp.conv_dim.size() != hp.conv_kernel.size()) {
        error_msg_ = "CTC model conv arrays have mismatched lengths";
        return false;
    }
    if (hp.hidden_size % hp.n_heads != 0) {
        error_msg_ = "CTC model hidden size is not divisible by the head count";
        return false;
    }
    if (hp.pos_conv_groups <= 0 || hp.hidden_size % hp.pos_conv_groups != 0) {
        error_msg_ = "CTC model hidden size is not divisible by the positional conv groups";
        return false;
    }

    // The CTC vocabulary is stored index-ordered; the aligner needs token -> id.
    int64_t tok_idx = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (tok_idx < 0) {
        error_msg_ = "CTC model is missing tokenizer.ggml.tokens";
        return false;
    }
    const size_t n_vocab = gguf_get_arr_n(ctx, tok_idx);
    for (size_t i = 0; i < n_vocab; ++i) {
        token_to_id_[gguf_get_arr_str(ctx, tok_idx, i)] = (int32_t) i;
    }

    return true;
}

bool CtcAligner::create_tensors() {
    const auto & hp = hparams_;
    const size_t n_conv = hp.conv_dim.size();
    const size_t n_tensors = n_conv * 4 + 4 /*feat_proj*/ + 2 /*pos_conv*/ + 2 /*enc norm*/
                              + (size_t) hp.n_layers * 16 + 2 /*lm_head*/;

    ggml_init_params params = {
        /*.mem_size   =*/ (n_tensors + 8) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    model_ctx_ = ggml_init(params);
    if (!model_ctx_) {
        error_msg_ = "Failed to create CTC model ggml context";
        return false;
    }

    auto reg = [&](ggml_tensor * t, const std::string & name) {
        ggml_set_name(t, name.c_str());
        tensors_[name] = t;
        return t;
    };
    auto new_1d = [&](int64_t n) { return ggml_new_tensor_1d(model_ctx_, GGML_TYPE_F32, n); };
    auto new_2d = [&](int64_t a, int64_t b) { return ggml_new_tensor_2d(model_ctx_, GGML_TYPE_F16, a, b); };

    conv_layers_.resize(n_conv);
    for (size_t i = 0; i < n_conv; ++i) {
        const int64_t in_ch = (i == 0) ? 1 : hp.conv_dim[i - 1];
        const std::string p = "conv." + std::to_string(i);
        conv_layers_[i].w = reg(ggml_new_tensor_3d(model_ctx_, GGML_TYPE_F16,
                                 hp.conv_kernel[i], in_ch, hp.conv_dim[i]), p + ".weight");
        conv_layers_[i].b      = reg(new_1d(hp.conv_dim[i]), p + ".bias");
        conv_layers_[i].norm_w = reg(new_1d(hp.conv_dim[i]), p + ".norm.weight");
        conv_layers_[i].norm_b = reg(new_1d(hp.conv_dim[i]), p + ".norm.bias");
    }

    const int64_t last_conv_dim = hp.conv_dim.back();
    feat_proj_norm_w_ = reg(new_1d(last_conv_dim), "feat_proj.norm.weight");
    feat_proj_norm_b_ = reg(new_1d(last_conv_dim), "feat_proj.norm.bias");
    feat_proj_w_      = reg(new_2d(last_conv_dim, hp.hidden_size), "feat_proj.weight");
    feat_proj_b_      = reg(new_1d(hp.hidden_size), "feat_proj.bias");

    pos_conv_w_ = reg(ggml_new_tensor_3d(model_ctx_, GGML_TYPE_F16,
                       hp.pos_conv_kernel, hp.hidden_size / hp.pos_conv_groups, hp.hidden_size),
                       "pos_conv.weight");
    pos_conv_b_ = reg(new_1d(hp.hidden_size), "pos_conv.bias");

    enc_norm_w_ = reg(new_1d(hp.hidden_size), "encoder.norm.weight");
    enc_norm_b_ = reg(new_1d(hp.hidden_size), "encoder.norm.bias");

    layers_.resize(hp.n_layers);
    for (int32_t i = 0; i < hp.n_layers; ++i) {
        const std::string p = "blk." + std::to_string(i);
        auto & l = layers_[i];
        l.attn_norm_w = reg(new_1d(hp.hidden_size), p + ".attn_norm.weight");
        l.attn_norm_b = reg(new_1d(hp.hidden_size), p + ".attn_norm.bias");
        l.attn_q_w    = reg(new_2d(hp.hidden_size, hp.hidden_size), p + ".attn_q.weight");
        l.attn_q_b    = reg(new_1d(hp.hidden_size), p + ".attn_q.bias");
        l.attn_k_w    = reg(new_2d(hp.hidden_size, hp.hidden_size), p + ".attn_k.weight");
        l.attn_k_b    = reg(new_1d(hp.hidden_size), p + ".attn_k.bias");
        l.attn_v_w    = reg(new_2d(hp.hidden_size, hp.hidden_size), p + ".attn_v.weight");
        l.attn_v_b    = reg(new_1d(hp.hidden_size), p + ".attn_v.bias");
        l.attn_out_w  = reg(new_2d(hp.hidden_size, hp.hidden_size), p + ".attn_out.weight");
        l.attn_out_b  = reg(new_1d(hp.hidden_size), p + ".attn_out.bias");
        l.ffn_norm_w  = reg(new_1d(hp.hidden_size), p + ".ffn_norm.weight");
        l.ffn_norm_b  = reg(new_1d(hp.hidden_size), p + ".ffn_norm.bias");
        l.ffn_up_w    = reg(new_2d(hp.hidden_size, hp.ffn_dim), p + ".ffn_up.weight");
        l.ffn_up_b    = reg(new_1d(hp.ffn_dim), p + ".ffn_up.bias");
        l.ffn_down_w  = reg(new_2d(hp.ffn_dim, hp.hidden_size), p + ".ffn_down.weight");
        l.ffn_down_b  = reg(new_1d(hp.hidden_size), p + ".ffn_down.bias");
    }

    lm_head_w_ = reg(new_2d(hp.hidden_size, hp.vocab_size), "lm_head.weight");
    lm_head_b_ = reg(new_1d(hp.vocab_size), "lm_head.bias");

    model_buffer_ = ggml_backend_alloc_ctx_tensors(model_ctx_, backend_cpu_);
    if (!model_buffer_) {
        error_msg_ = "Failed to allocate CTC model tensors";
        return false;
    }
    return true;
}

bool CtcAligner::load_tensor_data(const std::string & path, gguf_context * ctx) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        error_msg_ = "Failed to reopen CTC model: " + path;
        return false;
    }

    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    std::vector<char> buf;
    size_t n_loaded = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            error_msg_ = std::string("Unexpected tensor '") + name + "' in CTC model";
            return false;
        }
        ggml_tensor * dst = it->second;

        const int64_t * ne = gguf_get_tensor_ne(ctx, i);
        if (ne[0] != dst->ne[0] || ne[1] != dst->ne[1] || ne[2] != dst->ne[2]) {
            error_msg_ = std::string("Tensor '") + name + "' has an unexpected shape in the CTC model";
            return false;
        }
        if (gguf_get_tensor_type(ctx, i) != dst->type) {
            error_msg_ = std::string("Tensor '") + name + "' has an unexpected type in the CTC model";
            return false;
        }

        const size_t nbytes = ggml_nbytes(dst);
        buf.resize(nbytes);
        fin.seekg(data_offset + gguf_get_tensor_offset(ctx, i), std::ios::beg);
        fin.read(buf.data(), nbytes);
        if (!fin) {
            error_msg_ = std::string("Truncated data for tensor '") + name + "'";
            return false;
        }
        ggml_backend_tensor_set(dst, buf.data(), 0, nbytes);
        n_loaded++;
    }

    if (n_loaded != tensors_.size()) {
        error_msg_ = "CTC model is missing tensors (loaded " + std::to_string(n_loaded) +
                      ", expected " + std::to_string(tensors_.size()) + ")";
        return false;
    }
    return true;
}

bool CtcAligner::load_model(const std::string & model_path) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };

    gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        error_msg_ = "Failed to open CTC GGUF file: " + model_path;
        return false;
    }

    bool ok = parse_hparams(ctx);

    if (ok) {
        backend_cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend_cpu_) {
            error_msg_ = "Failed to initialize CPU backend for the CTC aligner";
            ok = false;
        }
    }
    if (ok) ok = create_tensors();
    if (ok) ok = load_tensor_data(model_path, ctx);

    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);
    if (!ok) return false;

    backend_gpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);

    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> bufts;
    if (backend_gpu_) {
        backends.push_back(backend_gpu_);
        bufts.push_back(ggml_backend_get_default_buffer_type(backend_gpu_));
    }
    backends.push_back(backend_cpu_);
    bufts.push_back(ggml_backend_get_default_buffer_type(backend_cpu_));

    sched_ = ggml_backend_sched_new(backends.data(), bufts.data(), backends.size(),
                                     QWEN3_CTC_MAX_NODES, false, true);
    if (!sched_) {
        error_msg_ = "Failed to create the CTC backend scheduler";
        return false;
    }
    compute_meta_.resize(ggml_tensor_overhead() * QWEN3_CTC_MAX_NODES + ggml_graph_overhead_custom(QWEN3_CTC_MAX_NODES, false));

    set_n_threads(n_threads_);
    model_loaded_ = true;
    return true;
}

ggml_cgraph * CtcAligner::build_graph(ggml_context * ctx0, ggml_tensor * input) {
    const auto & hp = hparams_;
    const int n_head   = hp.n_heads;
    const int n_state  = hp.hidden_size;
    const int head_dim = n_state / n_head;
    const float kq_scale = 1.0f / std::sqrt((float) head_dim);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_CTC_MAX_NODES, false);

    // --- convolutional feature encoder -------------------------------------
    // Layout note: ggml_conv_1d takes [time, channels] and returns [time, channels],
    // but LayerNorm has to run across channels, which ggml normalises along ne[0].
    // So each layer transposes into [channels, time] to normalise and back again.
    ggml_tensor * cur = input;  // [n_samples, 1]
    for (size_t i = 0; i < conv_layers_.size(); ++i) {
        const auto & cl = conv_layers_[i];
        cur = ggml_conv_1d(ctx0, cl.w, cur, hp.conv_stride[i], 0, 1);
        cur = add_channel_bias(ctx0, cur, cl.b);

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));           // [channels, time]
        cur = layer_norm(ctx0, cur, cl.norm_w, cl.norm_b, hp.layer_norm_eps);
        cur = ggml_gelu_erf(ctx0, cur);

        if (i + 1 < conv_layers_.size()) {
            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));       // back to [time, channels]
        }
    }
    // cur is [conv_dim, n_frames], which is already what the projection wants.

    // --- feature projection -------------------------------------------------
    cur = layer_norm(ctx0, cur, feat_proj_norm_w_, feat_proj_norm_b_, hp.layer_norm_eps);
    cur = ggml_mul_mat(ctx0, feat_proj_w_, cur);
    cur = ggml_add(ctx0, cur, feat_proj_b_);                        // [n_state, n_frames]

    const int64_t n_frames = cur->ne[1];

    // --- positional convolution embedding -----------------------------------
    {
        ggml_tensor * pos_in = ggml_cont(ctx0, ggml_transpose(ctx0, cur));  // [n_frames, n_state]

        // Grouped convolution: ggml has no groups parameter, so run one convolution
        // per group over a contiguous slice of channels and concatenate the results.
        const int groups  = hp.pos_conv_groups;
        const int in_per  = n_state / groups;
        const int out_per = n_state / groups;

        ggml_tensor * pos = nullptr;
        for (int g = 0; g < groups; ++g) {
            ggml_tensor * xin = ggml_view_2d(ctx0, pos_in, n_frames, in_per,
                                              pos_in->nb[1], (size_t) g * in_per * pos_in->nb[1]);
            ggml_tensor * wg = ggml_view_3d(ctx0, pos_conv_w_,
                                             pos_conv_w_->ne[0], pos_conv_w_->ne[1], out_per,
                                             pos_conv_w_->nb[1], pos_conv_w_->nb[2],
                                             (size_t) g * out_per * pos_conv_w_->nb[2]);
            ggml_tensor * cg = ggml_conv_1d(ctx0, wg, xin, 1, hp.pos_conv_kernel / 2, 1);
            pos = (g == 0) ? cg : ggml_concat(ctx0, pos, cg, 1);
        }

        pos = add_channel_bias(ctx0, pos, pos_conv_b_);

        // "Same" padding for an even kernel leaves one extra frame; drop it.
        if (hp.pos_conv_kernel % 2 == 0) {
            pos = ggml_view_2d(ctx0, pos, n_frames, n_state, pos->nb[1], 0);
        }
        pos = ggml_gelu_erf(ctx0, pos);
        pos = ggml_cont(ctx0, ggml_transpose(ctx0, pos));           // [n_state, n_frames]

        cur = ggml_add(ctx0, cur, pos);
    }

    // --- transformer encoder (stable layer norm / pre-LN) -------------------
    for (int il = 0; il < hp.n_layers; ++il) {
        const auto & l = layers_[il];
        ggml_tensor * residual = cur;

        ggml_tensor * h = layer_norm(ctx0, cur, l.attn_norm_w, l.attn_norm_b, hp.layer_norm_eps);

        ggml_tensor * Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_q_w, h), l.attn_q_b);
        ggml_tensor * Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_k_w, h), l.attn_k_b);
        ggml_tensor * Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_v_w, h), l.attn_v_b);

        ggml_tensor * Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_frames), 0, 2, 1, 3);
        ggml_tensor * K = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Kcur, head_dim, n_head, n_frames), 0, 2, 1, 3);

        ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        // Bidirectional encoder: every frame attends to every other, so no mask.
        KQ = ggml_soft_max_ext(ctx0, KQ, nullptr, kq_scale, 0.0f);

        ggml_tensor * V = ggml_cont(ctx0, ggml_permute(ctx0,
                            ggml_reshape_3d(ctx0, Vcur, head_dim, n_head, n_frames), 1, 2, 0, 3));

        ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        h = ggml_cont_2d(ctx0, ggml_permute(ctx0, KQV, 0, 2, 1, 3), n_state, n_frames);

        h = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_out_w, h), l.attn_out_b);
        cur = ggml_add(ctx0, residual, h);

        ggml_tensor * ff = layer_norm(ctx0, cur, l.ffn_norm_w, l.ffn_norm_b, hp.layer_norm_eps);
        ff = ggml_add(ctx0, ggml_mul_mat(ctx0, l.ffn_up_w, ff), l.ffn_up_b);
        ff = ggml_gelu_erf(ctx0, ff);
        ff = ggml_add(ctx0, ggml_mul_mat(ctx0, l.ffn_down_w, ff), l.ffn_down_b);

        cur = ggml_add(ctx0, cur, ff);
    }

    cur = layer_norm(ctx0, cur, enc_norm_w_, enc_norm_b_, hp.layer_norm_eps);

    // --- CTC head -----------------------------------------------------------
    // Raw logits: the log-softmax is done on the host, where subtracting the row
    // max keeps it stable even for very confident frames.
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, lm_head_w_, cur), lm_head_b_);

    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

bool CtcAligner::compute_emissions(const float * samples, int n_samples,
                                    std::vector<float> & emissions, int & n_frames) {
    // wav2vec2 was trained on zero-mean unit-variance waveforms.
    std::vector<float> norm(samples, samples + n_samples);
    {
        double mean = 0.0;
        for (float v : norm) mean += v;
        mean /= std::max(1, n_samples);
        double var = 0.0;
        for (float v : norm) var += (v - mean) * (v - mean);
        var /= std::max(1, n_samples);
        const float inv = 1.0f / std::sqrt((float) var + 1e-7f);
        for (float & v : norm) v = (float) ((v - mean) * inv);
    }

    ggml_init_params gparams = { compute_meta_.size(), compute_meta_.data(), true };
    ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        error_msg_ = "Failed to create the CTC compute context";
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_samples, 1);
    ggml_set_name(input, "input");
    ggml_set_input(input);

    ggml_cgraph * gf = build_graph(ctx0, input);

    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        error_msg_ = "Failed to allocate the CTC compute buffer";
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(input, norm.data(), 0, norm.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "CTC forward pass failed";
        ggml_backend_sched_reset(sched_);
        ggml_free(ctx0);
        return false;
    }

    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    const int64_t n_vocab = logits->ne[0];
    n_frames = (int) logits->ne[1];

    emissions.resize((size_t) n_frames * n_vocab);
    ggml_backend_tensor_get(logits, emissions.data(), 0, emissions.size() * sizeof(float));

    ggml_backend_sched_reset(sched_);
    ggml_free(ctx0);

    // Stable log-softmax per frame.
    for (int t = 0; t < n_frames; ++t) {
        float * row = emissions.data() + (size_t) t * n_vocab;
        float max_v = row[0];
        for (int64_t v = 1; v < n_vocab; ++v) max_v = std::max(max_v, row[v]);
        double sum = 0.0;
        for (int64_t v = 0; v < n_vocab; ++v) sum += std::exp((double) row[v] - max_v);
        const float log_sum = max_v + (float) std::log(sum);
        for (int64_t v = 0; v < n_vocab; ++v) row[v] -= log_sum;
    }

    return true;
}

alignment_result CtcAligner::align(const float * samples, int n_samples,
                                    const std::vector<std::string> & chars) {
    alignment_result result;

    if (!model_loaded_) {
        result.error_msg = "CTC model not loaded";
        return result;
    }
    if (chars.empty()) {
        result.success = true;
        return result;
    }

    std::vector<float> emissions;
    int n_frames = 0;
    if (!compute_emissions(samples, n_samples, emissions, n_frames)) {
        result.error_msg = error_msg_;
        return result;
    }

    const int64_t n_vocab = hparams_.vocab_size;
    const int32_t blank_id = hparams_.blank_id;

    // Characters the acoustic model has no symbol for (typically punctuation) are
    // matched against a synthetic "any non-blank" column, so they still take a
    // position in the path instead of derailing the alignment.
    bool needs_wildcard = false;
    for (const auto & c : chars) {
        if (!token_to_id_.count(c)) { needs_wildcard = true; break; }
    }

    int64_t n_cols = n_vocab;
    if (needs_wildcard) {
        n_cols = n_vocab + 1;
        std::vector<float> extended((size_t) n_frames * n_cols);
        for (int t = 0; t < n_frames; ++t) {
            const float * src = emissions.data() + (size_t) t * n_vocab;
            float * dst = extended.data() + (size_t) t * n_cols;
            std::copy(src, src + n_vocab, dst);
            float best = -std::numeric_limits<float>::infinity();
            for (int64_t v = 0; v < n_vocab; ++v) {
                if (v != blank_id) best = std::max(best, src[v]);
            }
            dst[n_vocab] = best;
        }
        emissions.swap(extended);
    }
    const int32_t wildcard_id = (int32_t) n_vocab;

    std::vector<int32_t> tokens;
    tokens.reserve(chars.size());
    for (const auto & c : chars) {
        auto it = token_to_id_.find(c);
        tokens.push_back(it != token_to_id_.end() ? it->second : wildcard_id);
    }

    const int T = n_frames;
    const int N = (int) tokens.size();
    if (T < N) {
        result.error_msg = "CTC alignment needs at least one frame per character (" +
                            std::to_string(T) + " frames for " + std::to_string(N) + " characters)";
        return result;
    }

    auto emis = [&](int t, int32_t v) { return emissions[(size_t) t * n_cols + v]; };

    // Viterbi trellis over (frame, characters consumed). Every frame is spent
    // either staying on the current character (emitting blank) or advancing to
    // the next one, so the resulting path is monotonic by construction.
    const size_t stride = (size_t) N + 1;
    std::vector<float> trellis((size_t) (T + 1) * stride);
    auto tr = [&](int t, int j) -> float & { return trellis[(size_t) t * stride + j]; };

    const float NEG_INF = -std::numeric_limits<float>::infinity();
    const float POS_INF = std::numeric_limits<float>::infinity();

    tr(0, 0) = 0.0f;
    for (int j = 1; j <= N; ++j) tr(0, j) = NEG_INF;
    float running = 0.0f;
    for (int t = 0; t < T; ++t) {
        running += emis(t, blank_id);
        tr(t + 1, 0) = running;
    }
    // Boundary condition that forces the backtrack to terminate at the start.
    for (int t = std::max(0, T + 1 - N); t <= T; ++t) tr(t, 0) = POS_INF;

    for (int t = 0; t < T; ++t) {
        const float blank_score = emis(t, blank_id);
        for (int j = 1; j <= N; ++j) {
            const float stayed  = tr(t, j) + blank_score;
            const float changed = tr(t, j - 1) + emis(t, tokens[j - 1]);
            tr(t + 1, j) = std::max(stayed, changed);
        }
    }

    int j = N;
    int t_start = 0;
    float best = NEG_INF;
    for (int t = 0; t <= T; ++t) {
        if (tr(t, j) > best) { best = tr(t, j); t_start = t; }
    }

    std::vector<int> first_frame(N, -1), last_frame(N, -1);
    bool completed = false;
    for (int t = t_start; t > 0; --t) {
        const float stayed  = tr(t - 1, j) + emis(t - 1, blank_id);
        const float changed = tr(t - 1, j - 1) + emis(t - 1, tokens[j - 1]);

        const int idx = j - 1;
        const int frame = t - 1;
        if (last_frame[idx] < 0) last_frame[idx] = frame;
        first_frame[idx] = frame;

        if (changed > stayed) {
            if (--j == 0) { completed = true; break; }
        }
    }

    if (!completed) {
        result.error_msg = "CTC backtrack failed to reach the start of the transcript";
        return result;
    }

    // Frame duration measured against this buffer: the final frame may cover
    // fewer samples than the rest, so derive it from the actual lengths.
    const double ratio = ((double) n_samples / hparams_.sample_rate) / (double) T;

    result.words.reserve(chars.size());
    double prev_end = 0.0;
    for (int i = 0; i < N; ++i) {
        aligned_word w;
        w.word = chars[i];
        if (first_frame[i] < 0) {
            // Never visited (possible when the path skips a character entirely):
            // pin it to where the previous one ended so ordering stays intact.
            w.start = (float) prev_end;
            w.end   = (float) prev_end;
        } else {
            w.start = (float) (first_frame[i] * ratio);
            w.end   = (float) ((last_frame[i] + 1) * ratio);
            prev_end = w.end;
        }
        result.words.push_back(w);
    }

    result.success = true;
    return result;
}

} // namespace qwen3_asr
