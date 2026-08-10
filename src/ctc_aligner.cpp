#include "ctc_aligner.h"

#include <ggml-cpu.h>
#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <cstring>
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

bool starts_with(const std::string & s, const char * prefix) {
    const size_t n = strlen(prefix);
    return s.size() >= n && s.compare(0, n, prefix) == 0;
}

// Maps a tensor name onto the single naming scheme the rest of this file uses.
// Two converters produce this same architecture with identical shapes but
// different names - our own scripts/convert_wav2vec2_ctc_to_gguf.py, and the
// third-party GGUFs Subtitle Edit already downloads for its wav2vec2 aligners -
// so accepting both means users can reuse a model they may already have.
std::string canonical_tensor_name(const std::string & name) {
    // cnn.<i>.conv.* / cnn.<i>.norm.* -> conv.<i>.*
    if (starts_with(name, "cnn.")) {
        const size_t dot = name.find('.', 4);
        if (dot != std::string::npos) {
            const std::string idx  = name.substr(4, dot - 4);
            const std::string rest = name.substr(dot + 1);
            if (starts_with(rest, "conv.")) return "conv." + idx + "." + rest.substr(5);
            if (starts_with(rest, "norm.")) return "conv." + idx + ".norm." + rest.substr(5);
        }
        return name;
    }

    if (starts_with(name, "feat_proj.ln.")) {
        return "feat_proj.norm." + name.substr(13);
    }
    if (starts_with(name, "enc.ln.")) {
        return "encoder.norm." + name.substr(7);
    }

    // enc.<i>.* -> blk.<i>.*
    if (starts_with(name, "enc.")) {
        const size_t dot = name.find('.', 4);
        if (dot == std::string::npos) return name;
        const std::string idx  = name.substr(4, dot - 4);
        const std::string rest = name.substr(dot + 1);
        const std::string p    = "blk." + idx + ".";

        if (starts_with(rest, "ln1."))          return p + "attn_norm." + rest.substr(4);
        if (starts_with(rest, "ln2."))          return p + "ffn_norm."  + rest.substr(4);
        if (starts_with(rest, "attn.q."))       return p + "attn_q."    + rest.substr(7);
        if (starts_with(rest, "attn.k."))       return p + "attn_k."    + rest.substr(7);
        if (starts_with(rest, "attn.v."))       return p + "attn_v."    + rest.substr(7);
        if (starts_with(rest, "attn.out."))     return p + "attn_out."  + rest.substr(9);
        if (starts_with(rest, "ffn.fc1."))      return p + "ffn_up."    + rest.substr(8);
        if (starts_with(rest, "ffn.fc2."))      return p + "ffn_down."  + rest.substr(8);
        return name;
    }

    return name;
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
    // Accept our own key names and the ones used by the third-party wav2vec2
    // GGUFs (which keep the HuggingFace config spellings under a "wav2vec2."
    // prefix), so either file loads without conversion.
    auto find = [&](const char * a, const char * b) -> int64_t {
        int64_t idx = gguf_find_key(ctx, a);
        return idx >= 0 ? idx : gguf_find_key(ctx, b);
    };
    auto get_u32 = [&](const char * a, const char * b, int32_t def) -> int32_t {
        int64_t idx = find(a, b);
        return idx < 0 ? def : (int32_t) gguf_get_val_u32(ctx, idx);
    };
    auto get_f32 = [&](const char * a, const char * b, float def) -> float {
        int64_t idx = find(a, b);
        return idx < 0 ? def : gguf_get_val_f32(ctx, idx);
    };

    auto & hp = hparams_;
    hp.hidden_size     = get_u32("wav2vec2-ctc.embedding_length", "wav2vec2.hidden_size", 1024);
    hp.n_layers        = get_u32("wav2vec2-ctc.block_count", "wav2vec2.num_hidden_layers", 24);
    hp.n_heads         = get_u32("wav2vec2-ctc.attention.head_count", "wav2vec2.num_attention_heads", 16);
    hp.ffn_dim         = get_u32("wav2vec2-ctc.feed_forward_length", "wav2vec2.intermediate_size", 4096);
    hp.layer_norm_eps  = get_f32("wav2vec2-ctc.attention.layer_norm_epsilon", "wav2vec2.layer_norm_eps", 1e-5f);
    hp.vocab_size      = get_u32("wav2vec2-ctc.vocab_size", "wav2vec2.vocab_size", 2341);
    hp.pos_conv_kernel = get_u32("wav2vec2-ctc.pos_conv.kernel", "wav2vec2.num_conv_pos_embeddings", 128);
    hp.pos_conv_groups = get_u32("wav2vec2-ctc.pos_conv.groups", "wav2vec2.num_conv_pos_embedding_groups", 16);
    hp.sample_rate     = get_u32("wav2vec2-ctc.sample_rate", "wav2vec2.sample_rate", 16000);
    hp.blank_id        = get_u32("wav2vec2-ctc.blank_token_id", "wav2vec2.pad_token_id", 0);

    // This implementation only covers the layer-norm / stable-layer-norm variant
    // (the large XLSR family); the group-norm variant places its normalisation
    // differently and would silently produce wrong emissions.
    int64_t stable_idx = gguf_find_key(ctx, "wav2vec2.do_stable_layer_norm");
    if (stable_idx >= 0 && gguf_get_val_u32(ctx, stable_idx) == 0) {
        error_msg_ = "Unsupported CTC model: only the stable-layer-norm wav2vec2 variant is implemented";
        return false;
    }

    // Feature extractor normalisation variant. Only our own converter writes
    // this key, so its absence (third-party GGUFs) defaults to the original
    // layer-norm behaviour those files were always assumed to have.
    int64_t norm_idx = gguf_find_key(ctx, "wav2vec2-ctc.feat_extract_norm");
    if (norm_idx >= 0 && gguf_get_kv_type(ctx, norm_idx) == GGUF_TYPE_STRING) {
        hp.feat_extract_group_norm = (std::string(gguf_get_val_str(ctx, norm_idx)) == "group");
    }

    // Conv stack: either three arrays (ours) or one key per layer (theirs).
    auto get_i32_array = [&](const char * key, std::vector<int32_t> & out) -> bool {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0 || gguf_get_kv_type(ctx, idx) != GGUF_TYPE_ARRAY) {
            return false;
        }
        const int32_t * data = (const int32_t *) gguf_get_arr_data(ctx, idx);
        out.assign(data, data + gguf_get_arr_n(ctx, idx));
        return true;
    };

    if (!get_i32_array("wav2vec2-ctc.conv.dim", hp.conv_dim) ||
        !get_i32_array("wav2vec2-ctc.conv.stride", hp.conv_stride) ||
        !get_i32_array("wav2vec2-ctc.conv.kernel", hp.conv_kernel)) {
        const int32_t n_conv = get_u32("wav2vec2-ctc.conv.count", "wav2vec2.num_feat_extract_layers", 0);
        if (n_conv <= 0) {
            error_msg_ = "CTC model does not describe its convolutional feature encoder";
            return false;
        }
        hp.conv_dim.clear(); hp.conv_stride.clear(); hp.conv_kernel.clear();
        for (int32_t i = 0; i < n_conv; ++i) {
            const std::string s = std::to_string(i);
            int64_t d = gguf_find_key(ctx, ("wav2vec2.conv_dim_" + s).c_str());
            int64_t k = gguf_find_key(ctx, ("wav2vec2.conv_kernel_" + s).c_str());
            int64_t t = gguf_find_key(ctx, ("wav2vec2.conv_stride_" + s).c_str());
            if (d < 0 || k < 0 || t < 0) {
                error_msg_ = "CTC model is missing conv parameters for layer " + s;
                return false;
            }
            hp.conv_dim.push_back((int32_t) gguf_get_val_u32(ctx, d));
            hp.conv_kernel.push_back((int32_t) gguf_get_val_u32(ctx, k));
            hp.conv_stride.push_back((int32_t) gguf_get_val_u32(ctx, t));
        }
    }
    if (hp.conv_dim.size() != hp.conv_stride.size() || hp.conv_dim.size() != hp.conv_kernel.size()) {
        error_msg_ = "CTC model conv parameters have mismatched lengths";
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

bool CtcAligner::create_tensors(gguf_context * ctx, ggml_context * meta_ctx) {
    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    ggml_init_params params = {
        /*.mem_size   =*/ (size_t) (n_tensors + 8) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    model_ctx_ = ggml_init(params);
    if (!model_ctx_) {
        error_msg_ = "Failed to create CTC model ggml context";
        return false;
    }

    // Mirror each tensor exactly as stored - same shape, same type - so a
    // quantised checkpoint needs no conversion here: ggml_mul_mat consumes the
    // quantised matrices directly, and the convolutions are stored unquantised.
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * src_name = gguf_get_tensor_name(ctx, i);
        ggml_tensor * meta = ggml_get_tensor(meta_ctx, src_name);
        if (!meta) {
            error_msg_ = std::string("Missing metadata for tensor '") + src_name + "'";
            return false;
        }
        const std::string name = canonical_tensor_name(src_name);
        if (tensors_.count(name)) {
            error_msg_ = "Duplicate tensor '" + name + "' in CTC model";
            return false;
        }
        ggml_tensor * t = ggml_dup_tensor(model_ctx_, meta);
        ggml_set_name(t, name.c_str());
        tensors_[name] = t;
    }

    model_buffer_ = ggml_backend_alloc_ctx_tensors(model_ctx_, backend_cpu_);
    if (!model_buffer_) {
        error_msg_ = "Failed to allocate CTC model tensors";
        return false;
    }

    auto need = [&](const std::string & n) -> ggml_tensor * {
        auto it = tensors_.find(n);
        if (it == tensors_.end()) {
            if (error_msg_.empty()) {
                error_msg_ = "CTC model is missing tensor '" + n + "'";
            }
            return nullptr;
        }
        return it->second;
    };
    auto maybe = [&](const std::string & n) -> ggml_tensor * {
        auto it = tensors_.find(n);
        return it == tensors_.end() ? nullptr : it->second;
    };

    const auto & hp = hparams_;

    conv_layers_.resize(hp.conv_dim.size());
    for (size_t i = 0; i < conv_layers_.size(); ++i) {
        const std::string p = "conv." + std::to_string(i);
        conv_layers_[i].w = need(p + ".weight");
        // Some checkpoints (e.g. conv_bias=False) omit the conv bias entirely.
        conv_layers_[i].b = maybe(p + ".bias");
        // Group-norm variant: only conv layer 0 has a norm; leave norm_w/norm_b
        // null for the rest so build_graph() skips normalisation for them.
        if (!hp.feat_extract_group_norm || i == 0) {
            conv_layers_[i].norm_w = need(p + ".norm.weight");
            conv_layers_[i].norm_b = need(p + ".norm.bias");
        }
    }

    feat_proj_norm_w_ = need("feat_proj.norm.weight");
    feat_proj_norm_b_ = need("feat_proj.norm.bias");
    feat_proj_w_      = need("feat_proj.weight");
    feat_proj_b_      = need("feat_proj.bias");

    pos_conv_w_ = need("pos_conv.weight");
    pos_conv_b_ = need("pos_conv.bias");

    enc_norm_w_ = need("encoder.norm.weight");
    enc_norm_b_ = need("encoder.norm.bias");

    layers_.resize(hp.n_layers);
    for (int32_t i = 0; i < hp.n_layers; ++i) {
        const std::string p = "blk." + std::to_string(i);
        auto & l = layers_[i];
        l.attn_norm_w = need(p + ".attn_norm.weight");
        l.attn_norm_b = need(p + ".attn_norm.bias");
        l.attn_q_w    = need(p + ".attn_q.weight");
        l.attn_q_b    = need(p + ".attn_q.bias");
        l.attn_k_w    = need(p + ".attn_k.weight");
        l.attn_k_b    = need(p + ".attn_k.bias");
        l.attn_v_w    = need(p + ".attn_v.weight");
        l.attn_v_b    = need(p + ".attn_v.bias");
        l.attn_out_w  = need(p + ".attn_out.weight");
        l.attn_out_b  = need(p + ".attn_out.bias");
        l.ffn_norm_w  = need(p + ".ffn_norm.weight");
        l.ffn_norm_b  = need(p + ".ffn_norm.bias");
        l.ffn_up_w    = need(p + ".ffn_up.weight");
        l.ffn_up_b    = need(p + ".ffn_up.bias");
        l.ffn_down_w  = need(p + ".ffn_down.weight");
        l.ffn_down_b  = need(p + ".ffn_down.bias");
    }

    lm_head_w_ = need("lm_head.weight");
    lm_head_b_ = need("lm_head.bias");

    return error_msg_.empty();
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
        const char * src_name = gguf_get_tensor_name(ctx, i);
        const std::string name = canonical_tensor_name(src_name);
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            error_msg_ = std::string("Unexpected tensor '") + src_name + "' in CTC model";
            return false;
        }
        ggml_tensor * dst = it->second;

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
    if (ok) ok = create_tensors(ctx, meta_ctx);
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
    // Layout note: ggml_conv_1d takes [time, channels] and returns [time, channels].
    //
    // Layer-norm variant: LayerNorm runs across channels, which ggml normalises
    // along ne[0], so each layer transposes into [channels, time] to normalise
    // and back again.
    //
    // Group-norm variant: only conv layer 0 is normalised, with GroupNorm
    // configured so num_groups == num_channels - i.e. each channel is
    // normalised independently across time, not across channels at a fixed
    // time step. That maps onto ggml_group_norm's own axis convention
    // (groups along ne[2], normalising over ne0*ne1*channels_per_group), so
    // the [time, channels] tensor is reshaped to 4D [time, 1, channels, 1]
    // instead of transposed. Layers 1+ get no normalisation at all, matching
    // Wav2Vec2NoLayerNormConvLayer.
    ggml_tensor * cur = input;  // [n_samples, 1]
    for (size_t i = 0; i < conv_layers_.size(); ++i) {
        const auto & cl = conv_layers_[i];
        cur = ggml_conv_1d(ctx0, cl.w, cur, hp.conv_stride[i], 0, 1);
        if (cl.b) {
            cur = add_channel_bias(ctx0, cur, cl.b);
        }

        if (hp.feat_extract_group_norm) {
            if (cl.norm_w) {
                const int64_t n_time = cur->ne[0];
                const int64_t n_ch   = cur->ne[1];
                ggml_tensor * cur4 = ggml_reshape_4d(ctx0, cur, n_time, 1, n_ch, 1);
                cur4 = ggml_group_norm(ctx0, cur4, (int) n_ch, hp.layer_norm_eps);
                ggml_tensor * w4 = ggml_reshape_4d(ctx0, cl.norm_w, 1, 1, n_ch, 1);
                ggml_tensor * b4 = ggml_reshape_4d(ctx0, cl.norm_b, 1, 1, n_ch, 1);
                cur4 = ggml_mul(ctx0, cur4, w4);
                cur4 = ggml_add(ctx0, cur4, b4);
                cur = ggml_reshape_2d(ctx0, cur4, n_time, n_ch);
            }
            cur = ggml_gelu_erf(ctx0, cur);
        } else {
            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));           // [channels, time]
            cur = layer_norm(ctx0, cur, cl.norm_w, cl.norm_b, hp.layer_norm_eps);
            cur = ggml_gelu_erf(ctx0, cur);

            if (i + 1 < conv_layers_.size()) {
                cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));       // back to [time, channels]
            }
        }
    }
    if (hp.feat_extract_group_norm) {
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));               // [time, channels] -> [channels, time]
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
