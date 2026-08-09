#include "vad.h"

#include <ggml-cpu.h>
#include <ggml-alloc.h>

#include <algorithm>
#include <cstdint>
#include <fstream>

namespace qwen3_asr {

namespace {

// Exact tensor names used by the ggml Silero VAD model files (ported from
// whisper.cpp's whisper-vad implementation) - these must match the file
// verbatim, they are not something we get to choose.
const char * TENSOR_STFT_BASIS      = "_model.stft.forward_basis_buffer";
const char * TENSOR_ENC_WEIGHT[4]   = {
    "_model.encoder.0.reparam_conv.weight",
    "_model.encoder.1.reparam_conv.weight",
    "_model.encoder.2.reparam_conv.weight",
    "_model.encoder.3.reparam_conv.weight",
};
const char * TENSOR_ENC_BIAS[4]     = {
    "_model.encoder.0.reparam_conv.bias",
    "_model.encoder.1.reparam_conv.bias",
    "_model.encoder.2.reparam_conv.bias",
    "_model.encoder.3.reparam_conv.bias",
};
const char * TENSOR_LSTM_WEIGHT_IH  = "_model.decoder.rnn.weight_ih";
const char * TENSOR_LSTM_WEIGHT_HH  = "_model.decoder.rnn.weight_hh";
const char * TENSOR_LSTM_BIAS_IH    = "_model.decoder.rnn.bias_ih";
const char * TENSOR_LSTM_BIAS_HH    = "_model.decoder.rnn.bias_hh";
const char * TENSOR_FINAL_CONV_W    = "_model.decoder.decoder.2.weight";
const char * TENSOR_FINAL_CONV_B    = "_model.decoder.decoder.2.bias";

constexpr int VAD_SAMPLE_RATE = 16000;

template <typename T>
bool read_safe(std::ifstream & fin, T & dest) {
    fin.read(reinterpret_cast<char *>(&dest), sizeof(T));
    return fin.good();
}

} // namespace

VoiceActivityDetector::VoiceActivityDetector() = default;

VoiceActivityDetector::~VoiceActivityDetector() {
    if (state_buffer_) ggml_backend_buffer_free(state_buffer_);
    if (state_ctx_) ggml_free(state_ctx_);
    if (model_buffer_) ggml_backend_buffer_free(model_buffer_);
    if (model_ctx_) ggml_free(model_ctx_);
    if (backend_cpu_) ggml_backend_free(backend_cpu_);
}

void VoiceActivityDetector::set_n_threads(int n_threads) {
    n_threads_ = std::max(1, n_threads);
    if (backend_cpu_) {
        ggml_backend_cpu_set_n_threads(backend_cpu_, n_threads_);
    }
}

bool VoiceActivityDetector::create_tensors() {
    const auto & hp = hparams_;
    const size_t n_tensors = 1 /*stft*/ + 4 * 2 /*encoder*/ + 4 /*lstm*/ + 2 /*final conv*/;

    ggml_init_params params = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    model_ctx_ = ggml_init(params);
    if (!model_ctx_) {
        error_msg_ = "Failed to create VAD model ggml context";
        return false;
    }

    auto reg = [&](ggml_tensor * t, const char * name) {
        ggml_set_name(t, name);
        tensors_[name] = t;
        return t;
    };

    stft_forward_basis_ = reg(ggml_new_tensor_3d(model_ctx_, GGML_TYPE_F16, 256, 1, 258), TENSOR_STFT_BASIS);

    ggml_tensor ** enc_w[4] = { &encoder_0_weight_, &encoder_1_weight_, &encoder_2_weight_, &encoder_3_weight_ };
    ggml_tensor ** enc_b[4] = { &encoder_0_bias_,   &encoder_1_bias_,   &encoder_2_bias_,   &encoder_3_bias_   };
    for (int i = 0; i < 4; ++i) {
        *enc_w[i] = reg(ggml_new_tensor_3d(model_ctx_, GGML_TYPE_F16,
            hp.kernel_sizes[i], hp.encoder_in_channels[i], hp.encoder_out_channels[i]), TENSOR_ENC_WEIGHT[i]);
        *enc_b[i] = reg(ggml_new_tensor_1d(model_ctx_, GGML_TYPE_F32, hp.encoder_out_channels[i]), TENSOR_ENC_BIAS[i]);
    }

    const int hstate_dim = hp.lstm_hidden_size * 4;
    lstm_ih_weight_ = reg(ggml_new_tensor_2d(model_ctx_, GGML_TYPE_F32, hp.lstm_hidden_size, hstate_dim), TENSOR_LSTM_WEIGHT_IH);
    lstm_ih_bias_   = reg(ggml_new_tensor_1d(model_ctx_, GGML_TYPE_F32, hstate_dim), TENSOR_LSTM_BIAS_IH);
    lstm_hh_weight_ = reg(ggml_new_tensor_2d(model_ctx_, GGML_TYPE_F32, hp.lstm_hidden_size, hstate_dim), TENSOR_LSTM_WEIGHT_HH);
    lstm_hh_bias_   = reg(ggml_new_tensor_1d(model_ctx_, GGML_TYPE_F32, hstate_dim), TENSOR_LSTM_BIAS_HH);

    final_conv_weight_ = reg(ggml_new_tensor_2d(model_ctx_, GGML_TYPE_F16, hp.final_conv_in, hp.final_conv_out), TENSOR_FINAL_CONV_W);
    final_conv_bias_   = reg(ggml_new_tensor_1d(model_ctx_, GGML_TYPE_F32, hp.final_conv_out), TENSOR_FINAL_CONV_B);

    model_buffer_ = ggml_backend_alloc_ctx_tensors(model_ctx_, backend_cpu_);
    if (!model_buffer_) {
        error_msg_ = "Failed to allocate VAD model tensors";
        return false;
    }

    return true;
}

bool VoiceActivityDetector::load_model(const std::string & model_path) {
    std::ifstream fin(model_path, std::ios::binary);
    if (!fin) {
        error_msg_ = "Failed to open VAD model: " + model_path;
        return false;
    }

    uint32_t magic = 0;
    read_safe(fin, magic);
    if (magic != GGML_FILE_MAGIC) {
        error_msg_ = "Invalid VAD model file (bad magic): " + model_path;
        return false;
    }

    // The header carries a model type string, a semantic version and a context
    // size that this implementation doesn't need; they still have to be consumed
    // to stay aligned with the byte stream.
    int32_t str_len = 0;
    read_safe(fin, str_len);
    std::vector<char> type_buf(str_len);
    fin.read(type_buf.data(), str_len);

    int32_t major = 0, minor = 0, patch = 0;
    read_safe(fin, major);
    read_safe(fin, minor);
    read_safe(fin, patch);

    int32_t n_context = 0;
    read_safe(fin, n_window_);
    read_safe(fin, n_context);

    int32_t n_encoder_layers = 0;
    read_safe(fin, n_encoder_layers);
    if (n_encoder_layers != vad_hparams::n_encoder_layers) {
        error_msg_ = "Unsupported VAD model: expected 4 encoder layers, got " + std::to_string(n_encoder_layers);
        return false;
    }
    for (int i = 0; i < n_encoder_layers; ++i) {
        read_safe(fin, hparams_.encoder_in_channels[i]);
        read_safe(fin, hparams_.encoder_out_channels[i]);
        read_safe(fin, hparams_.kernel_sizes[i]);
    }
    read_safe(fin, hparams_.lstm_input_size);
    read_safe(fin, hparams_.lstm_hidden_size);
    read_safe(fin, hparams_.final_conv_in);
    read_safe(fin, hparams_.final_conv_out);

    backend_cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend_cpu_) {
        error_msg_ = "Failed to initialize CPU backend for VAD";
        return false;
    }

    if (!create_tensors()) {
        return false;
    }

    // Weight stream: [n_dims, name_len, ttype, ne[0..n_dims), name, raw data] repeated until EOF.
    std::vector<char> read_buf;
    size_t total_size = 0;
    int n_loaded = 0;
    while (fin.peek() != EOF && fin.good()) {
        int32_t n_dims = 0, length = 0, ttype = 0;
        read_safe(fin, n_dims);
        read_safe(fin, length);
        read_safe(fin, ttype);
        if (!fin.good()) break;

        if (n_dims < 0 || n_dims > 4) {
            error_msg_ = "Invalid VAD tensor dims in model file";
            return false;
        }

        int64_t nelements = 1;
        int64_t ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < n_dims; ++i) {
            int32_t d = 0;
            read_safe(fin, d);
            ne[i] = d;
            nelements *= d;
        }

        std::string name(length, '\0');
        fin.read(name.data(), length);

        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            error_msg_ = "Unknown VAD tensor '" + name + "' in model file";
            return false;
        }
        ggml_tensor * tensor = it->second;

        if (ggml_nelements(tensor) != nelements ||
            tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2]) {
            error_msg_ = "VAD tensor '" + name + "' has unexpected shape in model file";
            return false;
        }

        read_buf.resize(ggml_nbytes(tensor));
        fin.read(read_buf.data(), read_buf.size());
        ggml_backend_tensor_set(tensor, read_buf.data(), 0, read_buf.size());

        total_size += read_buf.size();
        n_loaded++;
    }

    if (n_loaded != (int) tensors_.size()) {
        error_msg_ = "VAD model file is missing tensors (loaded " + std::to_string(n_loaded) +
                     ", expected " + std::to_string(tensors_.size()) + ")";
        return false;
    }

    // Persistent LSTM state (h, c) - lives outside the per-frame compute graph
    // context so it survives across the many small graph builds/frees below.
    {
        ggml_init_params sparams = { 2 * ggml_tensor_overhead(), nullptr, true };
        state_ctx_ = ggml_init(sparams);
        h_state_ = ggml_new_tensor_1d(state_ctx_, GGML_TYPE_F32, hparams_.lstm_hidden_size);
        c_state_ = ggml_new_tensor_1d(state_ctx_, GGML_TYPE_F32, hparams_.lstm_hidden_size);
        ggml_set_name(h_state_, "h_state");
        ggml_set_name(c_state_, "c_state");
        state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_, backend_cpu_);
        if (!state_buffer_) {
            error_msg_ = "Failed to allocate VAD LSTM state";
            return false;
        }
    }

    set_n_threads(n_threads_);
    model_loaded_ = true;
    return true;
}

static ggml_tensor * build_stft(ggml_context * ctx0, ggml_tensor * stft_basis, ggml_tensor * frame, int lstm_input_size) {
    ggml_tensor * padded = ggml_pad_reflect_1d(ctx0, frame, 64, 64);
    ggml_tensor * stft = ggml_conv_1d(ctx0, stft_basis, padded, lstm_input_size, 0, 1);

    int cutoff = stft_basis->ne[2] / 2;
    ggml_tensor * real_part = ggml_view_2d(ctx0, stft, 4, cutoff, stft->nb[1], 0);
    ggml_tensor * img_part  = ggml_view_2d(ctx0, stft, 4, cutoff, stft->nb[1], (size_t) cutoff * stft->nb[1]);

    ggml_tensor * mag_sq = ggml_add(ctx0, ggml_mul(ctx0, real_part, real_part), ggml_mul(ctx0, img_part, img_part));
    return ggml_sqrt(ctx0, mag_sq);
}

static ggml_tensor * build_encoder(ggml_context * ctx0, const vad_hparams & hp,
        ggml_tensor * enc_w[4], ggml_tensor * enc_b[4], ggml_tensor * cur) {
    // Strides aren't stored in the model file - they're fixed by the Silero v5
    // architecture, where layers 1 and 2 halve the time resolution.
    const int strides[vad_hparams::n_encoder_layers] = {1, 2, 2, 1};
    for (int i = 0; i < vad_hparams::n_encoder_layers; ++i) {
        cur = ggml_conv_1d(ctx0, enc_w[i], cur, strides[i], 1, 1);
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, enc_b[i], 1, hp.encoder_out_channels[i], 1));
        cur = ggml_relu(ctx0, cur);
    }
    return cur;
}

static ggml_tensor * build_lstm(ggml_context * ctx0, ggml_cgraph * gf,
        ggml_tensor * lstm_ih_w, ggml_tensor * lstm_ih_b,
        ggml_tensor * lstm_hh_w, ggml_tensor * lstm_hh_b,
        ggml_tensor * h_state, ggml_tensor * c_state, int hdim, ggml_tensor * cur) {
    ggml_tensor * x_t = ggml_transpose(ctx0, cur);

    ggml_tensor * inp_gate = ggml_add(ctx0, ggml_mul_mat(ctx0, lstm_ih_w, x_t), lstm_ih_b);
    ggml_tensor * hid_gate = ggml_add(ctx0, ggml_mul_mat(ctx0, lstm_hh_w, h_state), lstm_hh_b);
    ggml_tensor * out_gate = ggml_add(ctx0, inp_gate, hid_gate);

    const size_t hdim_size = ggml_row_size(out_gate->type, hdim);
    ggml_tensor * i_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gate, hdim, 0 * hdim_size));
    ggml_tensor * f_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gate, hdim, 1 * hdim_size));
    ggml_tensor * g_t = ggml_tanh(ctx0,   ggml_view_1d(ctx0, out_gate, hdim, 2 * hdim_size));
    ggml_tensor * o_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gate, hdim, 3 * hdim_size));

    ggml_tensor * c_out = ggml_add(ctx0, ggml_mul(ctx0, f_t, c_state), ggml_mul(ctx0, i_t, g_t));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, c_out, c_state));

    ggml_tensor * out = ggml_mul(ctx0, o_t, ggml_tanh(ctx0, c_out));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, out, h_state));

    return out;
}

std::vector<float> VoiceActivityDetector::compute_probs(const float * samples, int n_samples) {
    ggml_backend_buffer_clear(state_buffer_, 0);

    int n_chunks = n_samples / n_window_;
    if (n_samples % n_window_ != 0) n_chunks += 1;

    std::vector<float> probs(n_chunks, 0.0f);
    std::vector<float> window(n_window_, 0.0f);

    ggml_tensor * enc_w[4] = { encoder_0_weight_, encoder_1_weight_, encoder_2_weight_, encoder_3_weight_ };
    ggml_tensor * enc_b[4] = { encoder_0_bias_,   encoder_1_bias_,   encoder_2_bias_,   encoder_3_bias_   };

    for (int i = 0; i < n_chunks; ++i) {
        const int idx_start = i * n_window_;
        const int idx_end = std::min(idx_start + n_window_, n_samples);
        const int chunk_len = idx_end - idx_start;

        std::fill(window.begin(), window.end(), 0.0f);
        std::copy(samples + idx_start, samples + idx_start + chunk_len, window.begin());

        constexpr size_t MAX_NODES = 256;
        std::vector<uint8_t> meta(ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead());
        ggml_init_params gparams = { meta.size(), meta.data(), true };
        ggml_context * ctx0 = ggml_init(gparams);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx0, MAX_NODES, false);

        ggml_tensor * frame = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_window_, 1);
        ggml_set_name(frame, "frame");
        ggml_set_input(frame);

        ggml_tensor * cur = build_stft(ctx0, stft_forward_basis_, frame, hparams_.lstm_input_size);
        cur = build_encoder(ctx0, hparams_, enc_w, enc_b, cur);
        cur = ggml_view_2d(ctx0, cur, 1, 128, cur->nb[1], 0);
        cur = build_lstm(ctx0, gf, lstm_ih_weight_, lstm_ih_bias_, lstm_hh_weight_, lstm_hh_bias_,
                          h_state_, c_state_, hparams_.lstm_hidden_size, cur);
        cur = ggml_relu(ctx0, cur);
        cur = ggml_conv_1d(ctx0, final_conv_weight_, cur, 1, 0, 1);
        cur = ggml_add(ctx0, cur, final_conv_bias_);
        cur = ggml_sigmoid(ctx0, cur);
        ggml_set_name(cur, "prob");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_cpu_));
        ggml_gallocr_alloc_graph(galloc, gf);

        ggml_backend_tensor_set(frame, window.data(), 0, ggml_nelements(frame) * sizeof(float));
        ggml_backend_graph_compute(backend_cpu_, gf);

        ggml_backend_tensor_get(cur, &probs[i], 0, sizeof(float));

        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
    }

    return probs;
}

std::vector<vad_segment> VoiceActivityDetector::detect_speech(const float * samples, int n_samples,
                                                                const vad_params & params) {
    std::vector<vad_segment> result;
    if (!model_loaded_) {
        error_msg_ = "VAD model not loaded";
        return result;
    }

    std::vector<float> probs = compute_probs(samples, n_samples);

    // Port of whisper.cpp's whisper_vad_segments_from_probs(): converts
    // per-frame speech probabilities into speech segments (start/end
    // samples), with hysteresis thresholding, minimum speech/silence
    // duration filtering, max-duration force-splitting, small-gap merging
    // and edge padding.
    const int sample_rate = VAD_SAMPLE_RATE;
    const int min_silence_samples = sample_rate * params.min_silence_duration_ms / 1000;
    const int min_speech_samples  = sample_rate * params.min_speech_duration_ms / 1000;
    const int speech_pad_samples  = sample_rate * params.speech_pad_ms / 1000;
    const int audio_length_samples = (int) probs.size() * n_window_;

    int64_t max_speech_samples;
    if (params.max_speech_duration_s > 100000.0f) {
        max_speech_samples = INT32_MAX / 2;
    } else {
        max_speech_samples = (int64_t) sample_rate * (int64_t) params.max_speech_duration_s -
                              n_window_ - 2 * speech_pad_samples;
        if (max_speech_samples < 0) max_speech_samples = INT32_MAX / 2;
    }
    const int min_silence_samples_at_max_speech = sample_rate * 98 / 1000;

    float neg_threshold = params.threshold - 0.15f;
    if (neg_threshold < 0.01f) neg_threshold = 0.01f;

    struct speech_t { int start; int end; };
    std::vector<speech_t> speeches;

    bool is_speech_segment = false;
    int temp_end = 0, prev_end = 0, next_start = 0, curr_speech_start = 0;
    bool has_curr_speech = false;

    for (size_t i = 0; i < probs.size(); ++i) {
        float p = probs[i];
        int curr_sample = n_window_ * (int) i;

        if (p >= params.threshold && temp_end) {
            temp_end = 0;
            if (next_start < prev_end) next_start = curr_sample;
        }

        if (p >= params.threshold && !is_speech_segment) {
            is_speech_segment = true;
            curr_speech_start = curr_sample;
            has_curr_speech = true;
            continue;
        }

        if (is_speech_segment && (curr_sample - curr_speech_start) > max_speech_samples) {
            if (prev_end) {
                speeches.push_back({curr_speech_start, prev_end});
                has_curr_speech = true;
                if (next_start < prev_end) {
                    is_speech_segment = false;
                    has_curr_speech = false;
                } else {
                    curr_speech_start = next_start;
                }
                prev_end = next_start = temp_end = 0;
            } else {
                speeches.push_back({curr_speech_start, curr_sample});
                prev_end = next_start = temp_end = 0;
                is_speech_segment = false;
                has_curr_speech = false;
                continue;
            }
        }

        if (p < neg_threshold && is_speech_segment) {
            if (!temp_end) temp_end = curr_sample;
            if ((curr_sample - temp_end) > min_silence_samples_at_max_speech) {
                prev_end = temp_end;
            }
            if ((curr_sample - temp_end) < min_silence_samples) {
                continue;
            }
            if ((temp_end - curr_speech_start) > min_speech_samples) {
                speeches.push_back({curr_speech_start, temp_end});
            }
            prev_end = next_start = temp_end = 0;
            is_speech_segment = false;
            has_curr_speech = false;
            continue;
        }
    }

    if (has_curr_speech && (audio_length_samples - curr_speech_start) > min_speech_samples) {
        speeches.push_back({curr_speech_start, audio_length_samples});
    }

    // Merge adjacent segments with a small gap between them.
    if (speeches.size() > 1) {
        const int max_merge_gap_samples = sample_rate * 200 / 1000;
        for (int i = 0; i < (int) speeches.size() - 1; ++i) {
            if (speeches[i + 1].start - speeches[i].end < max_merge_gap_samples) {
                speeches[i].end = speeches[i + 1].end;
                speeches.erase(speeches.begin() + i + 1);
                --i;
            }
        }
    }

    for (int i = 0; i < (int) speeches.size(); ++i) {
        if (speeches[i].end - speeches[i].start < min_speech_samples) {
            speeches.erase(speeches.begin() + i);
            --i;
        }
    }

    // Apply edge padding.
    for (int i = 0; i < (int) speeches.size(); ++i) {
        if (i == 0) {
            speeches[i].start = (speeches[i].start > speech_pad_samples) ? (speeches[i].start - speech_pad_samples) : 0;
        }
        if (i < (int) speeches.size() - 1) {
            int silence_duration = speeches[i + 1].start - speeches[i].end;
            if (silence_duration < 2 * speech_pad_samples) {
                speeches[i].end += silence_duration / 2;
                speeches[i + 1].start = (speeches[i + 1].start > silence_duration / 2) ?
                    (speeches[i + 1].start - silence_duration / 2) : 0;
            } else {
                speeches[i].end = (speeches[i].end + speech_pad_samples < audio_length_samples) ?
                    (speeches[i].end + speech_pad_samples) : audio_length_samples;
                speeches[i + 1].start = (speeches[i + 1].start > speech_pad_samples) ?
                    (speeches[i + 1].start - speech_pad_samples) : 0;
            }
        } else {
            speeches[i].end = (speeches[i].end + speech_pad_samples < audio_length_samples) ?
                (speeches[i].end + speech_pad_samples) : audio_length_samples;
        }

        result.push_back({(double) speeches[i].start / sample_rate, (double) speeches[i].end / sample_rate});
    }

    return result;
}

} // namespace qwen3_asr
