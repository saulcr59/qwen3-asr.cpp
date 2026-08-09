#pragma once

#include "qwen3asr_win_export.h"
#include "forced_aligner.h"   // aligned_word / alignment_result

#include <ggml.h>
#include <ggml-backend.h>
#include <gguf.h>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen3_asr {

struct ctc_hparams {
    int32_t hidden_size    = 1024;
    int32_t n_layers       = 24;
    int32_t n_heads        = 16;
    int32_t ffn_dim        = 4096;
    float   layer_norm_eps = 1e-5f;
    int32_t vocab_size     = 2341;

    std::vector<int32_t> conv_dim;
    std::vector<int32_t> conv_stride;
    std::vector<int32_t> conv_kernel;

    int32_t pos_conv_kernel = 128;
    int32_t pos_conv_groups = 16;

    int32_t sample_rate = 16000;
    int32_t blank_id    = 0;
};

struct ctc_conv_layer {
    ggml_tensor * w      = nullptr;
    ggml_tensor * b      = nullptr;
    ggml_tensor * norm_w = nullptr;
    ggml_tensor * norm_b = nullptr;
};

struct ctc_encoder_layer {
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * attn_norm_b = nullptr;

    ggml_tensor * attn_q_w = nullptr, * attn_q_b = nullptr;
    ggml_tensor * attn_k_w = nullptr, * attn_k_b = nullptr;
    ggml_tensor * attn_v_w = nullptr, * attn_v_b = nullptr;
    ggml_tensor * attn_out_w = nullptr, * attn_out_b = nullptr;

    ggml_tensor * ffn_norm_w = nullptr;
    ggml_tensor * ffn_norm_b = nullptr;
    ggml_tensor * ffn_up_w = nullptr, * ffn_up_b = nullptr;
    ggml_tensor * ffn_down_w = nullptr, * ffn_down_b = nullptr;
};

// wav2vec2 CTC acoustic model used purely as a forced aligner.
//
// The transcript text is not produced here - it comes from the ASR model and is
// only *positioned* in time by this class. Alignment runs a Viterbi pass over
// the CTC emissions, which is monotonic by construction: every audio frame is
// assigned to some character in order. That is exactly the property the
// autoregressive Qwen aligner lacks, where a mispredicted timestamp can leave a
// multi-second hole in the middle of continuous speech.
//
// Note the aligner absorbs silence into whichever character precedes it rather
// than leaving a gap, so a line's last character can legitimately extend well
// past the sound - that keeps short lines on screen instead of flashing past.
class CtcAligner {
public:
    CtcAligner();
    ~CtcAligner();

    bool load_model(const std::string & model_path);
    void set_n_threads(int n_threads);

    // Force-aligns `chars` against 16kHz mono `samples`. Returned times are in
    // seconds relative to the start of the buffer. Characters missing from the
    // CTC vocabulary are matched against a wildcard, so punctuation the acoustic
    // model never saw still gets a position.
    alignment_result align(const float * samples, int n_samples,
                            const std::vector<std::string> & chars);

    [[nodiscard]] const std::string & get_error() const { return error_msg_; }

private:
    bool parse_hparams(gguf_context * ctx);
    bool create_tensors(gguf_context * ctx, ggml_context * meta_ctx);
    bool load_tensor_data(const std::string & path, gguf_context * ctx);

    // Runs the acoustic model and returns row-major [n_frames][vocab_size]
    // log-probabilities.
    bool compute_emissions(const float * samples, int n_samples,
                            std::vector<float> & emissions, int & n_frames);

    ggml_cgraph * build_graph(ggml_context * ctx0, ggml_tensor * input);

    ctc_hparams hparams_;
    std::vector<ctc_conv_layer>    conv_layers_;
    std::vector<ctc_encoder_layer> layers_;

    ggml_tensor * feat_proj_norm_w_ = nullptr, * feat_proj_norm_b_ = nullptr;
    ggml_tensor * feat_proj_w_ = nullptr, * feat_proj_b_ = nullptr;
    ggml_tensor * pos_conv_w_ = nullptr, * pos_conv_b_ = nullptr;
    ggml_tensor * enc_norm_w_ = nullptr, * enc_norm_b_ = nullptr;
    ggml_tensor * lm_head_w_ = nullptr, * lm_head_b_ = nullptr;

    std::map<std::string, ggml_tensor *> tensors_;
    std::unordered_map<std::string, int32_t> token_to_id_;

    ggml_context * model_ctx_ = nullptr;
    ggml_backend_buffer_t model_buffer_ = nullptr;

    ggml_backend_t backend_cpu_ = nullptr;
    ggml_backend_t backend_gpu_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    std::vector<uint8_t> compute_meta_;

    bool model_loaded_ = false;
    std::string error_msg_;
    int n_threads_ = 4;
};

} // namespace qwen3_asr
