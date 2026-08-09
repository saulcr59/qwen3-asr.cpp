#pragma once

#include "qwen3asr_win_export.h"

#include <ggml.h>
#include <ggml-backend.h>
#include <map>
#include <string>
#include <vector>

namespace qwen3_asr {

// A detected speech region, in seconds.
struct vad_segment {
    double start;
    double end;
};

struct vad_params {
    float threshold               = 0.5f;   // speech probability threshold
    int   min_speech_duration_ms  = 250;
    int   min_silence_duration_ms = 300;    // silence needed to end a speech segment
    float max_speech_duration_s   = 1e9f;   // segments longer than this get force-split
    int   speech_pad_ms           = 30;     // padding added around each detected segment
};

struct vad_hparams {
    static constexpr int32_t n_encoder_layers = 4;

    int32_t encoder_in_channels[4]  = {0, 0, 0, 0};
    int32_t encoder_out_channels[4] = {0, 0, 0, 0};
    int32_t kernel_sizes[4]         = {0, 0, 0, 0};
    int32_t lstm_input_size  = 0;
    int32_t lstm_hidden_size = 0;
    int32_t final_conv_in  = 0;
    int32_t final_conv_out = 0;
};

// Silero VAD (v5), ported from whisper.cpp's ggml implementation. A tiny
// (~1MB) streaming model: STFT -> 4-layer conv encoder -> 1 LSTM cell ->
// final conv -> sigmoid, run one ~32ms frame at a time. It is used to find
// real silence gaps in the audio so long-audio chunking never has to guess a
// fixed clock-time cut point and risk slicing through a word or sentence.
class VoiceActivityDetector {
public:
    VoiceActivityDetector();
    ~VoiceActivityDetector();

    // Loads a model in the legacy ggml binary format used by whisper.cpp's
    // VAD models (e.g. ggml-silero-v5.1.2.bin) - NOT GGUF.
    bool load_model(const std::string & model_path);
    void set_n_threads(int n_threads);

    // Runs VAD over the whole signal (16kHz mono) and returns non-overlapping
    // speech segments, in seconds, sorted by start time.
    std::vector<vad_segment> detect_speech(const float * samples, int n_samples,
                                            const vad_params & params = vad_params());

    [[nodiscard]] const std::string & get_error() const { return error_msg_; }

private:
    bool create_tensors();

    // Per-frame speech probabilities for the whole signal.
    std::vector<float> compute_probs(const float * samples, int n_samples);

    vad_hparams hparams_;

    ggml_tensor * stft_forward_basis_ = nullptr;
    ggml_tensor * encoder_0_weight_ = nullptr;
    ggml_tensor * encoder_0_bias_   = nullptr;
    ggml_tensor * encoder_1_weight_ = nullptr;
    ggml_tensor * encoder_1_bias_   = nullptr;
    ggml_tensor * encoder_2_weight_ = nullptr;
    ggml_tensor * encoder_2_bias_   = nullptr;
    ggml_tensor * encoder_3_weight_ = nullptr;
    ggml_tensor * encoder_3_bias_   = nullptr;
    ggml_tensor * lstm_ih_weight_ = nullptr;
    ggml_tensor * lstm_ih_bias_   = nullptr;
    ggml_tensor * lstm_hh_weight_ = nullptr;
    ggml_tensor * lstm_hh_bias_   = nullptr;
    ggml_tensor * final_conv_weight_ = nullptr;
    ggml_tensor * final_conv_bias_   = nullptr;

    std::map<std::string, ggml_tensor *> tensors_;

    ggml_context * model_ctx_ = nullptr;
    ggml_backend_buffer_t model_buffer_ = nullptr;

    ggml_backend_t backend_cpu_ = nullptr;
    ggml_context * state_ctx_ = nullptr;
    ggml_backend_buffer_t state_buffer_ = nullptr;
    ggml_tensor * h_state_ = nullptr;
    ggml_tensor * c_state_ = nullptr;

    int32_t n_window_ = 0; // samples per frame

    bool model_loaded_ = false;
    std::string error_msg_;
    int n_threads_ = 4;
};

} // namespace qwen3_asr
