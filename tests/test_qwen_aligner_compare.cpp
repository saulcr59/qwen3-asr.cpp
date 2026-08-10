// Runs the built-in Qwen3 forced aligner (autoregressive, non-monotonic) on
// real audio with a known-correct reference text, for comparing its output
// against the wav2vec2 CTC aligners' Viterbi timings. Reads the reference
// text from a file rather than argv - Windows converts the wide command
// line to narrow using the ANSI codepage before main() runs, which mangles
// non-ASCII CLI arguments.
//
// Usage: test_qwen_aligner_compare <aligner.gguf> <audio.wav> <ref_text.txt> [language]

#include "forced_aligner.h"
#include "mel_spectrogram.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s <aligner.gguf> <audio.wav> <ref_text.txt> [language]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string wav_path   = argv[2];
    const std::string language   = argc == 5 ? argv[4] : "";

    std::ifstream ref_in(argv[3], std::ios::binary);
    if (!ref_in) {
        fprintf(stderr, "Failed to open reference text file: %s\n", argv[3]);
        return 1;
    }
    std::string ref_text((std::istreambuf_iterator<char>(ref_in)), std::istreambuf_iterator<char>());
    while (!ref_text.empty() && (ref_text.back() == '\n' || ref_text.back() == '\r')) {
        ref_text.pop_back();
    }

    std::vector<float> samples;
    int sample_rate = 0;
    if (!load_wav(wav_path, samples, sample_rate)) {
        fprintf(stderr, "Failed to load wav: %s\n", wav_path.c_str());
        return 1;
    }
    fprintf(stderr, "Loaded %zu samples (%.2fs)\n", samples.size(), samples.size() / 16000.0);

    qwen3_asr::ForcedAligner aligner;
    if (!aligner.load_model(model_path)) {
        fprintf(stderr, "Failed to load aligner model: %s\n", aligner.get_error().c_str());
        return 1;
    }
    aligner.set_n_threads(4);

    auto result = aligner.align(samples.data(), (int) samples.size(), ref_text, language);
    if (!result.success) {
        fprintf(stderr, "align() failed: %s\n", result.error_msg.c_str());
        return 1;
    }

    fprintf(stderr, "align() produced %zu word(s):\n", result.words.size());
    float prev_end = -1.0f;
    bool monotonic = true;
    for (const auto & w : result.words) {
        fprintf(stderr, "  [%7.3f - %7.3f] %s\n", w.start, w.end, w.word.c_str());
        if (w.start < prev_end) monotonic = false;
        prev_end = w.end;
    }
    fprintf(stderr, "Monotonic: %s\n", monotonic ? "yes" : "NO");
    if (!result.words.empty()) {
        fprintf(stderr, "Span: %.3fs - %.3fs (audio is %.3fs)\n",
                result.words.front().start, result.words.back().end,
                samples.size() / 16000.0);
    }
    return 0;
}
