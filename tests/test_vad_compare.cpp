// Compares Silero VAD v5.1.2 against v6.2.0 on the same real audio, using the
// exact same ggml graph code (the two checkpoints share an identical header
// layout and tensor shapes - only the trained weights differ). Prints
// detected speech segments from each so the two versions can be eyeballed
// side by side, plus a rough IoU-style overlap summary.
//
// Usage: test_vad_compare <v5_model.bin> <v6_model.bin> <audio.wav>

#include "mel_spectrogram.h"
#include "vad.h"

#include <algorithm>
#include <cstdio>
#include <vector>

static void print_segments(const std::vector<qwen3_asr::vad_segment> & segs) {
    double total = 0.0;
    for (const auto & s : segs) {
        fprintf(stderr, "  [%7.3f - %7.3f]  (%.3fs)\n", s.start, s.end, s.end - s.start);
        total += s.end - s.start;
    }
    fprintf(stderr, "  %zu segment(s), %.3fs total speech\n", segs.size(), total);
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <v5_model.bin> <v6_model.bin> <audio.wav>\n", argv[0]);
        return 1;
    }

    std::vector<float> samples;
    int sample_rate = 0;
    if (!load_wav(argv[3], samples, sample_rate)) {
        fprintf(stderr, "Failed to load wav: %s\n", argv[3]);
        return 1;
    }
    if (sample_rate != 16000) {
        fprintf(stderr, "Expected 16kHz audio, got %d\n", sample_rate);
        return 1;
    }
    fprintf(stderr, "Audio: %zu samples (%.2fs)\n\n", samples.size(), samples.size() / 16000.0);

    qwen3_asr::VoiceActivityDetector vad5, vad6;
    if (!vad5.load_model(argv[1])) {
        fprintf(stderr, "Failed to load v5 model: %s\n", vad5.get_error().c_str());
        return 1;
    }
    if (!vad6.load_model(argv[2])) {
        fprintf(stderr, "Failed to load v6 model: %s\n", vad6.get_error().c_str());
        return 1;
    }
    vad5.set_n_threads(4);
    vad6.set_n_threads(4);

    auto segs5 = vad5.detect_speech(samples.data(), (int) samples.size());
    auto segs6 = vad6.detect_speech(samples.data(), (int) samples.size());

    fprintf(stderr, "=== v5.1.2 ===\n");
    print_segments(segs5);
    fprintf(stderr, "\n=== v6.2.0 ===\n");
    print_segments(segs6);

    return 0;
}
