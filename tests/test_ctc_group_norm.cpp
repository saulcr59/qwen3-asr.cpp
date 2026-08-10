// Dumps wav2vec2 CTC emissions (row-major [n_frames][vocab_size] log-probs) to
// a raw f32 binary file, for numeric comparison against a PyTorch reference
// forward pass. Used to validate the group-norm feature-extractor path added
// for the ReazonSpeech wav2vec2 family (feat_extract_norm="group").
//
// Optional 4th argument: path to a UTF-8 text file holding a reference
// transcript, split into per-codepoint tokens and force-aligned via
// CtcAligner::align(), as an end-to-end smoke test of the Viterbi pass
// running on top of the new group-norm emissions (that decoding logic itself
// is unchanged / previously validated). Read from a file rather than argv
// directly - Windows converts the wide command line to narrow using the
// ANSI codepage before main() runs, which mangles non-ASCII CLI arguments.
//
// Usage: test_ctc_group_norm <model.gguf> <audio.wav> <out_emissions.bin> [reference_text.txt]

#include "ctc_aligner.h"
#include "mel_spectrogram.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

static std::vector<std::string> split_utf8_codepoints(const std::string & text) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < text.size()) {
        const unsigned char c = (unsigned char) text[i];
        size_t len = 1;
        if      ((c & 0x80) == 0x00) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        out.push_back(text.substr(i, len));
        i += len;
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s <model.gguf> <audio.wav> <out_emissions.bin> [reference_text.txt]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string wav_path   = argv[2];
    const std::string out_path   = argv[3];

    std::string ref_text;
    if (argc == 5) {
        std::ifstream ref_in(argv[4], std::ios::binary);
        if (!ref_in) {
            fprintf(stderr, "Failed to open reference text file: %s\n", argv[4]);
            return 1;
        }
        ref_text.assign((std::istreambuf_iterator<char>(ref_in)), std::istreambuf_iterator<char>());
        while (!ref_text.empty() && (ref_text.back() == '\n' || ref_text.back() == '\r')) {
            ref_text.pop_back();
        }
    }

    std::vector<float> samples;
    int sample_rate = 0;
    if (!load_wav(wav_path, samples, sample_rate)) {
        fprintf(stderr, "Failed to load wav: %s\n", wav_path.c_str());
        return 1;
    }
    if (sample_rate != 16000) {
        fprintf(stderr, "Expected 16kHz audio, got %d\n", sample_rate);
        return 1;
    }
    fprintf(stderr, "Loaded %zu samples (%.2fs)\n", samples.size(), samples.size() / 16000.0);

    qwen3_asr::CtcAligner aligner;
    if (!aligner.load_model(model_path)) {
        fprintf(stderr, "Failed to load CTC model: %s\n", aligner.get_error().c_str());
        return 1;
    }
    aligner.set_n_threads(4);

    std::vector<float> emissions;
    int n_frames = 0;
    if (!aligner.compute_emissions(samples.data(), (int) samples.size(), emissions, n_frames)) {
        fprintf(stderr, "compute_emissions failed: %s\n", aligner.get_error().c_str());
        return 1;
    }
    const int vocab_size = n_frames > 0 ? (int) (emissions.size() / n_frames) : 0;
    fprintf(stderr, "Emissions: %d frames x %d vocab\n", n_frames, vocab_size);

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        fprintf(stderr, "Failed to open output file: %s\n", out_path.c_str());
        return 1;
    }
    int32_t header[2] = { n_frames, vocab_size };
    out.write(reinterpret_cast<const char *>(header), sizeof(header));
    out.write(reinterpret_cast<const char *>(emissions.data()), emissions.size() * sizeof(float));
    fprintf(stderr, "Wrote %s\n", out_path.c_str());

    if (!ref_text.empty()) {
        auto chars = split_utf8_codepoints(ref_text);
        fprintf(stderr, "\nRunning end-to-end align() with %zu reference tokens...\n", chars.size());
        auto result = aligner.align(samples.data(), (int) samples.size(), chars);
        if (!result.success) {
            fprintf(stderr, "align() failed: %s\n", result.error_msg.c_str());
            return 1;
        }
        fprintf(stderr, "align() produced %zu word(s):\n", result.words.size());
        float prev_end = -1.0f;
        bool monotonic = true;
        for (const auto & w : result.words) {
            fprintf(stderr, "  [%6.3f - %6.3f] %s\n", w.start, w.end, w.word.c_str());
            if (w.start < prev_end) monotonic = false;
            prev_end = w.end;
        }
        fprintf(stderr, "Monotonic: %s\n", monotonic ? "yes" : "NO (BUG)");
        if (!result.words.empty()) {
            fprintf(stderr, "Span: %.3fs - %.3fs (audio is %.3fs)\n",
                    result.words.front().start, result.words.back().end,
                    samples.size() / 16000.0);
        }
    }
    return 0;
}
