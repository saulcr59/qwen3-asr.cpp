#include "qwen3_asr.h"
#include "forced_aligner.h"
#include "ctc_aligner.h"
#include "vad.h"
#include "timing.h"

#include <ggml.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <clocale>
#include <string>
#include <fstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

struct cli_params {
    std::string model_path = "models/qwen3-asr-0.6b-f16.gguf";
    std::string aligner_model_path = "";
    std::string ctc_aligner_model_path = "";
    std::string vad_model_path = "";
    std::string audio_path = "";
    std::string output_path = "";
    std::string language = "";
    std::string align_text = "";
    int32_t max_tokens = 1024;
    int32_t n_threads = 4;
    bool print_progress = false;
    bool print_timing = true;
    bool print_tokens = false;
    bool align_mode = false;
    bool transcribe_align_mode = false;
    bool profile = false;
    bool output_srt = false;
};

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <path>     Path to GGUF model (default: models/qwen3-asr-0.6b-f16.gguf)\n");
    fprintf(stderr, "  -f, --audio <path>     Path to audio file (WAV, 16kHz mono) [required]\n");
    fprintf(stderr, "  -o, --output <path>    Output file path (default: stdout)\n");
    fprintf(stderr, "  -l, --language <code>  Language code (optional, e.g. 'korean' for Korean word splitting)\n");
    fprintf(stderr, "  -t, --threads <n>      Number of threads (default: 4)\n");
    fprintf(stderr, "  --max-tokens <n>       Maximum tokens to generate (default: 1024)\n");
    fprintf(stderr, "  --progress             Print progress during transcription\n");
    fprintf(stderr, "  --no-timing            Don't print timing information\n");
    fprintf(stderr, "  --tokens               Print token IDs\n");
    fprintf(stderr, "  --profile              Print detailed timing profile (requires QWEN3_ASR_TIMING build)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Forced Alignment:\n");
    fprintf(stderr, "  --align                Enable forced alignment mode\n");
    fprintf(stderr, "  --text <text>          Reference transcript for alignment\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Transcribe + Align:\n");
    fprintf(stderr, "  -a, --transcribe-align Run ASR then forced alignment\n");
    fprintf(stderr, "  --aligner-model <path> Path to forced aligner GGUF model (required with --transcribe-align)\n");
    fprintf(stderr, "  --ctc-align-model <p>  Path to a wav2vec2 CTC GGUF model. Used instead of --aligner-model to\n");
    fprintf(stderr, "                         time the transcript, via monotonic CTC alignment (no timestamp jumps).\n");
    fprintf(stderr, "  --vad-model <path>     Path to ggml Silero VAD model. When given, long audio is split into\n");
    fprintf(stderr, "                         chunks at detected silence instead of fixed 30s/2s-overlap windows.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Output Formats:\n");
    fprintf(stderr, "  -osrt, --output-srt    Output result in a SRT file\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  -h, --help             Show this help message\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  Transcription:\n");
    fprintf(stderr, "    %s -m models/qwen3-asr-0.6b-f16.gguf -f sample.wav\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "  Forced Alignment:\n");
    fprintf(stderr, "    %s -m models/qwen3-forced-aligner-0.6b-f16.gguf -f sample.wav --align --text \"Hello world\"\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "  Transcribe + Align:\n");
    fprintf(stderr, "    %s -m models/qwen3-asr-0.6b-f16.gguf --aligner-model models/qwen3-forced-aligner-0.6b-f16.gguf -f sample.wav --transcribe-align\n", prog);
}

static bool parse_args(int argc, char ** argv, cli_params & params) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        
        if (strcmp(arg, "-m") == 0 || strcmp(arg, "--model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.model_path = argv[++i];
        } else if (strcmp(arg, "-f") == 0 || strcmp(arg, "--audio") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.audio_path = argv[++i];
        } else if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.output_path = argv[++i];
        } else if (strcmp(arg, "-l") == 0 || strcmp(arg, "--language") == 0 || strcmp(arg, "--lang") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.language = argv[++i];
        } else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--threads") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.n_threads = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--max-tokens") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.max_tokens = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--progress") == 0) {
            params.print_progress = true;
        } else if (strcmp(arg, "--no-timing") == 0) {
            params.print_timing = false;
        } else if (strcmp(arg, "--tokens") == 0) {
            params.print_tokens = true;
        } else if (strcmp(arg, "--profile") == 0) {
            params.profile = true;
        } else if (strcmp(arg, "--align") == 0) {
            params.align_mode = true;
        } else if (strcmp(arg, "-osrt") == 0 || strcmp(arg, "--output-srt") == 0) {
            params.output_srt = true;
        } else if (strcmp(arg, "-a") == 0 || strcmp(arg, "--transcribe-align") == 0) {
            params.transcribe_align_mode = true;
        } else if (strcmp(arg, "--aligner-model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.aligner_model_path = argv[++i];
        } else if (strcmp(arg, "--ctc-align-model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.ctc_aligner_model_path = argv[++i];
        } else if (strcmp(arg, "--vad-model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.vad_model_path = argv[++i];
        } else if (strcmp(arg, "--text") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.align_text = argv[++i];
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Error: Unknown argument: %s\n", arg);
            return false;
        }
    }
    
    if (params.audio_path.empty()) {
        fprintf(stderr, "Error: Audio file path is required (-f/--audio)\n");
        return false;
    }
    
    if (params.align_mode && params.align_text.empty()) {
        fprintf(stderr, "Error: Reference text is required for alignment mode (--text)\n");
        return false;
    }

    if (params.align_mode && params.transcribe_align_mode) {
        fprintf(stderr, "Error: --align and --transcribe-align cannot be used together\n");
        return false;
    }

    if (params.transcribe_align_mode && params.aligner_model_path.empty() && params.ctc_aligner_model_path.empty()) {
        fprintf(stderr, "Error: --transcribe-align needs an aligner: --aligner-model or --ctc-align-model\n");
        return false;
    }
    
    return true;
}

static std::string normalize_language_name(std::string lang) {
    std::transform(lang.begin(), lang.end(), lang.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (lang == "zh" || lang == "zh-cn" || lang == "zh-tw" || lang == "zh-hans" ||
        lang == "zh-hant" || lang == "cmn" || lang == "mandarin" ||
        lang == "cantonese" || lang == "yue") {
        return "chinese";
    }
    if (lang == "ko" || lang == "kr") {
        return "korean";
    }
    if (lang == "ja" || lang == "jp") {
        return "japanese";
    }
    return lang;
}

static std::string detect_language(const std::string & asr_language_token) {
    std::string lang = asr_language_token;

    // Trim whitespace
    lang.erase(lang.begin(), std::find_if(lang.begin(), lang.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    lang.erase(std::find_if(lang.rbegin(), lang.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), lang.end());

    // Convert to lowercase
    std::transform(lang.begin(), lang.end(), lang.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Some common mappings if needed
    if (lang == "<|zh|>") return "chinese";
    if (lang == "<|en|>") return "english";
    if (lang == "<|ko|>") return "korean";

    return normalize_language_name(lang);
}

static std::string extract_transcript(const std::string & asr_text) {
    const std::string prefix = "language ";
    if (asr_text.size() < prefix.size() || asr_text.compare(0, prefix.size(), prefix) != 0) {
        if (asr_text.rfind("<asr_text>", 0) == 0) {
            return asr_text.substr(sizeof("<asr_text>") - 1);
        }
        return asr_text;
    }

    size_t pos = prefix.size();
    if (pos >= asr_text.size()) {
        return "";
    }

    unsigned char first = static_cast<unsigned char>(asr_text[pos]);
    if (!std::isupper(first)) {
        return asr_text;
    }

    ++pos;
    while (pos < asr_text.size()) {
        unsigned char c = static_cast<unsigned char>(asr_text[pos]);
        if (!std::islower(c)) {
            break;
        }
        ++pos;
    }

    while (pos < asr_text.size()) {
        unsigned char c = static_cast<unsigned char>(asr_text[pos]);
        if (c >= 0x80 || !std::isspace(c)) {
            break;
        }
        ++pos;
    }

    std::string transcript = asr_text.substr(pos);
    if (transcript.rfind("<asr_text>", 0) == 0) {
        transcript.erase(0, sizeof("<asr_text>") - 1);
    }
    return transcript;
}

static std::string escape_json_string(const std::string & s) {
    std::string result;
    result.reserve(s.size() + 10);
    for (char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    result += buf;
                } else {
                    result += c;
                }
        }
    }
    return result;
}

// Formats a timestamp for JSON. A failed alignment can leave a word's start/end as
// NaN or +/-Inf; "%.3f" would then emit "nan"/"-nan(ind)"/"inf", which is NOT valid
// JSON and makes strict parsers (e.g. System.Text.Json in Subtitle Edit) reject the
// whole file. Coerce any non-finite value to 0.0 so the output always parses.
static std::string json_time(float t) {
    if (!std::isfinite(t)) {
        t = 0.0f;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%.3f", t);
    return std::string(buf);
}

static std::string alignment_to_json(const qwen3_asr::alignment_result & result) {
    std::string json = "{\n  \"words\": [\n";

    for (size_t i = 0; i < result.words.size(); ++i) {
        const auto & w = result.words[i];
        // Build every field with std::string (never a fixed char buffer) so nothing is
        // ever truncated — a long word (e.g. a whole Chinese transcript aligned as a
        // single "word") or an unexpectedly large timestamp would otherwise cut the JSON
        // short and leave it unparseable. The word text is escaped; the timestamps are
        // coerced to finite numbers. Together this guarantees well-formed JSON.
        json += "    {\"word\": \"";
        json += escape_json_string(w.word);
        json += "\", \"start\": ";
        json += json_time(w.start);
        json += ", \"end\": ";
        json += json_time(w.end);
        json += "}";
        if (i + 1 < result.words.size()) {
            json += ",";
        }
        json += "\n";
    }

    json += "  ]\n}";
    return json;
}

static std::string to_timestamp(double t_sec, bool comma = false) {
    int64_t msec = static_cast<int64_t>(t_sec * 1000.0);
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);
    return std::string(buf);
}

static std::string alignment_to_srt(const qwen3_asr::alignment_result & result) {
    std::string srt = "";
    if (result.words.empty()) return srt;

    int segment_index = 1;
    size_t i = 0;

    while (i < result.words.size()) {
        std::string text = "";
        double start_time = result.words[i].start;
        double end_time = result.words[i].end;
        int word_count = 0;

        while (i < result.words.size()) {
            const auto & w = result.words[i];

            // Only break on pauses if the pause is very long (e.g. > 3 seconds) 
            // OR if the pause is > 1.5 seconds AND the previous word ended a clause
            bool is_end_of_clause = !text.empty() && 
                (text.back() == ',' || text.back() == '.' || text.back() == '?' || text.back() == '!' ||
                 text.find("\xEF\xBC\x8C") != std::string::npos || // ，
                 text.find("\xE3\x80\x82") != std::string::npos);  // 。

            if (word_count > 0 && (w.start - end_time > 3.0 || (w.start - end_time > 1.5 && is_end_of_clause))) {
                break;
            }

            // For non-Asian languages, we usually need spaces between words
            if (word_count > 0 && !text.empty() && 
                static_cast<unsigned char>(text.back()) < 0x80 &&
                static_cast<unsigned char>(w.word.front()) < 0x80 &&
                w.word.find_first_of(".,!?") != 0) {
                text += " ";
            }

            text += w.word;
            // Update end time only if the word has a valid end time
            if (w.end > end_time) {
                end_time = w.end;
            }
            word_count++;
            i++;

            // Break on terminal punctuation (sentences)
            if (w.word.find("\xE3\x80\x82") != std::string::npos || // 。
                w.word.find("\xEF\xBC\x9F") != std::string::npos || // ？
                w.word.find("\xEF\xBC\x81") != std::string::npos || // ！
                w.word.find(".") != std::string::npos ||
                w.word.find("?") != std::string::npos ||
                w.word.find("!") != std::string::npos ||
                w.word.find("\n") != std::string::npos) {
                break;
            }

            // Break if sentence gets too long (prevent endless blocks)
            if (word_count >= 20 || text.size() >= 80) {
                break;
            }
        }

        char buf[64];
        snprintf(buf, sizeof(buf), "%d\n", segment_index++);
        srt += buf;

        srt += to_timestamp(start_time, true) + " --> " + to_timestamp(end_time, true) + "\n";
        srt += text + "\n\n";
    }

    return srt;
}
static std::string find_korean_dict(const std::string & model_path) {
    auto dir_of = [](const std::string & path) -> std::string {
        size_t pos = path.find_last_of("/\\");
        return (pos != std::string::npos) ? path.substr(0, pos) : ".";
    };

    std::vector<std::string> candidates = {
        dir_of(model_path) + "/../assets/korean_dict_jieba.dict",
        dir_of(model_path) + "/assets/korean_dict_jieba.dict",
        "assets/korean_dict_jieba.dict",
    };

    for (const auto & p : candidates) {
        std::ifstream f(p);
        if (f.good()) return p;
    }
    return "";
}

static int run_alignment(const cli_params & params) {
    const std::string align_lang = normalize_language_name(params.language);

    fprintf(stderr, "qwen3-asr-cli (Forced Alignment Mode)\n");
    fprintf(stderr, "  Model: %s\n", params.model_path.c_str());
    fprintf(stderr, "  Audio: %s\n", params.audio_path.c_str());
    fprintf(stderr, "  Text: %s\n", params.align_text.c_str());
    if (!align_lang.empty()) {
        fprintf(stderr, "  Language: %s\n", align_lang.c_str());
    }
    fprintf(stderr, "\n");
    
    qwen3_asr::ForcedAligner aligner;
    
    if (!aligner.load_model(params.model_path)) {
        fprintf(stderr, "Error: %s\n", aligner.get_error().c_str());
        return 1;
    }
    aligner.set_n_threads(params.n_threads);
    
    if (align_lang == "korean") {
        std::string dict_path = find_korean_dict(params.model_path);
        if (dict_path.empty()) {
            fprintf(stderr, "Warning: Korean dictionary not found. Falling back to whitespace splitting.\n");
        } else {
            if (!aligner.load_korean_dict(dict_path)) {
                fprintf(stderr, "Warning: Failed to load Korean dictionary from %s\n", dict_path.c_str());
            }
        }
    }
    
    fprintf(stderr, "Model loaded. Running alignment...\n");
    
    auto result = aligner.align(params.audio_path, params.align_text, align_lang);
    
    if (!result.success) {
        fprintf(stderr, "Error: %s\n", result.error_msg.c_str());
        return 1;
    }
    
    if (params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Mel spectrogram: %lld ms\n", (long long)result.t_mel_ms);
        fprintf(stderr, "  Audio encoding:  %lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Text decoding:   %lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:           %lld ms\n", (long long)result.t_total_ms);
        fprintf(stderr, "  Words aligned:   %zu\n", result.words.size());
    }
    
    std::string string_output = params.output_srt ? alignment_to_srt(result) : alignment_to_json(result);
    
    if (params.output_path.empty()) {
        printf("%s\n", string_output.c_str());
    } else {
        std::ofstream out(params.output_path);
        if (!out) {
            fprintf(stderr, "Error: Failed to open output file: %s\n", params.output_path.c_str());
            return 1;
        }
        out << string_output << "\n";
        fprintf(stderr, "Output written to: %s\n", params.output_path.c_str());
    }
    
    if (params.profile) {
        QWEN3_TIMER_REPORT();
    }
    
    return 0;
}

static int run_transcription(const cli_params & params) {
    fprintf(stderr, "qwen3-asr-cli\n");
    fprintf(stderr, "  Model: %s\n", params.model_path.c_str());
    fprintf(stderr, "  Audio: %s\n", params.audio_path.c_str());
    fprintf(stderr, "  Threads: %d\n", params.n_threads);
    fprintf(stderr, "\n");
    
    qwen3_asr::Qwen3ASR asr;
    
    if (!asr.load_model(params.model_path)) {
        fprintf(stderr, "Error: %s\n", asr.get_error().c_str());
        return 1;
    }
    
    qwen3_asr::transcribe_params tp;
    tp.max_tokens = params.max_tokens;
    tp.language = params.language;
    tp.n_threads = params.n_threads;
    tp.print_progress = params.print_progress;
    tp.print_timing = params.print_timing;
    
    auto result = asr.transcribe(params.audio_path, tp);
    
    if (!result.success) {
        fprintf(stderr, "Error: %s\n", result.error_msg.c_str());
        return 1;
    }
    
    if (params.print_tokens) {
        fprintf(stderr, "\nTokens (%zu):\n", result.tokens.size());
        for (size_t i = 0; i < result.tokens.size(); ++i) {
            fprintf(stderr, "  [%zu] %d\n", i, result.tokens[i]);
        }
        fprintf(stderr, "\n");
    }
    
    if (params.output_path.empty()) {
        printf("%s\n", result.text.c_str());
    } else {
        std::ofstream out(params.output_path);
        if (!out) {
            fprintf(stderr, "Error: Failed to open output file: %s\n", params.output_path.c_str());
            return 1;
        }
        out << result.text << "\n";
        fprintf(stderr, "Output written to: %s\n", params.output_path.c_str());
    }
    
    if (params.profile) {
        QWEN3_TIMER_REPORT();
    }
    
    return 0;
}

// Splits a transcript the same way the Qwen aligner does, so switching aligners
// doesn't change the granularity of the JSON/SRT that consumers already parse:
// each CJK character stands alone (those scripts have no word separators), while
// runs of Latin text stay together as one token.
static std::vector<std::string> split_transcript_tokens(const std::string & text) {
    auto is_cjk = [](uint32_t cp) {
        return (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF) ||
               (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x3000 && cp <= 0x303F) ||
               (cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x31F0 && cp <= 0x31FF) ||
               (cp >= 0xFF00 && cp <= 0xFF65) || (cp >= 0x20000 && cp <= 0x2A6DF);
    };

    std::vector<std::string> out;
    std::string pending;
    size_t i = 0;
    while (i < text.size()) {
        const unsigned char c = (unsigned char) text[i];
        size_t len = 1;
        if      ((c & 0x80) == 0x00) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        len = std::min(len, text.size() - i);

        uint32_t cp = 0;
        if (len == 1) cp = c;
        else if (len == 2) cp = ((c & 0x1F) << 6) | (text[i + 1] & 0x3F);
        else if (len == 3) cp = ((c & 0x0F) << 12) | ((text[i + 1] & 0x3F) << 6) | (text[i + 2] & 0x3F);
        else              cp = ((c & 0x07) << 18) | ((text[i + 1] & 0x3F) << 12) |
                                ((text[i + 2] & 0x3F) << 6) | (text[i + 3] & 0x3F);

        const std::string unit = text.substr(i, len);
        i += len;

        const bool is_space = (len == 1 && isspace(c)) || cp == 0x3000;
        if (is_space || is_cjk(cp)) {
            if (!pending.empty()) { out.push_back(pending); pending.clear(); }
            if (!is_space) out.push_back(unit);
        } else {
            pending += unit;
        }
    }
    if (!pending.empty()) out.push_back(pending);
    return out;
}

// Builds non-overlapping [start, end) sample ranges covering the whole signal, each
// no longer than max_chunk_samples (the ASR/aligner models' ~30s hard window limit).
// Cuts are placed at the midpoint of a detected silence gap whenever one is available
// in range, so a chunk boundary never lands in the middle of a word or sentence. Only
// falls back to a hard cut at max_chunk_samples when a single speech segment alone
// exceeds that limit (no silence to cut at).
static std::vector<std::pair<int, int>> build_vad_chunk_ranges(
        const std::vector<qwen3_asr::vad_segment> & speech, int total_samples, int max_chunk_samples) {
    std::vector<std::pair<int, int>> chunks;
    if (total_samples <= max_chunk_samples) {
        chunks.push_back({0, total_samples});
        return chunks;
    }

    std::vector<int> cut_points;
    for (size_t i = 0; i + 1 < speech.size(); ++i) {
        int gap_start = (int) std::lround(speech[i].end * 16000.0);
        int gap_end = (int) std::lround(speech[i + 1].start * 16000.0);
        if (gap_end > gap_start) {
            cut_points.push_back((gap_start + gap_end) / 2);
        }
    }

    int chunk_start = 0;
    while (chunk_start < total_samples) {
        int hard_limit = std::min(total_samples, chunk_start + max_chunk_samples);
        if (hard_limit >= total_samples) {
            chunks.push_back({chunk_start, total_samples});
            break;
        }

        int best_cut = -1;
        for (int cp : cut_points) {
            if (cp > chunk_start && cp <= hard_limit) {
                best_cut = cp; // cut_points is ascending: keep advancing to the latest valid one
            } else if (cp > hard_limit) {
                break;
            }
        }

        int cut = (best_cut > chunk_start) ? best_cut : hard_limit;
        chunks.push_back({chunk_start, cut});
        chunk_start = cut;
    }

    return chunks;
}

static int run_transcribe_and_align(const cli_params & params) {
    fprintf(stderr, "qwen3-asr-cli (Transcribe + Align Mode)\n");
    fprintf(stderr, "  ASR Model: %s\n", params.model_path.c_str());
    fprintf(stderr, "  Aligner Model: %s\n", params.ctc_aligner_model_path.empty()
            ? params.aligner_model_path.c_str()
            : (params.ctc_aligner_model_path + " (CTC)").c_str());
    fprintf(stderr, "  Audio: %s\n", params.audio_path.c_str());
    fprintf(stderr, "  Threads: %d\n", params.n_threads);
    fprintf(stderr, "\n");

    qwen3_asr::Qwen3ASR asr;
    if (!asr.load_model(params.model_path)) {
        fprintf(stderr, "Error (ASR): %s\n", asr.get_error().c_str());
        return 1;
    }

    // Two interchangeable timing back-ends. CTC alignment is monotonic by
    // construction so it cannot leave a multi-second hole in the middle of
    // continuous speech, which the autoregressive aligner occasionally does;
    // the Qwen aligner in turn is multilingual from a single model. Either way
    // the transcript text always comes from the ASR model above.
    const bool use_ctc = !params.ctc_aligner_model_path.empty();

    qwen3_asr::ForcedAligner aligner;
    qwen3_asr::CtcAligner ctc_aligner;

    if (use_ctc) {
        if (!ctc_aligner.load_model(params.ctc_aligner_model_path)) {
            fprintf(stderr, "Error (CTC Aligner): %s\n", ctc_aligner.get_error().c_str());
            return 1;
        }
        ctc_aligner.set_n_threads(params.n_threads);
    } else {
        if (!aligner.load_model(params.aligner_model_path)) {
            fprintf(stderr, "Error (Aligner): %s\n", aligner.get_error().c_str());
            return 1;
        }
        aligner.set_n_threads(params.n_threads);
    }

    std::vector<float> all_samples;
    int sample_rate;
    if (!load_wav(params.audio_path, all_samples, sample_rate)) {
        fprintf(stderr, "Error: Failed to load audio file: %s\n", params.audio_path.c_str());
        return 1;
    }
    if (sample_rate != 16000) {
        fprintf(stderr, "Error: Audio must be 16kHz\n");
        return 1;
    }

    const int chunk_size_samples = 30 * 16000;
    const int chunk_stride_samples = 28 * 16000; // 2 seconds overlap (legacy, fixed-window mode only)

    // With --vad-model: cut only at real silence, chunks never overlap. Without it:
    // fall back to the original fixed-size sliding window (kept for callers that
    // don't have a VAD model available).
    bool use_vad_chunking = !params.vad_model_path.empty();
    std::vector<std::pair<int, int>> chunk_ranges;
    std::vector<qwen3_asr::vad_segment> speech_segments; // kept for the post-alignment gap sanity check below

    if (use_vad_chunking) {
        qwen3_asr::VoiceActivityDetector vad;
        if (!vad.load_model(params.vad_model_path)) {
            fprintf(stderr, "Error (VAD): %s\n", vad.get_error().c_str());
            return 1;
        }
        vad.set_n_threads(params.n_threads);

        speech_segments = vad.detect_speech(all_samples.data(), (int) all_samples.size());
        fprintf(stderr, "VAD: detected %zu speech segment(s)\n", speech_segments.size());

        chunk_ranges = build_vad_chunk_ranges(speech_segments, (int) all_samples.size(), chunk_size_samples);
    } else {
        int n_chunks = (all_samples.size() > chunk_size_samples)
                       ? (all_samples.size() - chunk_size_samples + chunk_stride_samples - 1) / chunk_stride_samples + 1
                       : 1;
        for (int c = 0; c < n_chunks; ++c) {
            int start = c * chunk_stride_samples;
            int end = std::min((int) all_samples.size(), start + chunk_size_samples);
            chunk_ranges.push_back({start, end});
        }
    }
    const int n_chunks = (int) chunk_ranges.size();

    qwen3_asr::alignment_result full_align_result;
    full_align_result.success = true;
    full_align_result.t_total_ms = 0;
    int64_t total_asr_ms = 0;
    int64_t total_align_ms = 0;

    std::string global_lang = normalize_language_name(params.language);
    if (!use_ctc && global_lang == "korean") {
        std::string dict_path = find_korean_dict(params.aligner_model_path);
        if (!dict_path.empty()) {
            aligner.load_korean_dict(dict_path);
        }
    }

    for (int c = 0; c < n_chunks; ++c) {
        int start = chunk_ranges[c].first;
        int end = chunk_ranges[c].second;
        int chunk_len = end - start;
        
        fprintf(stderr, "\n--- Processing Chunk %d/%d (%.2fs - %.2fs) ---\n", 
                c + 1, n_chunks, start / 16000.0f, end / 16000.0f);

        qwen3_asr::transcribe_params tp;
        tp.max_tokens = params.max_tokens;
        tp.language = global_lang;
        tp.n_threads = params.n_threads;
        tp.print_progress = params.print_progress;
        tp.print_timing = params.print_timing;

        auto asr_result = asr.transcribe(all_samples.data() + start, chunk_len, tp);
        if (!asr_result.success) {
            fprintf(stderr, "Error (ASR) in chunk %d: %s\n", c + 1, asr_result.error_msg.c_str());
            continue;
        }

        std::string detected_lang = detect_language(asr_result.language);
        if (global_lang.empty() && !detected_lang.empty()) {
            global_lang = detected_lang;
            if (!use_ctc && global_lang == "korean") {
                std::string dict_path = find_korean_dict(params.aligner_model_path);
                if (!dict_path.empty()) aligner.load_korean_dict(dict_path);
            }
        }
        
        std::string align_lang = global_lang.empty() ? detected_lang : global_lang;
        std::string transcript = extract_transcript(asr_result.text);
        
        fprintf(stderr, "  Detected language: %s\n", detected_lang.empty() ? "(none)" : detected_lang.c_str());
        fprintf(stderr, "  Alignment language: %s\n", align_lang.empty() ? "(none)" : align_lang.c_str());
        fprintf(stderr, "  Transcript: %s\n", transcript.c_str());
        
        if (transcript.empty()) continue;

        qwen3_asr::alignment_result align_result;
        if (use_ctc) {
            align_result = ctc_aligner.align(all_samples.data() + start, chunk_len,
                                              split_transcript_tokens(transcript));
        } else {
            align_result = aligner.align(all_samples.data() + start, chunk_len, transcript, align_lang);
        }
        if (!align_result.success) {
            fprintf(stderr, "Error (Aligner) in chunk %d: %s\n", c + 1, align_result.error_msg.c_str());
            continue;
        }

        float time_offset = start / 16000.0f;

        // Define strict ownership boundaries for this chunk. With VAD chunking, chunks are
        // contiguous and non-overlapping, so a chunk simply owns everything up to its own end.
        // With the legacy fixed-stride window, ownership stops at the start of the next
        // chunk's stride to avoid double-counting words in the overlap region.
        double chunk_own_start = time_offset;
        double chunk_own_end;
        if (c == n_chunks - 1) {
            chunk_own_end = 999999.0;
        } else if (use_vad_chunking) {
            chunk_own_end = end / 16000.0f;
        } else {
            chunk_own_end = time_offset + (chunk_stride_samples / 16000.0f);
        }
        
        for (auto & w : align_result.words) {
            double abs_start = w.start + time_offset;
            double abs_end = w.end + time_offset;

            // Hard partitioning: This chunk only "owns" words that start within its active stride window.
            // This prevents trailing truncated words (e.g. "For your.") in Chunk N from duplicating with Chunk N+1.
            if (abs_start >= chunk_own_start && abs_start < chunk_own_end) {
                // Only the legacy overlapping window can duplicate a word: there, chunk N and
                // chunk N+1 both align the shared overlap audio independently, and when the true
                // boundary between two words falls close to chunk_own_end each chunk's own
                // (slightly different) alignment can land on its own "owned" side of the cutoff -
                // e.g. "設" emitted once by chunk N (ending just before the cutoff) and again by
                // chunk N+1 (starting just after it). VAD chunks are contiguous and share no
                // audio, so they cannot produce this and must not be filtered, or a genuinely
                // repeated character straddling a boundary would be dropped. Restrict the check
                // to the start of the chunk's contribution for the same reason.
                bool near_chunk_boundary = !use_vad_chunking && (abs_start - chunk_own_start) < 1.0;
                if (near_chunk_boundary && !full_align_result.words.empty()) {
                    const auto & prev = full_align_result.words.back();
                    if (prev.word == w.word && std::fabs(abs_start - prev.end) < 0.25) {
                        continue;
                    }
                }
                w.start = abs_start;
                w.end = abs_end;
                full_align_result.words.push_back(w);
            }
        }
        
        total_asr_ms += asr_result.t_total_ms;
        total_align_ms += align_result.t_total_ms;
        full_align_result.t_total_ms += (asr_result.t_total_ms + align_result.t_total_ms);
    }

    // Sanity-check word-to-word gaps against the VAD's own silence detection. The
    // aligner occasionally misplaces a timestamp for a short/weakly-cued word (e.g.
    // "本" ending abruptly and "体" not resuming for 3.5s in "本体。", even though
    // there is no real pause between them - confirmed by inspecting the raw JSON: the
    // VAD never detected silence there). We already know exactly where real silence
    // is, so clamp any inter-word gap that isn't backed by an actual VAD silence
    // region - it can only be an alignment artifact, not a real pause, and left as-is
    // it gets misread downstream (e.g. by Subtitle Edit) as a sentence/paragraph
    // break. This does not attempt to reconstruct the "true" timestamp - it just
    // keeps a bogus gap from corrupting the pause-based heuristics that consume it.
    if (use_vad_chunking && !speech_segments.empty()) {
        const double max_ungrounded_gap = 0.4; // seconds
        int clamped = 0;
        for (size_t k = 0; k + 1 < full_align_result.words.size(); ++k) {
            auto & w0 = full_align_result.words[k];
            auto & w1 = full_align_result.words[k + 1];
            double gap = w1.start - w0.end;
            if (gap <= max_ungrounded_gap) {
                continue;
            }

            double mid = (w0.end + w1.start) / 2.0;
            bool still_in_speech = false;
            for (const auto & seg : speech_segments) {
                if (mid >= seg.start && mid <= seg.end) {
                    still_in_speech = true;
                    break;
                }
            }
            if (!still_in_speech) {
                continue; // genuine silence per VAD - trust the aligner's gap
            }

            double new_start = w0.end + max_ungrounded_gap;
            if (w1.start > new_start) {
                w1.start = new_start;
                if (w1.end < w1.start) {
                    w1.end = w1.start;
                }
                clamped++;
            }
        }
        if (params.print_timing && clamped > 0) {
            fprintf(stderr, "VAD sanity check: clamped %d ungrounded timestamp gap(s)\n", clamped);
        }
    }

    if (params.print_timing) {
        fprintf(stderr, "\nCombined Timing:\n");
        fprintf(stderr, "  ASR:           %lld ms\n", (long long) total_asr_ms);
        fprintf(stderr, "  Alignment:     %lld ms\n", (long long) total_align_ms);
        fprintf(stderr, "  Total:         %lld ms\n", (long long) full_align_result.t_total_ms);
        fprintf(stderr, "  Words aligned: %zu\n", full_align_result.words.size());
    }

    std::string string_output = params.output_srt ? alignment_to_srt(full_align_result) : alignment_to_json(full_align_result);

    if (params.output_path.empty()) {
        printf("%s\n", string_output.c_str());
    } else {
        std::ofstream out(params.output_path);
        if (!out) {
            fprintf(stderr, "Error: Failed to open output file: %s\n", params.output_path.c_str());
            return 1;
        }
        out << string_output << "\n";
        fprintf(stderr, "Output written to: %s\n", params.output_path.c_str());
    }

    if (params.profile) {
        QWEN3_TIMER_REPORT();
    }

    return 0;
}

static void ggml_log_callback_quiet(enum ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static void configure_console_utf8() {
    std::setlocale(LC_ALL, ".UTF-8");
    // On Windows, ".UTF-8" only changes the codepage; LC_NUMERIC still follows the
    // user's regional settings. In locales that use ',' as the decimal separator,
    // this makes "%.3f"-style formatting (timestamps, JSON output) emit commas
    // instead of periods, which breaks strict JSON parsers. Force '.' regardless
    // of the user's region.
    std::setlocale(LC_NUMERIC, "C");
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

int main(int argc, char ** argv) {
    configure_console_utf8();
    ggml_log_set(ggml_log_callback_quiet, nullptr);

    cli_params params;
    
    if (!parse_args(argc, argv, params)) {
        fprintf(stderr, "\n");
        print_usage(argv[0]);
        return 1;
    }
    
    if (params.transcribe_align_mode) {
        return run_transcribe_and_align(params);
    }

    if (params.align_mode) {
        return run_alignment(params);
    } else {
        return run_transcription(params);
    }
}
