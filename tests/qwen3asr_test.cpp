#include <iostream>
#include <string>
#include <qwen3_asr.h>
#include <forced_aligner.h>

using namespace std;

static void info(const string & text) {
    cerr << text << endl;
}

static int fail(const string & text) {
    cerr << "ERROR: " << text << endl;
    return 1;
}

int main() {
    /*
    curl -L -o assets/qwen3-asr-0.6b-q8_0.gguf https://huggingface.co/Jaffe2718/Qwen3-ASR-GGUF/resolve/main/qwen3-asr-0.6b-q8_0.gguf?download=true
    curl -L -o assets/qwen3-forcedaligner-0.6b-f16.gguf https://huggingface.co/Jaffe2718/Qwen3-ASR-GGUF/resolve/main/qwen3-forcedaligner-0.6b-f16.gguf?download=true
    curl -L -o assets/qwen3-asr-1.7b-q4_1.gguf https://huggingface.co/Jaffe2718/Qwen3-ASR-GGUF/resolve/main/qwen3-asr-1.7b-q4_1.gguf?download=true
    curl -L -o assets/jfk.wav https://raw.githubusercontent.com/ggml-org/whisper.cpp/refs/heads/master/samples/jfk.wav
     */
    qwen3_asr::Qwen3ASR asr;
    if (!asr.load_model("assets/qwen3-asr-0.6b-q8_0.gguf")) {
        return fail(asr.get_error());
    }
    info("qwen3-asr-0.6b-q8_0.gguf loaded");
    qwen3_asr::transcribe_result result = asr.transcribe("assets/jfk.wav");
    if (!result.success) {
        return fail(result.error_msg);
    }
    info(result.text);
    info("-----------------");

    qwen3_asr::Qwen3ASR asr1_7;
    if (!asr1_7.load_model("assets/qwen3-asr-1.7b-q4_1.gguf")) {
        return fail(asr1_7.get_error());
    }
    info("qwen3-asr-1.7b-q4_1.gguf loaded");
    qwen3_asr::transcribe_result result1_7 = asr1_7.transcribe("assets/jfk.wav");
    if (!result1_7.success) {
        return fail(result1_7.error_msg);
    }
    info(result1_7.text);
    info("-----------------");

    qwen3_asr::ForcedAligner aligner;
    if (!aligner.load_model("assets/qwen3-forcedaligner-0.6b-f16.gguf")) {
        return fail(aligner.get_error());
    }
    info("qwen3-forcedaligner-0.6b-f16.gguf loaded");
    qwen3_asr::alignment_result align_result = aligner.align("assets/jfk.wav", result.text);
    if (!align_result.success) {
        return fail(align_result.error_msg);
    }
    for (auto & word : align_result.words) {
        cerr << "`" << word.word << "` (" << word.start << " --> " << word.end << ")" << endl;
    }
    info("-----------------");
    return 0;
}
