#include <forced_aligner.h>

#include <string>
#include <vector>

int main() {
    qwen3_asr::ForcedAligner aligner;
    std::vector<std::string> words;

    aligner.tokenize_with_timestamps("\xE4\xBD\xA0\xE5\xA5\xBD hello 123", words);
    if (words != std::vector<std::string>{"\xE4\xBD\xA0", "\xE5\xA5\xBD", "hello", "123"}) {
        return 1;
    }

    aligner.tokenize_with_timestamps("hello \xE4\xB8\x96\xE7\x95\x8C", words, "chinese");
    if (words != std::vector<std::string>{"hello", "\xE4\xB8\x96", "\xE7\x95\x8C"}) {
        return 1;
    }

    return 0;
}
