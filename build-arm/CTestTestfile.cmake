# CMake generated Testfile for 
# Source directory: /Users/nikolajolsson/git/qwen3-asr.cpp
# Build directory: /Users/nikolajolsson/git/qwen3-asr.cpp/build-arm
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(mel_spectrogram_test "/Users/nikolajolsson/git/qwen3-asr.cpp/build-arm/test_mel")
set_tests_properties(mel_spectrogram_test PROPERTIES  WORKING_DIRECTORY "/Users/nikolajolsson/git/qwen3-asr.cpp" _BACKTRACE_TRIPLES "/Users/nikolajolsson/git/qwen3-asr.cpp/CMakeLists.txt;288;add_test;/Users/nikolajolsson/git/qwen3-asr.cpp/CMakeLists.txt;0;")
add_test(audio_encoder_test "/Users/nikolajolsson/git/qwen3-asr.cpp/build-arm/test_encoder")
set_tests_properties(audio_encoder_test PROPERTIES  WORKING_DIRECTORY "/Users/nikolajolsson/git/qwen3-asr.cpp" _BACKTRACE_TRIPLES "/Users/nikolajolsson/git/qwen3-asr.cpp/CMakeLists.txt;292;add_test;/Users/nikolajolsson/git/qwen3-asr.cpp/CMakeLists.txt;0;")
add_test(text_decoder_test "/Users/nikolajolsson/git/qwen3-asr.cpp/build-arm/test_decoder")
set_tests_properties(text_decoder_test PROPERTIES  WORKING_DIRECTORY "/Users/nikolajolsson/git/qwen3-asr.cpp" _BACKTRACE_TRIPLES "/Users/nikolajolsson/git/qwen3-asr.cpp/CMakeLists.txt;296;add_test;/Users/nikolajolsson/git/qwen3-asr.cpp/CMakeLists.txt;0;")
add_test(audio_injection_test "/Users/nikolajolsson/git/qwen3-asr.cpp/build-arm/test_injection")
set_tests_properties(audio_injection_test PROPERTIES  WORKING_DIRECTORY "/Users/nikolajolsson/git/qwen3-asr.cpp" _BACKTRACE_TRIPLES "/Users/nikolajolsson/git/qwen3-asr.cpp/CMakeLists.txt;300;add_test;/Users/nikolajolsson/git/qwen3-asr.cpp/CMakeLists.txt;0;")
