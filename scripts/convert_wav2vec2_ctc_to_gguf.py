#!/usr/bin/env python3
"""
Convert a HuggingFace wav2vec2 CTC model to GGUF for use as a forced aligner.

Only the *timing* stage uses this model: the transcript text still comes from
Qwen3-ASR, and this model is force-aligned against it with a monotonic CTC
Viterbi pass. That guarantees per-character timings can never contain the
unexplained multi-second gaps the autoregressive Qwen aligner sometimes emits.

Any Wav2Vec2ForCTC checkpoint works as long as do_stable_layer_norm=True (the
transformer encoder must be the pre-norm variant). Either feat_extract_norm
variant is supported: "layer" (LayerNorm on every conv layer, e.g. the XLSR
family) or "group" (GroupNorm on conv layer 0 only, no normalisation on the
rest, e.g. the ReazonSpeech wav2vec2 family). Pick one whose tokenizer
vocabulary covers the script you align - for Japanese,
jonatasgrosman/wav2vec2-large-xlsr-53-japanese includes kanji, not just kana.

A checkpoint's config declares its CTC blank as tokenizer.pad_token_id, and
that is the default here. Verify it is actually correct before trusting it:
some released checkpoints never confidently predict that id at all, favouring
a different token as their de facto blank instead (confirmed on
reazon-research/japanese-wav2vec2-large-rs35kh, where >95% of frames land on
<unk> and effectively 0% land on the declared pad token - reproducible with
transformers' own Wav2Vec2Processor.batch_decode, so it's a property of the
released weights, not of this converter or the C++ aligner). Check with a
short forward pass on real audio (argmax token-id histogram) before
converting, and pass --blank-id to override if the declared pad token isn't
actually the dominant one.

Usage:
    python scripts/convert_wav2vec2_ctc_to_gguf.py \
        --input jonatasgrosman/wav2vec2-large-xlsr-53-japanese \
        --output models/wav2vec2-ctc-ja-f16.gguf \
        --type f16

    python scripts/convert_wav2vec2_ctc_to_gguf.py \
        --input reazon-research/japanese-wav2vec2-large-rs35kh \
        --output models/wav2vec2-ctc-ja-reazon-large-f16.gguf \
        --type f16 --blank-id 0
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import torch

import gguf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ARCH = "wav2vec2-ctc"

# Tensors kept at full precision: normalisation parameters and biases are tiny
# and precision-sensitive, so quantising them buys nothing and costs accuracy.
F32_SUFFIXES = (".bias", "_norm.weight", "norm.weight")


def is_f32_tensor(name: str) -> bool:
    return name.endswith(F32_SUFFIXES) or ".norm" in name


def add_tensor(writer: gguf.GGUFWriter, name: str, tensor: torch.Tensor, want_f16: bool) -> None:
    arr = tensor.detach().to(torch.float32).cpu().numpy()
    # ggml reads ne[0] as the fastest-varying axis, which is the reverse of numpy's
    # order - so PyTorch's own layouts already land correctly: Linear (out, in)
    # becomes ne [in, out] for ggml_mul_mat, and Conv1d (out, in, k) becomes
    # ne [k, in, out] for ggml_conv_1d. No transposes needed.
    if want_f16 and not is_f32_tensor(name) and arr.ndim > 1:
        arr = arr.astype(np.float16)
    else:
        arr = arr.astype(np.float32)
    writer.add_tensor(name, arr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="HF model id or local path")
    ap.add_argument("--output", required=True, help="output .gguf path")
    ap.add_argument("--type", choices=["f16", "f32"], default="f16")
    ap.add_argument("--blank-id", type=int, default=None,
                     help="override the CTC blank token id (default: tokenizer.pad_token_id). "
                          "Some checkpoints never confidently predict the declared pad token - "
                          "see the module docstring.")
    args = ap.parse_args()

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    logger.info("loading %s", args.input)
    processor = Wav2Vec2Processor.from_pretrained(args.input)
    model = Wav2Vec2ForCTC.from_pretrained(args.input).eval()
    cfg = model.config

    if cfg.feat_extract_norm not in ("layer", "group") or not cfg.do_stable_layer_norm:
        logger.error(
            "unsupported variant: feat_extract_norm=%s do_stable_layer_norm=%s "
            "(the C++ side implements the stable-layer-norm encoder only, with "
            "either the layer-norm or group-norm feature extractor)",
            cfg.feat_extract_norm, cfg.do_stable_layer_norm,
        )
        return 1

    want_f16 = args.type == "f16"
    writer = gguf.GGUFWriter(path=None, arch=ARCH)

    # ---- hyper-parameters -------------------------------------------------
    writer.add_uint32(f"{ARCH}.embedding_length", cfg.hidden_size)
    writer.add_uint32(f"{ARCH}.block_count", cfg.num_hidden_layers)
    writer.add_uint32(f"{ARCH}.attention.head_count", cfg.num_attention_heads)
    writer.add_uint32(f"{ARCH}.feed_forward_length", cfg.intermediate_size)
    writer.add_float32(f"{ARCH}.attention.layer_norm_epsilon", cfg.layer_norm_eps)
    writer.add_uint32(f"{ARCH}.vocab_size", cfg.vocab_size)

    writer.add_uint32(f"{ARCH}.conv.count", len(cfg.conv_dim))
    writer.add_array(f"{ARCH}.conv.dim", [int(x) for x in cfg.conv_dim])
    writer.add_array(f"{ARCH}.conv.stride", [int(x) for x in cfg.conv_stride])
    writer.add_array(f"{ARCH}.conv.kernel", [int(x) for x in cfg.conv_kernel])

    writer.add_uint32(f"{ARCH}.pos_conv.kernel", cfg.num_conv_pos_embeddings)
    writer.add_uint32(f"{ARCH}.pos_conv.groups", cfg.num_conv_pos_embedding_groups)

    writer.add_uint32(f"{ARCH}.sample_rate", processor.feature_extractor.sampling_rate)
    blank_id = args.blank_id if args.blank_id is not None else processor.tokenizer.pad_token_id
    if args.blank_id is not None:
        logger.info("blank token id overridden to %d (tokenizer.pad_token_id is %d)",
                    blank_id, processor.tokenizer.pad_token_id)
    writer.add_uint32(f"{ARCH}.blank_token_id", blank_id)
    writer.add_string(f"{ARCH}.feat_extract_norm", cfg.feat_extract_norm)

    # ---- CTC vocabulary ---------------------------------------------------
    # Stored index-ordered so the C++ side can build id -> token directly; the
    # aligner needs the reverse (token -> id) and builds it from this.
    vocab = processor.tokenizer.get_vocab()
    tokens = [""] * (max(vocab.values()) + 1)
    for tok, idx in vocab.items():
        tokens[idx] = tok
    writer.add_token_list(tokens)
    logger.info("vocab: %d tokens, blank id %d", len(tokens), blank_id)

    # ---- tensors ----------------------------------------------------------
    w2v = model.wav2vec2

    # feat_extract_norm="layer": every conv layer has its own LayerNorm.
    # feat_extract_norm="group": only conv layer 0 has a norm (GroupNorm with
    # num_groups == num_channels, i.e. per-channel normalisation across time);
    # layers 1+ are plain conv+activation with no normalisation at all, so
    # they have no `layer_norm` submodule to export.
    for i, layer in enumerate(w2v.feature_extractor.conv_layers):
        add_tensor(writer, f"conv.{i}.weight", layer.conv.weight, want_f16)
        # conv_bias=False (seen on the ReazonSpeech group-norm family) leaves
        # this None - ggml's conv_1d has no bias term of its own either way.
        if layer.conv.bias is not None:
            add_tensor(writer, f"conv.{i}.bias", layer.conv.bias, want_f16)
        if hasattr(layer, "layer_norm"):
            add_tensor(writer, f"conv.{i}.norm.weight", layer.layer_norm.weight, want_f16)
            add_tensor(writer, f"conv.{i}.norm.bias", layer.layer_norm.bias, want_f16)

    add_tensor(writer, "feat_proj.norm.weight", w2v.feature_projection.layer_norm.weight, want_f16)
    add_tensor(writer, "feat_proj.norm.bias", w2v.feature_projection.layer_norm.bias, want_f16)
    add_tensor(writer, "feat_proj.weight", w2v.feature_projection.projection.weight, want_f16)
    add_tensor(writer, "feat_proj.bias", w2v.feature_projection.projection.bias, want_f16)

    # Reading .weight materialises the weight-norm parametrisation (g * v/||v||),
    # so the C++ side receives a plain convolution kernel.
    add_tensor(writer, "pos_conv.weight", w2v.encoder.pos_conv_embed.conv.weight, want_f16)
    add_tensor(writer, "pos_conv.bias", w2v.encoder.pos_conv_embed.conv.bias, want_f16)

    add_tensor(writer, "encoder.norm.weight", w2v.encoder.layer_norm.weight, want_f16)
    add_tensor(writer, "encoder.norm.bias", w2v.encoder.layer_norm.bias, want_f16)

    for i, blk in enumerate(w2v.encoder.layers):
        p = f"blk.{i}"
        add_tensor(writer, f"{p}.attn_q.weight", blk.attention.q_proj.weight, want_f16)
        add_tensor(writer, f"{p}.attn_q.bias", blk.attention.q_proj.bias, want_f16)
        add_tensor(writer, f"{p}.attn_k.weight", blk.attention.k_proj.weight, want_f16)
        add_tensor(writer, f"{p}.attn_k.bias", blk.attention.k_proj.bias, want_f16)
        add_tensor(writer, f"{p}.attn_v.weight", blk.attention.v_proj.weight, want_f16)
        add_tensor(writer, f"{p}.attn_v.bias", blk.attention.v_proj.bias, want_f16)
        add_tensor(writer, f"{p}.attn_out.weight", blk.attention.out_proj.weight, want_f16)
        add_tensor(writer, f"{p}.attn_out.bias", blk.attention.out_proj.bias, want_f16)
        add_tensor(writer, f"{p}.attn_norm.weight", blk.layer_norm.weight, want_f16)
        add_tensor(writer, f"{p}.attn_norm.bias", blk.layer_norm.bias, want_f16)
        add_tensor(writer, f"{p}.ffn_up.weight", blk.feed_forward.intermediate_dense.weight, want_f16)
        add_tensor(writer, f"{p}.ffn_up.bias", blk.feed_forward.intermediate_dense.bias, want_f16)
        add_tensor(writer, f"{p}.ffn_down.weight", blk.feed_forward.output_dense.weight, want_f16)
        add_tensor(writer, f"{p}.ffn_down.bias", blk.feed_forward.output_dense.bias, want_f16)
        add_tensor(writer, f"{p}.ffn_norm.weight", blk.final_layer_norm.weight, want_f16)
        add_tensor(writer, f"{p}.ffn_norm.bias", blk.final_layer_norm.bias, want_f16)

    add_tensor(writer, "lm_head.weight", model.lm_head.weight, want_f16)
    add_tensor(writer, "lm_head.bias", model.lm_head.bias, want_f16)

    logger.info("writing %s", args.output)
    writer.write_header_to_file(path=args.output)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
