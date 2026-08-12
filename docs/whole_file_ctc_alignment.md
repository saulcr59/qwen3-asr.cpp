# Whole-File CTC Forced Alignment — Design

Status: **designed and measured, not implemented.** Everything below is backed by
measurements on a real 6:33 Japanese recording and a 1.97 h synthetic stress case.

## The problem this solves

The pipeline is: OpenAI `gpt-transcribe` returns **text with no timestamps**, and a
local wav2vec2 CTC aligner supplies the time codes.

The encoder is quadratic in audio length, so a long recording cannot be aligned in one
call. The current answer is to cut the audio into windows **and split the transcript
across them**, guessing how much text belongs to each window from a reading-speed
estimate.

That guess is the defect. Forced alignment must place every character it is given — it
cannot report "this text is not in this audio" — so when the estimate is off, lines land
in the wrong window and are confidently mis-timed.

Measured case: a line whose speech starts at **50.12 s** was placed at **54.56 s**, in
silence. The same aligner, given the correct audio region, placed it at **50.14 s**.
The aligner was never wrong; the bookkeeping around it was.

A series of heuristics exists purely to paper over this — `AcceptChunk`'s duration
ratio, an absolute slack, window sizing, a refine pass with a maximum shift. They are
all patches on a design that cannot be made robust, because any reading-speed estimate
eventually meets material that deviates from it.

## The insight

The quadratic cost is **only in the encoder**. The alignment itself is a dynamic program
whose cost is frames × characters, and whose memory can be made linear.

So: chunk the audio **only for encoding**, concatenate the resulting per-frame
log-probability matrices, and run **one global Viterbi** over the whole recording.

The question "which text belongs to which audio window" then does not exist.

## Evidence

Measured against a single-pass alignment of the same 393 s recording:

| | Single pass | Chunked encode + global Viterbi |
|---|---|---|
| Peak memory | 23.6 GB | **0.24 GB** |
| Wall clock | >600 s, unfinished | **155 s** |
| Median character error | — | **0 ms** |
| Characters within one frame (20 ms) | — | **94.7 %** |

Against the current text-partitioning pipeline, character starts move by **0.9 ms mean /
40 ms max**, versus the **4,440 ms** error partitioning produces.

**Chunk seams are invisible.** Characters within 1 s of a seam are 100 % identical
(n=26). The worst residual errors sit **more than 10 s from any seam** — they are not
caused by chunking. The largest (2.24 s) is the silence-absorption behaviour already
documented in `include/ctc_aligner.h:71-73`, reproducible on unchunked audio by adding
gaussian noise to the reference.

## Encoder chunking

**40 s chunks, 5 s overlap, seam at the overlap midpoint, no extra trimming, global
normalisation statistics.**

Every number is measured, not conventional:

- **Minimum overlap is 2×64 frames = 2.56 s.** The convolutional positional embedding
  (`src/ctc_aligner.cpp:497`, padding 64) makes frame *t* read frames `[t-64, t+63]`, so
  a seam must sit ≥64 frames inside both chunks. The measured degradation cliff falls at
  exactly 1.28 s = 64 frames.
- **Overlap beyond 5 s buys nothing**: hard cut 29.1 ms → 5 s overlap 15.2 ms → 10 s
  15.1 ms → 20 s 15.5 ms.
- **40 s is a flat optimum** on both error and wall clock (155 s for the recording;
  15 s→172 s, 30 s→157 s, 45 s→178 s, 60 s→189 s, 90 s→229 s).
- The conv receptive field is only 400 samples = 25 ms = 1.25 frames, so it is not the
  binding constraint — the positional convolution is.

**Global normalisation statistics are required.** `compute_emissions` normalises the
waveform per buffer (`src/ctc_aligner.cpp:563-574`), so chunks would each normalise by
their own statistics. On this recording per-45 s-chunk standard deviation varies 1.30×,
which alone costs little — but the model is genuinely gain-sensitive (a −6 dB shift
costs more than losing 20 s of context), so per-chunk normalisation is an uncontrolled
AGC whose error is bounded-small *here* and unbounded on material with level swings.

Note the normalisation is **affine-invariant**: pre-scaling the input cannot compensate,
because the function rescales whatever it is given. Sharing statistics requires an
overload accepting precomputed `(mean, inv_std)`.

**Keep the last chunk full length** (extend backwards, discard the extra). Short chunks
are systematically less blank-confident, and the uniform sign of that bias is what makes
it cancel in the Viterbi.

### Two traps, both measured

1. **Never concatenate emissions end-to-end.** A 40 s chunk yields **1999** frames, not
   2000 — `n_frames(n) = floor((n-400)/320)+1`. Naive concatenation loses one frame per
   boundary, dropping agreement from 83.7 % to **53.1 %** and raising error from 15.1 ms
   to 36.0 ms with a *single* boundary. Index absolutely: chunk starts must be multiples
   of 320 samples, and `abs_frame = start_sample/320 + t`.
2. **Do not reuse the per-chunk `ratio`** at `src/ctc_aligner.cpp:753`. For a 40 s chunk
   40/1999 = 20.01 ms against a true 20.00 ms hop — 20 ms of drift accumulated by the end
   of every chunk, the same magnitude as the errors being controlled. Derive times from
   the absolute frame index.

## Global Viterbi

**Two live rows in `double`, 1 bit per cell of backpointers, backtrack from the bits
alone.**

- **`double`, not `float`.** float32 still produces a bit-identical alignment at 2 h
  today, but the smallest real decision margin is ~0.01 nats against a differential error
  of 0.0041 — a safety factor of 2.4×, which the growth rate crosses at roughly **4
  hours**. Two `double` rows cost 640 KB at N=40,000. If f32 rows are wanted for cache
  reasons, per-row renormalisation is exactly value-preserving (both candidates at frame
  *t+1* come from row *t*, so a uniform row offset cancels); carry the offset in a
  `double`.
- **1 bit per cell suffices** — each cell has exactly two predecessors (stay / advance).
- **Backtrack must not need the emissions.** The current backtrack
  (`src/ctc_aligner.cpp:732-744`) recomputes `stayed`/`changed`, so it re-reads both the
  trellis *and* the emissions matrix. Explicit backpointers remove both dependencies,
  which is what makes streaming possible.
- **Keep the free-end `argmax`** (`:723-728`). Frames after `t_start` are simply not
  scored, which is what gives the last character a sensible end time.
- **Drop `POS_INF`** (`:711-712`) — see bugs below.

Verified end-to-end at 1.97 h / 21,744 characters: **6.8 s**, **0.96 GB**, alignment
identical to a `double` reference.

### Above ~45–60 min: checkpointed backtrack

Backpointer memory is quadratic in duration (`bytes ≈ 6.25·r·D²`). Checkpoint the full
`double` row every `K = 8√T` frames and backtrack block by block, recomputing each
block's bits from the nearest checkpoint. Memory becomes `2N√T` bytes; time roughly 2×
the forward pass.

| Duration | 1-bit single pass | Checkpointed |
|---|---|---|
| 6.5 min | 5.4 MB | 0.6 MB |
| 60 min | 450 MB | 17.0 MB |
| 120 min | 1.80 GB | **48.0 MB** |
| 480 min | 28.8 GB | **384.3 MB** |

Verified at 1.97 h: 75 checkpoints, 28.8 MB total, output **byte-identical** to the
single-pass reference (43,488/43,488 entries).

The recomputation needs the emissions a second time — either retain them reduced, or
retain the PCM (230 MB as int16 for 2 h) and re-encode.

**There is no audio length at which exactness must be abandoned.** The practical ceiling
is the encoder, not the decoder.

## Rejected: banded Viterbi

A band on predicted text position **is the reading-speed guess in a new coat**, and it
fails the same way: if the band is too narrow the true path is excluded and the aligner
returns a confident wrong answer with no error signal.

Measured on the real recording: the true path deviates −22.7 to +71.9 characters from a
constant-rate diagonal (6.0 % of N); the constant-rate guess has a mean per-character
error of **11.7 s** and a maximum of 23.1 s; inter-character gaps reach **9.74 s**.

A *score* band (beam) is data-driven and startlingly narrow — the true path never runs
more than 19.06 nats below the row maximum, so an 11-state beam would have been exact
here against 1,209 states. But there is no bound on that 19.06, and the cases that push
the true path down are exactly the ones that matter: a mis-heard proper noun, a dropped
line, a transcript that disagrees with the audio. A beam commits locally and silently.

Neither is needed: the exact schemes above fit comfortably.

## Honest limitation

The concatenated emission matrix **depends on where the cuts fall**. Comparing 40 s vs
60 s encoding of the same audio: argmax flips in 1.40 % of frames, spread across the
whole window rather than at edges, and the greedy decode differs (129 vs 127 characters).

The global Viterbi is exact *for the matrix it is given*, and that matrix is
chunk-dependent. Aligning the same transcript against both matrices moves character
starts by mean 0.9 ms / max 40 ms — two orders of magnitude better than the 4,440 ms the
current design produces, but **do not claim bit-exactness** against a hypothetical
single global encode.

## Bugs found during this analysis

Independent of this design; present in shipping code.

1. **Out-of-bounds read.** The Ivydata GGUF declares `vocab_size = 2938` / `blank = 2937`
   but ships **2943** tokenizer tokens (2938 `<s>`, 2939 `</s>`, 2940 `|`, 2941-2942 raw
   JSON). `parse_hparams` (`src/ctc_aligner.cpp:203-205`) inserts all of them, so a
   transcript containing a literal `|` yields `tokens[i] = 2940` and `emis(t, 2940)`
   (`:692`) indexes past the row — and past the end of the vector on the last frame.
   `wildcard_id` (`:675`) `= n_vocab = 2938` also collides with `<s>`.
   **Fix:** reject or clamp ids `>= vocab_size` when building `token_to_id_`.
2. **The wildcard column allocates a second full emissions matrix** (`:659-674`):
   `extended` lives alongside `emissions` for the whole copy loop. 8.46 GB peak at 2 h to
   append one column per frame, and it fires whenever the transcript contains any
   character outside the CTC vocabulary — i.e. essentially always, since punctuation
   triggers it. **Fix:** compute the wildcard value per frame as it streams.
3. **`POS_INF` is dead code** (`:711-712`). Written before the forward loop, it marks the
   infeasible triangle `t - j >= T+1-N`. The backtrack starts outside that region and
   preserves the invariant, so those cells are never read. Measured: 0 `+inf` reads and
   0/1125 characters changed when deleted, across 4,000 randomised trials. It is also a
   latent NaN source (`+inf + -inf`). The `first_frame[i] < 0` fallback at `:760-764` is
   likewise unreachable. **Fix:** drop both.

## Implementation plan

**Phase 0 — the out-of-bounds read.** Small, independent, worth doing on its own.

**Phase 1 — C++.** New `--align-chunked` mode: overlapped chunked encoding, streaming DP
with `double` rows and 1-bit backpointers, wildcard computed on the fly, no `POS_INF`.
Leave `--align` byte-identical. Emit `align-progress: <done> <total>` on **stderr**
(stdout carries the JSON when `-o` is omitted), plus a distinct marker when encoding ends
and the Viterbi begins — that phase has no progress hook and will otherwise look hung.

*Verify:* same token count and per-token times within ~40 ms of `--align` on the 60 s
fixture; then peak RSS and wall clock on a 20-minute file.

**Phase 2 — C#.** Extract `ApplyTimeCodes` into a shared `ForcedAlignTimeCodes` (pure
refactor, no test edits). Add `IWholeFileRunner` + `WholeFileForcedAligner` as a parallel
class. Wire only `SpeechToTextViewModel.TryRefineTimingsWithForcedAlignerAsync` to prefer
it, **with a mandatory fallback** to the existing windowed path.

Do **not** add a flag inside `ForcedAligner` branching between windowed and global: every
heuristic in that class is coupled to the loop, and a flag would leave the crispasr path
one `if` away from every new bug. `ForcedAligner`, `ForcedAlignPlanner`,
`CrispAsrAlignOnlyRunner` and `ImportPlainTextViewModel` are not otherwise touched.

The fallback is mandatory, not optional, because failure becomes all-or-nothing: `T < N`
and "backtrack failed to reach the start" currently kill one window and the loop steps
on; globally they kill the entire recording. An over-long transcript from
`gpt-transcribe` is a live scenario.

Add a **character-accounting guard**: total visible characters across all lines must
equal the total across all returned tokens, else fall back. Today a miscount corrupts one
window; globally it shifts every remaining line in the file.

Detect `Error: Unknown argument:` on stderr specifically and fall back silently — there
is no CLI version handshake, so an older binary is expected in the wild.

**Phase 3 — checkpointed backtrack.** Only if recordings beyond ~an hour matter. At 6:33
the simple scheme uses 5.4 MB.

*Verify after each phase against `tests/UI/Features/Files/ImportPlainText/ForcedAlignerLiveRepro.cs`
in the Subtitle Edit fork, and add the measured regression as an assertion: the line whose
speech starts at 50.12 s must land within ~0.3 s of it. That single test proves the change
did what it was for.*

## What can be retired afterwards

Nothing immediately — crispasr and the fallback both need the windowed machinery. After a
soak on real material: `ForcedAlignPlanner.ChunkSize`, `AcceptChunk`,
`ForcedAligner.RefineAsync`, the tail-rejoin block, and the options `WindowSeconds`,
`LinesPerChunk`, `MaxChunkShareOfWindow`, `RefineBatchLines`, `RefinePaddingSeconds`,
`MaxRefineShiftSeconds`, `SkipSeconds`, `MaxDurationSlackSeconds`.

`TrustMeasuredDurations` and `LingerSeconds` are orthogonal to windowing and carry over
verbatim, as does `Qwen3AsrAlignOnlyRunner`'s `SpanOf` silence trimming — the aligner
still absorbs silence into the preceding character by design.
