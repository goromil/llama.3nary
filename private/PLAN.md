# PLAN - updated plan (start at the DSV4.1 TERNARY section, 11-02)

Status markers: [x] done, [ ] open, [B] blocked on user.

## DSV4.1 TERNARY (11-02, ACTIVE - start here)

Project: requantize DeepSeek-V4.1-Flash-1017 (Unsloth UD-Q4_K_XL,
9 shards, 501.68 GB, 467 GiB) to the fork's PTQ1_0 + PQ2_0 and
port the missing deepseek41 arch from the upstream clone
(C:\Users\admin\source\ai\llama.cpp, final state after
a07421535, read-only git show, never touch its working tree).
Size + fit + perf + quality expectations and the Unsloth IQ1/2
+ PrismML Bonsai comparison: DESIGN-dsv41-ternary-expectations.md.

Locked user decisions (11-02):
A. Ternarize the engram table (per_layer_token_embd, 196.61B
   params): PTQ1_0 43.0 GB / PQ2_0 52.2 GB (not the upstream
   BF16 escape hatch).
B. Produce BOTH PTQ1_0 and PQ2_0.
C. Keep the 9-shard split (1 = KV, 2-7 = experts, 8 = table,
   9 = misc).
D. Port the dspark dflash.cpp hunk now (+~263 ln, P0 item 5).

Source facts (measured 11-02, byte-exact, fork GGUF variant):
748.5B params = 543.58B experts (120 MXFP4 tensors type 39
[884736, 5120] @ 4.25 bpw = 4.5298B params each, shards 2-7,
48.13 GB/shard = 288.78 GB) + 8.19B Q8_0 (351 tensors, 8.69 GB,
incl engram_kv [6144, 25600] blk.1/14, output/token_embd
[5120, 129280]) + 196.61B table (shard 8, per_layer_token_embd
[256, 768022850], file type field 43 = MXFP8 33 B/32, 202.76 GB)
+ 0.12B F32 (571 tensors, 0.47 GB, stays F32). Engram tensors
only in blk.1/blk.14 (engram_gate.weight [5120, 4] F32). All
tensor dims 128-multiples. Arch: deepseek41, 40 blocks, 384
experts / 6 used, embd 5120, ctx 1048576, n_vocab 129280,
hc_mult 4. Cross-check: our 297.94 + 202.76 GB = Unsloth's
claimed "Rest 297.9" + "MXFP8 202.8" (exact).

P0 [ ] port into the fork (commit order):
1. ggml MXFP8 = 144 (33 B/32, COUNT -> 145) + dequant (modeled
   on the fork MXFP4 dequant) + type_traits.
2. gguf load-time remap: file type field 43 -> MXFP8, guarded
   by row-size verification (the fork cannot read type-43
   files today, so no ambiguity).
3. deepseek41 arch: llama-arch.{cpp,h} (arch + 2 new
   LLM_TENSOR engram keys + 5 new indexer KV keys + all 92
   model KVs), llama-hparams.h (net-new PLE section incl
   ple_n_modules + dsv4_hc_single_pass), models.h, NEW
   src/models/deepseek41.cpp (adapted from
   %TEMP%\opencode\up-deepseek41-final.cpp, f32_src param
   removed, engram_pad_id in hparams, engram_token_map in
   llama_model), llama-model.{h,cpp}, kv-cache-dsv4.{h,cpp}
   (+279/+17: comp_il_src, mirror copy, ngram token history),
   deepseek4.cpp build_hc_mixes split (no f32_src),
   llama-context.cpp + llama-graph.cpp dsv41 branches (moe
   weights_sum: ggml_scale_bias(1.0f, 1e-20f) instead of the
   F16 clamp), 3 ggml log1pf softplus fixes, chat template
   V4.1. Skip: conversion/deepseek41.py, convert_hf_to_gguf.py,
   gguf-py hunks, qwen4exp.cpp hunk, tests.
4. llama-quant.cpp: engram_gate.weight exclusion + accept
   MXFP8 (and F8_E4M3, upstream 785322d17 pattern) as
   requantize inputs. No TENSOR_READ_LAZY port (mmap proven
   for the 288 GB experts; the table follows the same path).
5. dflash.cpp dspark hunk (+~263) + parsers.
Reference material in %TEMP%\opencode\: up-deepseek41-final.cpp
(809 ln), up-deepseek4-{pre,final}.cpp, dsv41-1.diff (2752 ln),
dsv41-2.diff (393 ln, fully read), fork-vs-up-deepseek4.diff
(255 ln, fully read). UNREAD before coding: dsv41-1.diff
:540-1374 (kv-cache-dsv4/.h, llama-kv-cache.cpp, model-loader,
model-saver, model.{cpp,h}, quant.cpp) + :2330-2739 (dflash.cpp
+ models.h hunks); chat.cpp + parsers hunks :1-251.

P1 [ ] requant: plain PTQ, NO imatrix,
llama-quantize --allow-requantize, both targets, 9-shard split
preserved. Tensor map: 120 MXFP4 experts + all 351 Q8_0 ->
PTQ1_0/PQ2_0; all 571 F32 stay F32. Table path is chunked
with bounded memory: per-row f32 staging 3.07 GB, output
staging 43.0/52.2 GB (fits 64 GB RAM), streaming writes.
Expected outputs: PTQ1_0 164.2 GB (152.8 GiB, 1.76 bpw),
PQ2_0 199.3 GB (185.3 GiB, 2.13 bpw) - addendum.

P2 [ ] validate: fit verdict below forbids full residency on
this box, so hybrid offload (GPU ~50-54 GB weights, rest +
table on CPU, small ctx, live service stopped); load +
generate + compare vs the live UD-Q4_K_XL service on fixed
prompts; bench vs N10 anchors; check the t/s + quality ranges
in the addendum. DSpark drafter (separate Q8_0 7.98 GB / BF16
8.51 GB files, not local) optional follow-up for the 1.3-1.5x
decode multiplier.

P2 staged 11-03: recipe complete in private/P2-RUNBOOK.md -
flag-verified launch lines (KV F16/F16 + FA auto default,
-ot ...=CPU, -sm layer, -fit on), paper fit PTQ1_0 -ngl 22
(62.0 GiB GPU + 50.8 GiB CPU + 40 GiB table on disk) / PQ2_0
-ngl 18 (61.5 GiB GPU, ~16 GiB backbone paged out), KV at
-c 32768 ~100 MB (non-constraint), 7-prompt fixed A/B set +
E:\tmp\ab-dsv41.ps1 capture staged (step 0 baseline runs
against the live 8080 without a stop; stop window = step 1
only). Awaiting the user's stop window.

Fit verdict for THIS machine (2 x 32 GB B70 = 64 GB VRAM,
63.9 GB RAM, 128 GB total): A) all-on-GPU 152.8 GiB: NO
(~2.4x over). B) model on GPU + table on CPU: 112.8 GiB
weights: still NO (~1.8x over). Any fully-resident split:
152.8 GiB > 128 GB total -> not runnable fully-resident; only
a mixed GPU/CPU split with a small ctx and the live service
stopped, decode then CPU-bandwidth-bound. The engram table
per-token access is KB-scale (a few gathered rows per token),
so table off-GPU/CPU/SSD is the intended design (Unsloth ships
the same offload: --override-tensor ...=CPU + mmap), not a
bandwidth problem. Unsloth's own floor for 1-bit serving is
128-145 GB RAM+VRAM.

## 11-01 items (background, still open - superseded as the
start point by the DSV4.1 TERNARY section above)

Context: B12 tensor-mode pivot (10-23, 7cb134632) and the KV wall two
stages (10-25, 6a30f9157 + db5cf4d80) are SHIPPED; B15 adaptive ctx +
B17 handover (10-27/28) SHIPPED. B19 (fit settle, F40) + B20 (two-pass
grow) + the tensor-mode draft fit are COMMITTED (c9c4e20f1) and built,
NOT deployed. B14 (twin-server + router) is REJECTED 10-28 by the user
(final, dead end) - the record is BACKLOG B14 + F39, no re-presentation.
11-01 probes closed the batch-6144 memory question (F40 addendum 1);
10-31 crash/hang forensics in F40 addendum 2; the co-resident CLI
batteries v4+v5 (ternary+MTP live-verified on B19+fit) in F40
addendum 3.

1. [B] DEPLOY + verify (needs an 8080 stop window): deploy to macube
   (build\bin re-linked 11-01, b10753-593599d06), then the
   deterministic live growth-WARN catch (B20). The tensor + draft
   fit sanity run is COVERED by the 11-01 co-resident batteries
   (fit on by default, passed 16384-262144, F40 addendum 3).
2. [x] T1 cold test: RETIRED 11-01 - the user rejected the stop
   window ("try without stop/starting 8080"); superseded by the
   co-resident CLI batteries (F40 addendum 3). The release-lag
   theory is NOT falsified (the batteries had no kill-churn) and
   joins the 10-31 law under item 2b.
2b. [x] pre-B19 A/B (F40 addendum 3, 11-01): the deployed b10750-
   4f4116a3a build ran standalone on port 8123, -sm layer, PTQ1_0
   + mtp-Q4_0, -c 262144, 4 slots, one HTTP completion - NO
   DEVICE_LOST, no hang, clean teardown, service healthy (E:\tmp\
   ab1031.ps1 + ab1031-server.err). Binary and static split
   variables ruled out for the 10-31 law; the pre-B19 demotion
   anatomy captured (main 265 MiB -> CPU, draft 774 MiB SYCL1 +
   521 MiB SYCL_Host, pp 2.62 t/s). The law is now tied to the
   churn variable only (release-lag theory); testing it needs a
   macube restart window - [B] if the user offers one, otherwise
   wait for a natural occurrence with log capture.
3. [ ] B7a (next code item in this fork): startup loud-fail on
   unfittable ctx/KV + closed --dev gate. Target: the SPLIT_MODE_TENSOR
   fit path + the -sm/-dev parse area (re-locate lines, the 10-25
   numbers are stale). Note: B19 changed what "unfittable" means -
   over-budget now settles down, so B7a's fail surface is the settled
   floor + the pinned-ctx deficit report.
4. [ ] B13 recon (read-only, clone hosts live service): diff fork vs
   clone meta backend, ternary enum mapping, missing/divergent work
   map (F25), seq_rm question (F28). THEN port commit-by-commit, scope
   user-confirmed.
5. [ ] N17 Q5_K_S tensor bench (GDN correctness check first): still
   user-run; no free card while the service holds both - run co-
   resident (11-01 probe style, variance caveat) or in a stop window.
6. [B] bad_alloc repro: needs the user's real workload + the next
   occurrence with preserved log context (logs truncate, the catch
   site is lost). Proposals on the table: detached log-tail watcher
   (E:\tmp only) / top-level try-catch + stack trace. Await user call.
7. [B] TG-status line discrepancy: the user's display shows a TG line
   that -lv 2 should suppress (SLT_INF, server-context.cpp:594);
   blocked on a sample line from the user.

## 10-25 items - resolved / retired (reference)

- 1-4 [x] KV wall stage 1 (6a30f9157) + stage 2 -fitm (db5cf4d80) +
  SYCL=OFF root cause (F30) + private/ refresh: done, pushed.
- 5 [x] B14 PRESENT (twin-server + router draft): REJECTED 10-28 by
  the user (final: "I will never start. It was dead end."). Tensor
  split and layer split both work in multi-chat; the successor is the
  parked B18 (steady-state decode pipeline). Record: BACKLOG B14 +
  F39 + STATE. No re-presentation.
- 6 [x] REPORT to the user (9B MTP fix gone from tree, SYCL fix,
  stage-2 grow): folded into the continuous 10-26..11-01 docs sync;
  the tree fact re-verified 11-01 (src/models/qwen35.cpp still matches
  HEAD, the 9B MTP fix still absent).
- 7 [x] extraArgs bisection (-ngl 99 / -np 2 / -c 262144): outcome
  never reported; superseded - F39 settled the -np 2 question (drop
  it, user-confirmed 10-27/28) and the 10-31 forensics took over the
  crash/hang work. Nothing left to fold.
- 8: Q5_K_S tensor bench: carried to open item 5 above.

## PARKED (do not start without user re-scoping, 10-22)

- B7 preemption C0/C1/C2: CLOSED 10-22 (twin-server data-parallel
  design DROPPED; C0/C1 DROPPED per user). Plan intact in F11 for the
  record only.
- B7a (startup-side loud-fail on unfittable ctx/KV + closed --dev
  gate): APPROVED 10-22, NOT parked - open item 3 above.
- B5 free-VRAM suite / B6 Part-3 trigger / B8 PQ2_0 sweep / B9 final
  report / B10 small follow-ups: status as in BACKLOG.md (parked /
  open-low). The final report (B9) should include the tensor-mode work
  when it lands.
- B18 (alternating pipeline / steady-state decode): DRAFT parked,
  user-requested 10-28 (DESIGN-alternating-pipeline.md), no code.

## DONE (reference)

- [x] 10-19/10-20: KV ring overcommit guard + diagnostic (a3f36885c,
  355b4a2b3); MTP 3-part fix (0589b1e1a); PTQ1_0 SYCL byte-parallel
  kernel B11 (bd993a9d7 + 033a6f0ad + 5b80a3bb1, tg128 6.90 -> 26.72);
  fit-params draft-aware fix (48e9082ca); private/ knowledge library
  (02af668bf, e844b224a, 11c3d328a, 7faf42ec4, 835d728b3).
- [x] 10-22: tensor-mode root cause pinned (F20), meta model mapped
  (F21), H-B dead (F22), H-A designed (F23), PQ2_0/PTQ1_0 codec facts
  (F24), port-target clone located (F25). F15-F18 reconstructed/retained.
- [x] 10-22 (evening): user decisions recorded - H-A APPROVED (3nary
  first), B7a APPROVED, B7 preemption CLOSED; clone git state verified
  (no local commits, 3 user mods to preserve). Docs: BACKLOG (B7a),
  STATE, PLAN refreshed.
- [x] 10-23: B12 H-A fold implemented + verified + committed
  (7cb134632) + unit test (tests/test-hadamard-fold.cpp); ssm_out
  gather bug found + fixed (F27); PQ2_0 + PTQ1_0 tensor-verified;
  F29 (H-C kernel fusion) documented as fallback.
- [x] 10-25: KV wall two stages - no-crash at the wall (6a30f9157) +
  fit-time auto-size -fitm (db5cf4d80), both pushed, grow verified
  on-device (F31); SYCL=OFF = CPU-only runtime root-caused + fixed,
  build recipe permanently SYCL=ON (F30); B14 (twin-server + router)
  drafted - REJECTED 10-28 (final).
- [x] 11-01: B19 settle + B20 two-pass grow + tensor-mode draft fit
  committed (c9c4e20f1) + private sync with F40 addenda 1+2 and the
  B20 design doc (0ce10b479); -b/-ub probe matrix ran, batch-6144
  memory question answered (no code change needed); deploy pending
  the stop window (open item 1).
- [x] 11-01 (pm): co-resident CLI batteries v4+v5 (F40 addendum 3),
  all on the B19+fit binary with the 8080 service up: v4 4/4 exit 0
  (draft REAL-loaded + speculating at 16384; 9B built-in nextn
  UNSUPPORTED - H-A fold missing on the shared-weights MTP path);
  v5 4/4 exit 0 across the 10-31 failure regimes incl. the -b 6144
  -ub 6144 argset (no hang) and the 51823 262144 geometry (no
  DEVICE_LOST); no VRAM residue. 10-31 law not falsified in the
  layer world (binary + split caveats -> item 2b); T1 cold test
  retired.
- [x] 11-01 (evening): pre-B19 A/B (F40 addendum 3) + push
  c9c4e20f1..71df7aa3b: the deployed pre-B19 build survives the
  10-31 geometry (layer, 262144, 4 slots, MTP, co-resident) - no
  DEVICE_LOST, no hang; binary and static split variables ruled
  out, churn (release-lag) is the only untested 10-31 variable.
