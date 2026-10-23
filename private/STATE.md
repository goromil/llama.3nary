# STATE - snapshot (refresh at each milestone)

Last refresh: 2026-10-23 (evening). B12 DONE + COMMITTED (7cb134632):
H-A load-time fold implemented in this fork (gather bug found + fixed,
F27), extended to ffn_down + attn_output (128 folds total), both
PQ2_0 and PTQ1_0 verified in tensor mode, formula pinned by
tests/test-hadamard-fold.cpp. USER 10-23: temp 1.0 is the production
sampling and never loops (the temp-0.8 loop seen in verify runs is a
sampling artifact, not a fold error); greedy-isolation test DROPPED;
unit test coverage APPROVED (done). F26 resolved: exactly 3
weight families (ssm_out/ffn_down/attn_output) get axis-split
activations; padding rejected (hidden-dim / attention-geometry
rewrite; does not fix ssm_out perm); fold = only viable option.
F28 OPEN: live service now runs from the CLONE tree and hit a
state-save bad allocation at 88K tok. NEXT: B7a (startup loud-fail),
then B13 port to the clone.

## Environment

- Machine: Windows, AMD Ryzen 9 5950X 16C, 64 GB RAM.
- GPU: TWO physical Intel Arc Pro B70 32 GB GPUs (SYCL0 + SYCL1) +
  64 GB RAM. GGML_SYCL_DEVICE is a NO-OP in this fork (F9). Pin:
  `-dev SYCL0` or `--split-mode none -mg 0`. Tensor mode INTENTIONALLY
  uses both (`-dev SYCL0,SYCL1`, F19/N14).
- Build tree: `build/` (Ninja, Release, shared libs, tests ON). Build
  recipe: NOTES.md N3 (oneAPI + vcvars64 Professional + oneAPI lib
  prepended to LIB).
- Install tree (macube services run from here):
  `C:\Users\admin\.local\3nary-sycl\` (state unknown after the 10-22
  service restart; re-check before advising a redeploy).
- Workspace: working dir `C:\Users\admin\source\ai\llama.3nary`,
  workspace root `C:\Users\gorom\source\ai\llama.3nary` (SAME tree, one
  build, shared build\bin).
- Macube: `C:\Users\gorom\source\macube\` (DO NOT EDIT its config
  files). `TernaryModels.presets.ini` (model roots) = DO-NOT-EDIT.

## Services (macube)

- LIVE (10-23, do not disturb, user-owned): REDEPLOYED FROM THE CLONE
  TREE `C:\Users\admin\source\ai\llama.cpp` (F28 - compile-time paths
  in its logs point there). Old 10-22 PIDs 24252/26876 STALE.
  Observed 10-23: 62743 (child, died slot 3 / 88212 tok, F28), 50630
  (child, recovered, serving Q5_K_S tensor), 59148 (router, LCP
  lines). Re-derive PIDs: `Get-NetTCPConnection -LocalPort 8080,
  58452 -State Listen`.
- History: 24252/26876 (10-22 service, Q5_K_S + MTP, replaced), 52416
  (tg 13.2+ at 73.6K, healthy, then restarted), 54447 (dead, F16),
  65463 (dead, F15), 61850/52510 (stopped 10-19), 58458 / 58186
  (tracked dead, F6/F7).

## Active work

- B12: make ternary PTQ1_0/PQ2_0 `-sm tensor` compatible in this
  fork. DONE 10-23, committed 7cb134632 (7 src files + unit test).
  Implementation: `fold_hadamard_weight` (llama-model.cpp) - load-time
  dequant -> in-place block FWHT -> signs -> ssm_out-only scatter
  (grouped->tiled) -> requant, 128 weights (48 ssm_out + 64 ffn_down +
  16 attn_output). Gather bug found + fixed (F27, proven exact by
  numeric mini-model). Verified: PQ2_0 + PTQ1_0 tensor runs, 128 fold
  lines, no :537 assert, SPLIT_STATE ssm_out {1024x3,1024x3} /
  ffn_down {8704x1,8704x1}, sane tokens. Unit test
  tests/test-hadamard-fold.cpp pins the formula (all checks pass).
  Kernel-level fusion fallback (H-C, zero requant noise) documented in
  F29 - adopt only if the fold's requant noise shows up in real
  workloads.
- B7a: startup-side loud-fail (unfittable ctx/KV) + closed --dev gate.
  Status: APPROVED 10-22, OPEN - NEXT in this fork. Target site:
  common/fit.cpp:182-183 (SPLIT_MODE_TENSOR fit throw, currently
  warn-then-continue per log).
- B13: port the (fixed) ternary work to the upstream clone
  `C:\Users\admin\source\ai\llama.cpp` (F25). Status: recon NOT started.
  De-risked 10-23: the clone NOW HOSTS THE LIVE SERVICE in TENSOR mode
  (Q5_K_S), so it already has the tensor enum + meta backend.
  Clone is a LIVE deploy tree: recon is READ-ONLY. Open question:
  does the 3nary fork have the clone's seq_rm fix (server-context.cpp
  ~:3444) - relevant to F28. 3 uncommitted user mods (chat.cpp,
  ui-assets.cmake, server-context.cpp) to PRESERVE.
- B7 (runtime preemption C0/C1): CLOSED 10-22 (user dropped).

## Git

- branch `main`, pushed HEAD: `7cb134632` (llama : fold
  activation-side hadamard into ternary weights for tensor mode).
  Full list: NOTES.md N12.
- Working tree: `private/` modified (this refresh, uncommitted) +
  user's file (do not touch):
  `tools/ui/src/lib/stores/models/index.svelte.ts`.
- `private/` = this knowledge library; keep it committed+pushed.

## Binaries

- `build\bin`: current as of the 10-23 B12 build (H-A fold, 10:58)
  + test-hadamard-fold (15:03). Supersedes the 10-20 B11 build.

## Models (verified on disk 10-22)

| Model | Path | Notes |
|---|---|---|
| Ternary PQ2_0 27B (crashing model) | `C:\DataVolume\TernaryModels\Ternary-Bonsai-2-27B-PQ2_0\` | PQ2_0 gguf 6.71 GB + mmproj-BF16 888 MB. 64 blocks (48 GDN + 16 full-attn at 3,7,...,63), NO MTP nextn, ternary type 142, qwen35 arch. ssm_out [K=6144, N=5120] PQ2_0. |
| Ternary PTQ1_0 27B (PREFERRED test model) | `E:\DataVolume\TernaryModels\Ternary-Bonsai-2-27B-PTQ1_0\` | PTQ1_0 gguf 5.54 GB + mmproj. B11 kernel applies. |
| Qwen3.8-27B-UD-Q5_K_S (target, live service) | `E:\DataVolume\Models\Qwen3.8-27B-UD-Q5_K_S\` | 17.38 GB + mtp-Q4_0 draft 1306 MiB + mmproj-BF16. 65 blocks, 16 full-attn + 48 GDN, embd 5120. Tensor mode proven at 160K ctx (31.9 t/s, N10). |
| Ternary family (small) | `E:\DataVolume\TernaryModels\` | 2B PTQ1_0, 4B PQ2_0 + 4B/9B MTP Q8_0 variants (co-residency candidates, F17). |

KV math (F17): KV q5_1 @131072 = 3072 MiB, draft KV q4_0 = 144 MiB,
GDN recurrent state 598.50 MiB (no per-token growth).

## Logs (ours, in $env:TEMP\opencode\)

| File | Content |
|---|---|
| verify_pq2_0_b12fix.log | 10-23 PQ2_0 16-tok tensor PASS (128 folds, no assert). |
| verify_pq2_0_128tok.log | 10-23 PQ2_0 128-tok tensor; perfect reasoning, answer "parang" (suspected sampling artifact, temp 0.8). |
| verify_ptq1_0_tensor.log | 10-23 PTQ1_0 128-tok tensor; Paris found in thinking, then temp-0.8 repeat loop (sampling artifact per user: production runs temp 1.0, no loops). |
| verify_ptq1_0_layer.log | 10-23 PTQ1_0 128-tok `-sm layer` baseline; clean Paris. |
| run_verify_direct.ps1 | direct-redirect verify runner (`$env:SM` toggles tensor/layer). |
| llama_completion_help.txt | llama-completion --help dump (it is local, full sampling args). |
| fold_order_test.py | numeric mini-model proving the ssm_out gather-bug fix (5.58 -> 0.000000). |
| pq2_meta_debug.txt | 4379-line GGML_META_DEBUG tensor-mode trace of the PQ2_0 crash; assert at :4378. |
| run_pq2_debug.ps1 | setvars + N14 repro runner. |
| fitprobeA/B/C.txt | 10-20 fit probes (12 MB each, grep only). |
| mtp-server-*.log, ternary-d-err.log, ptq10-* | historical (F1/F8/F10/F14). |

Macube logs: `C:\Users\gorom\source\macube\logs\llama.log` (per-child
srv-target lines, local clocks per run) + `llama-error.log` (router
lines + spawn args). The 10-22 service's blocks come after the 52416
block; grep by port 8080/58452.

## Next (10-23)

1. [DONE] B12 implemented + verified + committed (7cb134632) +
   unit test. Pushed to origin.
2. [NEXT] B7a (this fork): startup loud-fail on unfittable ctx/KV +
   closed --dev gate. Target: common/fit.cpp:182-183 SPLIT_MODE_TENSOR
   fit throw + arg.cpp:2804 -sm parse area.
3. B13 recon (read-only, clone hosts live service): diff fork vs
   clone meta backend (upstream PRs #26490/#27574), ternary enum
   mapping (fork non-standard PQ2_0=142/PTQ1_0=143), missing/divergent
   work map (F25), whether 3nary lacks the clone's seq_rm fix
   (server-context.cpp ~:3444, relevant to F28). THEN port
   commit-by-commit on user-confirmed scope.
4. F28 (open, read-only): live-service state-save bad allocation at
   88K tok (clone build); does 3nary have the equivalent fix?
5. Refresh private/ + commit+push at each milestone.
