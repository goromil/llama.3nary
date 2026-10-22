# STATE - snapshot (refresh at each milestone)

Last refresh: 2026-10-22. PIVOT to tensor-mode ternary compatibility
(user direction): root cause of the PQ2_0 `-sm tensor` assert FULLY
pinned (F20), meta execution model mapped (F21), activation-side fix
H-B proven dead (F22), lead fix H-A designed (F23) - implementation
AWAITING USER CONFIRMATION. Port target located: the upstream clone at
C:\Users\admin\source\ai\llama.cpp (F25, B13). Twin-server data
parallel DROPPED by user; B7 preemption DEMOTED (confirm before
touching).

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

- LIVE (10-22, do not disturb, user-owned): 24252 = llama-server
  router 0.0.0.0:8080; 26876 = llama-server child 127.0.0.1:58452
  (Q5_K_S + MTP). Re-verify PIDs after any reboot:
  `Get-NetTCPConnection -LocalPort 8080,58452 -State Listen`.
- History: 52416 (tg 13.2+ at 73.6K, healthy, then restarted), 54447
  (dead, F16), 65463 (dead, F15), 61850/52510 (stopped 10-19), 58458 /
  58186 (tracked dead, F6/F7).

## Active work

- B12 (LEAD): make ternary PTQ1_0/PQ2_0 `-sm tensor` compatible in this
  fork. Status: investigation COMPLETE, design READY, BLOCKED on user
  confirming the H-A design (F23) before code. Crash repro: N14
  (deterministic, warmup, block 0; trace in
  $env:TEMP\opencode\pq2_meta_debug.txt, assert at :4378).
- B13: port the (fixed) ternary work to the upstream clone
  `C:\Users\admin\source\ai\llama.cpp` (F25). Status: recon NOT started
  (delta unmeasured; upstream meta backend diverged from 3nary's).
  BLOCKED on B12 landing first.
- B7: DEMOTED 10-22 (twin-server design dropped; C0/C1 "may be" -
  explicit user confirmation required before implementing).

## Git

- branch `main`, pushed HEAD: `48e9082ca` (fit-params: include
  speculative draft model in the fit). Full list: NOTES.md N12.
- Working tree: `private/` modified (this refresh, uncommitted) +
  user's file (do not touch):
  `tools/ui/src/lib/stores/models/index.svelte.ts`.
- `private/` = this knowledge library; keep it committed+pushed.

## Binaries

- `build\bin`: current as of the 10-20 B11 build + 48e9082ca fit tool
  (byte-parallel PTQ1_0 kernel, MTP 3-part fix, KV guard, PTQ1_0
  prefill). A B12 tensor-mode build supersedes this.

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
| pq2_meta_debug.txt | 4379-line GGML_META_DEBUG tensor-mode trace of the PQ2_0 crash; assert at :4378. |
| run_pq2_debug.ps1 | setvars + N14 repro runner. |
| fitprobeA/B/C.txt | 10-20 fit probes (12 MB each, grep only). |
| mtp-server-*.log, ternary-d-err.log, ptq10-* | historical (F1/F8/F10/F14). |

Macube logs: `C:\Users\gorom\source\macube\logs\llama.log` (per-child
srv-target lines, local clocks per run) + `llama-error.log` (router
lines + spawn args). The 10-22 service's blocks come after the 52416
block; grep by port 8080/58452.

## Next (10-22)

1. PRESENT to user: corrected root-cause report (F20: no GDN
   corruption; crash = unrepresentable perm reshape, 1.5 rep-slab at any
   even split) + H-B death proof (F22) + H-A design (F23: load-time
   W'' fold, ~15 GFLOP, one extra requant pass ~sqrt(2)x error, size
   unchanged) vs the D1 mirror fallback. Get explicit OK (new pattern /
   large change).
2. ON CONFIRMATION: implement H-A per F23/B12 (loader callback hook,
   model-side transform, `folded` flag + build_lora_mm gate); rebuild
   (N3); verify with N14 (assert gone, SPLIT_STATE ssm_out chain =
   {axis 0, 2048x3}); test PQ2_0 then PTQ1_0 in tensor mode; refresh
   private/ + commit+push.
3. B13 recon (only after B12 lands): diff upstream vs 3nary meta
   backend, upstream ternary type enum, missing-work map (F25).
4. Optional: user to confirm B7 C0/C1 keep/drop; clone "true fork"
   (GitHub) step only if the user asks.
