# STATE - snapshot (refresh at each milestone)

Last refresh: 2026-10-27 (evening) - B15 COMPLETE, 3 commits
pushed: 9c8772ed8 (adaptive-ctx core + 3 E2E-found bug fixes),
5b8aa1fad (llama : regrow KV-row copy on device via
ggml_backend_tensor_copy between row-limited views, replaces the
host-buffer round trip up to ~6 GiB), 2822ce088 (server : skip the
idle KV shrink while retained slot KV does not fit the base floor). E2E on the FINAL binaries (gemma-4-E4B, port 18200): GROW content-
verified (ZQX-7741 survived 8192->32768 at 15054 tokens; 24054-
token prompt clean post-grow), FIRST EVER SUCCESSFUL IN-PLACE
SHRINK 16384->8192 in 68 ms, retained KV intact across the shrink
(BLT-2489), over-cap clean 400 EXCEED_CONTEXT_SIZE, server still
serving. Wide rebuild via build-sycl.ps1 green (247 steps, ~3
min, build\bin @ 2822ce088, reconfigure auto-picked the new HEAD).
Retained-KV shrink policy: C (never discard retained KV) shipped,
refined by the pre-check so a blocked shrink never reaches the
cell resize (F35). The 27B-scale shrink (262144->16284, ~6.3 GiB)
est. 0.5-1 s - its timing needs the B17 service-stop window. The
old --fit-ctx/-fitc, --fit-ctx-max/-fitm, --grow-ctx-max flags are
HARD DELETED: the live spawn line must be swapped at the next
install or the server will not start. NEW 10-27 (evening): F36
(B70 single-GPU references vs the live dual-card setup - the
41-46 t/s single-card decode floor), F37 (tensor-mode "bad
allocation" = host-side std::bad_alloc; proven host site = the 8
GiB default prompt cache; user reverted to -sm layer; --cache-ram
0 proposed), F38 (two-chat TG halving is structural: per-layer
blocking P2P waits + no slot pipelining -> B14 priority RAISED).
Carried: tensor-fit SHIPPED (1011b639d, F32); KV wall two stages
(F31); SYCL build fixed (F30); B12 DONE (7cb134632); F28 CLOSED
10-27; B14 PARKED (priority raised); B16 PARKED.
NEXT: B17 handover (install-dir swap + spawn line, incl. the
proposed --cache-ram 0 - user call) in a user stop window ->
measure the 27B-scale on-device copy timing in that window ->
B14 draft re-presentation -> B7a -> B13 recon. Await the user's
extraArgs bisection outcome + a free card for the Q5_K_S tensor
bench (N17).

## Environment

- Machine: Windows, AMD Ryzen 9 5950X 16C, 64 GB RAM.
- GPU: TWO physical Intel Arc Pro B70 32 GB GPUs (SYCL0 + SYCL1) +
  64 GB RAM. GGML_SYCL_DEVICE is a NO-OP in this fork (F9). Pin:
  `-dev SYCL0` or `--split-mode none -mg 0`. Tensor mode INTENTIONALLY
  uses both (`-dev SYCL0,SYCL1`, F19/N14).
- Build tree: `build/` (Ninja, Release, shared libs, tests OFF
  since the 10-27 script run). Configure ONLY via build-sycl.ps1:
  it pins C:\Program Files\CMake\bin\cmake.exe (4.4.4); mixing in
  the VS-bundled CMake 3.31.6 regenerates the build rules and
  forces a full rebuild (10-27 root cause of the repeated full
  builds). Recipe: NOTES.md N3 (oneAPI + vcvars64 Professional +
  oneAPI lib prepended to LIB).
- Install tree: `C:\Users\admin\.local\3nary-sycl\` (10-26 16:38
  install, PRE-ACTX - B15 binaries not in it). The LIVE CHILD RUNS
  FROM HERE (spawn line re-derived 10-27 evening, see Services).
  Swap = copy build\bin over this dir (service stop window:
  running exe/DLLs are locked) + swap the macube spawn line
  (hands-off) = B17.
  F30 note stands: the dlopen path is dead in this fork; SYCL works
  only via the static link in a SYCL=ON build.
- Workspace: working dir `C:\Users\admin\source\ai\llama.3nary`,
  workspace root `C:\Users\gorom\source\ai\llama.3nary` (SAME tree, one
  build, shared build\bin).
- Macube: `C:\Users\gorom\source\macube\` (DO NOT EDIT its config
  files). `TernaryModels.presets.ini` (model roots) = DO-NOT-EDIT.

## Services (macube)

- LIVE (do not disturb, user-owned): model Qwen3.8-27B-UD-Q6_K,
  child PID 22456 on 127.0.0.1:52439 (WS 13.36 GB), router PID
  15516 on 0.0.0.0:8080. The child spawns the INSTALL-dir exe
  (C:\Users\admin\.local\3nary-sycl\llama-server.exe). Spawn line
  re-derived 10-27 (evening) from macube/logs/llama-error.log
  (:14671): --ctx-size 163840 --fit on --fit-ctx 16284
  --fit-ctx-max 262144 --cache-type-k q5_1 --cache-type-v q5_1
  --flash-attn on --split-mode layer --batch-size 6144 --threads
  56 --threads-batch 28 --ubatch-size 6144 --n-gpu-layers all
  --numa distribute --jinja --rope-scaling yarn --load-mode none
  --reasoning-budget 4000 --reasoning-preserve
  --image-min/max-tokens 1024/4000 + api-key-file + mmproj BF16.
  NOTE 10-27 (evening): the user REVERTED this child to
  --split-mode layer after the tensor-mode bad-allocation episodes
  (F37; the day's tensor child was 19760 @ 56762, which had
  restarted 5x in ~4 h: 54418 -> 14088, -> 23940, -> 17416, ->
  17680, -> 19760). The old --fit-ctx flags work here only because
  the install dir is still PRE-ACTX; at the B17 swap the line must
  move to --adaptive-ctx (the old flags are gone from the new
  binary). F28's "CLONE tree" claim is STALE (install dir, not
  clone). The user is still bisecting the extraArgs tail (-ngl 99,
  -np 2, -c 262144); outcome pending (prediction table in F31).
  Re-derive PIDs: `Get-NetTCPConnection -LocalPort 8080 -State
  Listen` then the child via ParentProcessId (process details
  access-denied for non-service ports).
- History: 15516/22456 (10-27 evening router/child, CURRENT -
  layer mode after the user's tensor revert), 15516/19760 (10-27
  9:11 AM router/child, tensor, replaced by the user's revert),
  14096/17416 (10-27 3:37 AM, replaced), 17680 (10-27 child,
  replaced), 13988/23940 (10-27 00:05 child, replaced),
  54418->14088 (10-27
  child pair, bad-allocation episode, replaced), 13688/22072 (10-26
  1:10 PM service, replaced), 50630 (10-23 child) + 59148 (10-23
  router) + 62743 (dead
  10-23, F28, replaced by the 10-25 restart), 24252/26876 (10-22
  service, Q5_K_S + MTP, replaced), 52416
  (tg 13.2+ at 73.6K, healthy, then restarted), 54447 (dead, F16),
  65463 (dead, F15), 61850/52510 (stopped 10-19), 58458 / 58186
  (tracked dead, F6/F7).

## Active work

- Tensor-mode ctx fit (10-25, SHIPPED 1011b639d, F32):
  common_params_fit_tensor_impl (fit.cpp :179-434) - real model load
  + two real context creations (n_ctx 2048/32768), per-card free-mem
  deltas, linear solve for fixed + bytes/token per card, n_ctx to the
  min per-card budget floor-aligned to lcm(16384, n_streams) (trained
  cap kept exact when it is the limiting bound); -c 0 sentinel,
  explicit -c verified (fail prints the max fitting), -fitm grow
  branch; extra (draft/MTP) model throws. Same commit: the three meta
  accessors de-static (ggml-backend-meta.cpp + ggml-backend.h:407-409).
  Verified: 5-case matrix on 27B PTQ1_0 (A/A2/B/C/D, all as expected,
  F32). Findings: KV does not scale with n_seq_max (shared ring),
  hadamard-v2 gate blocks 10/20 exports, flash_attn auto-enabled.
- B15 (10-25 DESIGN CONFIRMED, 10-26 stages 1+2 DONE - F33/F34):
  runtime context growth = in-place tail extension of the shared
  ctx ring via per-layer regrow (the old per-slot recreate
  design was based on a WRONG premise: there is ONE shared ctx per
  server, slots are seq_ids, all slots pause briefly during a
  regrow). Stage 1 (core): llama_memory_i::resize virtual (default
  false); llama_kv_cells keep-existing resize; llama_kv_cache ctor
  restructure to per-(layer, buft) ggml ctx + buffer + resize()
  (per-layer alloc -> D2D copy rows [0, min(S, S')) -> free old ->
  swap k/v + rebuild stream views + re-point shared-layer entries;
  false if other != nullptr); llama_memory_hybrid::resize (mem_attn
  only); ctx-level public API (compute new n_ctx_seq mirroring the
  :376-392 derivation, memory->resize, update cparams). GDN is
  n_ctx-independent (rs_size = n_seq_max) - no grow; MTP draft ctx
  grown too (standalone plain KV, same size); prompt cache stays
  valid (host buffers, position-keyed) - no invalidation. Stage 2
  (server): the user-designed 4-flag adaptive-ctx family
  (-actx/--adaptive-ctx none|half|full default none,
  --adaptive-ctx-min default -1, --adaptive-ctx-max default -1,
  --adaptive-ctx-idle default 30; env
  LLAMA_ARG_ADAPTIVE_CTX[_MIN|_MAX|_IDLE]); triggers at
  task start (grow-before-reject) + decode wall (grow-then-
  DECODE_RETRY) + context-exceed sites; step-based target capped
  at the max; grow ctx_dft too, rollback both on draft failure;
  update n_ctx_slot + all slot.n_ctx; the idle hook shrinks to the
  load-time base after a sustained idle (t_idle_start_ms gate;
  idle 0 = grow-only; single jump to the floor); per-resize
  duration log lines ("grown/shrunk in place in X ms"). Stages
  1+2 BUILT GREEN 10-26/27 (F34):
  cells-first ordering bug in resize() found + fixed (unit test
  catches it, mutation-verified); unit tests
  tests/test-context-resize.cpp 100% green + deterministic (ctest,
  real stories15M GGUF, C API only - synthetic tests excluded on
  WIN32+shared); thread safety of the growth path verified in-
  process (single loop thread, lock-free like other KV mutations,
  doc note in llama.h). Stage 3: verify matrix on 27B PTQ1_0
  AFTER the user's install (user: "I will proceed with
  installation"). See BACKLOG B15 + FINDINGS F33/F34.
- KV wall (10-25, DONE - both stages committed+pushed, F31):
  - Stage 1 no-crash (6a30f9157): only n_batch == 1 && ret == 1 (the
    KV wall) is special-cased in decode() (bool -> int, file-local
    enum decode_result; single caller = the batch loop ~:2770):
    try_clear_idle_slots() -> DECODE_RETRY; else the failing slot is
    slot.id == batch_view.seq_id[0][0], gets send_error
    EXCEED_CONTEXT_SIZE + release() + prompt_clear() -> DECODE_SKIP.
    No preemption of active clients in v1.
  - Stage 2 auto-size (db5cf4d80): OPT-IN -fitm / --fit-ctx-max (env
    LLAMA_ARG_FIT_CTX_MAX, common_params.fit_params_max_ctx, 0 = off,
    units = total ctx like -c). Grow block in common_params_fit_impl
    (common/fit.cpp :310-360): measure the footprint ONCE at the cap,
    ctx_rate per device, budget = free - mb.total() - margin, grown =
    floor(min(limit, cap) to lcm(16384, n_streams)) - the 16K step;
    in-place dmds adjustment so the margin check applies to the grown
    ctx. No-op when the flag is absent; in-place KV ring growth
    REJECTED (pool fixed at context creation).
  - Verified on-device 10-25 (service holding the cards): host-only
    skips the grow; over-budget (-c 131072 -ngl 99 -fitm 262144)
    aborts correctly (projected 29501 > free 29008 MiB); GROW
    (-c 32768 -ngl 40 -fitm 65536) went 32768 -> 65536 with the margin
    check passing on both cards (S0 free 7132 / S1 6040 vs target
    1024 MiB).
- SYCL build (10-25, FIXED, F30): "no usable GPU" root-caused: the
  SYCL=OFF OOM-bypass builds were CPU-only runtimes (dlopen can never
  load SYCL with GGML_BACKEND_DL=OFF; the Release registry swallows
  the failure). One-time -DGGML_SYCL=ON reconfigure + ggml-sycl TU
  rebuild at -j 1 (105 steps, ~24 GB commit headroom) + downstream
  (297 steps), all green. build-3nary.bat permanently SYCL=ON with a
  comment; future builds incremental. Deploy only from SYCL=ON build
  states.
- B12: DONE 10-23 (7cb134632, 7 src files + unit test). H-A load-time
  fold of the 128 axis-split weights (48 ssm_out + 64 ffn_down + 16
  attn_output); ssm_out gather bug found + fixed (F27); PQ2_0 +
  PTQ1_0 tensor-verified; tests/test-hadamard-fold.cpp pins the
  formula. H-C (kernel fusion, zero requant noise) documented in F29
  as the fallback only.
- B14 (10-25, PARKED): twin-server + router - two pinned children
  (-dev SYCL0 / -dev SYCL1, -fit off + explicit -ngl) behind a router
  that aggregates /v1/models and routes by session. PARKED 10-25: the
  user chose the tensor-fit + growth path first (F32 shipped, F33
  drafted); B14 remains the alternative CAPACITY vehicle. See
  BACKLOG B14.
- B7a: startup-side loud-fail (unfittable ctx/KV) + closed --dev gate.
  Status: APPROVED 10-22, OPEN - next code candidate in this fork.
  The 10-25 runtime KV work is additive, NOT B7a. NOTE 10-25 (evening):
  the old target site (fit.cpp:182-183 warn-then-continue on
  SPLIT_MODE_TENSOR) is GONE - the tensor fit branch now exists
  (fit.cpp :179-434, F32) and throws a proper fit_exception. B7a's
  remaining target: the startup ctx/KV loud-fail for the non-tensor
  paths + the closed -sm/-dev parse gate (common-params parse area).
- B13: port the (fixed) ternary work to the upstream clone
  `C:\Users\admin\source\ai\llama.cpp` (F25). Status: UNBLOCKED (B12
  landed 10-23), recon NOT started. De-risked 10-23: the clone NOW
  HOSTS THE LIVE SERVICE in TENSOR mode (Q5_K_S), so it already has
  the tensor enum + meta backend. Clone is a LIVE deploy tree: recon
  is READ-ONLY. Open question: does the 3nary fork have the clone's
  seq_rm fix (server-context.cpp ~:3444) - relevant to F28. 3
  uncommitted user mods (chat.cpp, ui-assets.cmake, server-context.cpp)
  to PRESERVE.
- B7 (runtime preemption C0/C1): CLOSED 10-22 (user dropped).
- qwen35.cpp 9B MTP fix: GONE from the tree (working file matches
  HEAD; the local src/ commits since 48e9082ca do not touch it). Must
  be reported to the user.

## Git

- branch `main`, pushed HEAD: `2822ce088` (server : skip the idle
  KV shrink while retained slot KV does not fit). Before it:
  `5b8aa1fad` (llama : copy KV rows on device in the in-place
  cache regrow), `e141ac25b` (docs : private sync after B15 ship),
  `9c8772ed8` (llama : adaptive ctx, in-place runtime resize),
  `1011b639d` (fit : tensor-mode context fitting). Full list:
  NOTES.md N12.
- Working tree: `private/` modified (this refresh, uncommitted) +
  user's file (do not touch):
  `tools/ui/src/lib/stores/models/index.svelte.ts`.
- `private/` = this knowledge library; keep it committed+pushed.

## Binaries

- `build\bin`: 10-27 (evening) WIDE REBUILD via build-sycl.ps1 at
  HEAD 2822ce088 - the reconfigure auto-picked the new HEAD (cmake/
  build-info.cmake git rev-parse), 247 ninja steps, ~3 min, all
  targets relinked incl. llama-server.exe; build\bin in sync with
  the committed tree. This is the B17 swap source. The install dir
  (C:\Users\admin\.local\3nary-sycl\) is still PRE-ACTX - the swap
  is B17 (needs the user stop window).
- `build\bin`: 10-25 SYCL=ON rebuild (F30): ggml-sycl.dll 62.8 MB
  (~03:05), ggml.dll (~03:11), exes (~03:12) - the one-time ggml-sycl
  TU compile + 297 downstream steps. Supersedes the 10-23 B12 build.
  10-25 (evening, F32 build): llama-common.dll relinked 18:28 with
  the renamed tensor-fit helper; the exes + -impl dlls stay at 16:59
  (the build did not reach the exe relink - see the ninja quirk
  below). ABI-compatible in practice: the rename is a file-static
  function inside llama-common (fit.cpp), not in the -impl dlls, and
  the 5-case verify matrix ran green against exactly this binary set.
- NOTE 10-25 (evening): VS-bundled ninja 1.12.1
  (`C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\
  IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe`) exits 0
  after the [1/6] "Provisioning UI assets" custom target while
  `ninja -n` lists 6 pending edges (ui.cpp.obj, server-http.cpp.obj,
  llama-ui.lib, llama-server-impl.dll, llama-server.exe);
  llama-fit-params.exe never enters the plan. Root cause unresolved;
  not chased (F32 finding 6). Incremental compiles of common/ fit.cpp
  DO complete (obj + llama-common.dll relink).
- NOTE: the 10-21 install tree's ggml-sycl.dll lacks ggml_backend_init
  (same as the old build tree's) - verify the install tree's link
  state before any redeploy advice; dlopen is dead in this fork
  either way (F30).

## Models (verified on disk 10-22)

| Model | Path | Notes |
|---|---|---|
| Ternary PQ2_0 27B (crashing model) | `C:\DataVolume\TernaryModels\Ternary-Bonsai-2-27B-PQ2_0\` | PQ2_0 gguf 6.71 GB + mmproj-BF16 888 MB. 64 blocks (48 GDN + 16 full-attn at 3,7,...,63), NO MTP nextn, ternary type 142, qwen35 arch. ssm_out [K=6144, N=5120] PQ2_0. |
| Ternary PTQ1_0 27B (PREFERRED test model) | `E:\DataVolume\TernaryModels\Ternary-Bonsai-2-27B-PTQ1_0\` | PTQ1_0 gguf 5.54 GB + mmproj. B11 kernel applies. |
| Qwen3.8-27B-UD-Q5_K_S (target, live service) | `E:\DataVolume\Models\Qwen3.8-27B-UD-Q5_K_S\` | 17.38 GB + mtp-Q4_0 draft 1306 MiB + mmproj-BF16. 65 blocks, 16 full-attn + 48 GDN, embd 5120. Tensor mode proven at 160K ctx (31.9 t/s, N10). |
| Ternary family (small) | `E:\DataVolume\TernaryModels\` | 2B PTQ1_0, 4B PQ2_0 + 4B/9B MTP Q8_0 variants (co-residency candidates, F17). 10-25: the 10/20 small re-exports carry prism.hadamard.version=2 and are REJECTED by this tree (v1 gate, llama-model.cpp:1198-1201) - only the 9/20 27B files load (F32 finding 2). |

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
| fitm-test2.log | 10-25 host-only fit (nd == 0, grow skipped, fit OK). |
| fitm-test3.log | 10-25 GPU over-budget fit (-c 131072 -ngl 99 -fitm 262144): no grow, correct abort, exit 1. |
| fitm-grow.log / fitm-grow.bat | 10-25 GROW evidence: 32768 -> 65536, margin pass both cards, printed -c 65536 -ngl 40, exit 0. |
| run-fit-verify.ps1 | 10-25 tensor-fit matrix runner (27B PTQ1_0; cases A-auto, A2-auto-np2, B-explicit-big, C-full-ctx, D-grow). |
| fit-verify-{A-auto,A2-auto-np2,B-explicit-big,C-full-ctx,D-grow}.log | 10-25 tensor-fit verify matrix (F32): fitted -c lines at ~:2685, fit traces at ~:2681, per-card traces at ~:2310-2312. |
| fit-verify-probe-ptq10.log | 10-25 first successful 27B tensor fit (post service-restart; max fitting 913441). |
| build-3nary-fit.bat / build-3nary-fit{,2}.log | 10-25 fit build recipe + the ninja-1.12.1 early-exit evidence (F32 finding 6). |
| build-3nary.bat | N3 recipe, now permanently SYCL=ON (F30). |
| build-sycl-tu.bat / build-downstream.bat | one-time SYCL=ON ggml-sycl TU (-j 1, needs commit headroom) + downstream no-configure recipe (F30). |
| mtp-server-*.log, ternary-d-err.log, ptq10-* | historical (F1/F8/F10/F14). |

Macube logs: `C:\Users\gorom\source\macube\logs\llama.log` (per-child
srv-target lines, local clocks per run) + `llama-error.log` (router
lines + spawn args). The 10-22 service's blocks come after the 52416
block; grep by port 8080/58452.

## Next (10-27 evening)

1. B17 handover (TOP, needs the user stop window): stop the
   service -> copy build\bin over C:\Users\admin\.local\3nary-sycl\
   -> swap the macube spawn line (delete --fit-ctx/--fit-ctx-max;
   add --adaptive-ctx full --adaptive-ctx-min 16284
   --adaptive-ctx-max 262144 --adaptive-ctx-idle 30; PROPOSED
   --cache-ram 0, user call) -> start.
2. In that window: measure the 27B-scale on-device copy timing
   (262144->16284 load-shrink, ~6.3 GiB, est. 0.5-1 s) + verify
   the adaptive-ctx lines in the child log.
3. B14 draft re-presentation (priority RAISED by F37/F38 - the
   structural fix for BOTH reported issues); implement only on
   explicit user confirmation.
4. [B] Await the user's extraArgs bisection outcome (-ngl 99 / -np 2 /
   -c 262144; prediction table in F31) - fold into FINDINGS when
   reported.
5. [B] Await a free card (re-derive VRAM first) -> run the Q5_K_S
   tensor bench (N17; GDN correctness check first).
6. B7a (this fork, next code item): startup loud-fail on unfittable
   ctx/KV + closed -sm/-dev gate (the old tensor warn-then-continue
   target is GONE - F32; remaining: the non-tensor startup ctx/KV
   loud-fail + the parse gate).
7. B13 recon (read-only, clone hosts live service): diff fork vs
   clone meta backend (upstream PRs #26490/#27574), ternary enum
   mapping (fork non-standard PQ2_0=142/PTQ1_0=143), missing/
   divergent work map (F25), the seq_rm fix question (server-
   context.cpp ~:3444, relevant to F28). THEN port commit-by-
   commit on user-confirmed scope.
8. Refresh private/ + commit+push at each milestone.
