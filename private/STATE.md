# STATE - snapshot (refresh at each milestone)

Last refresh: 2026-10-28 (afternoon) - F39 measurement DONE +
the -np 2 anomaly root-caused (full analysis in FINDINGS F39,
[measured]/[certain]/[decision] blocks): on the DEFAULT spawn
config (no --parallel -> auto 4 slots + unified KV,
server.cpp:152-157) two concurrent chats DID batch - separate
slots, one decode batch of 2 seqs, per-stream tg_3s 10.7-10.8,
SUM ~21.4 t/s (~76% of the 2/H roofline vs solo ~14 at the
comparable ~35K ctx); the overlapped chats were 27.6K + 7K, NOT
pinning both cards at max ctx, so the pre-registered pipeline
gate is not met - pipeline stays unbuilt, B14 remains the
capacity vehicle. The -np 2 child (57327) root cause: explicit
--parallel skips the auto branch, kv_unified silently keeps its
default false (common.h:575) -> per-slot KV (total 2x, ~14.4
GiB) and the adaptive-ctx TOTAL cap doubles to 524288 BY DESIGN
(fit.cpp:528 = n_ctx_train x n_streams; the no_alloc fit
granted it); the child itself was HEALTHY (loaded 1.09 min,
served, zero SYCL alloc errors/warnings in its whole log
section); the user's GPU2 full->0 + CPU +30 GB read is
UNPROVEN (hypothesis: model + per-slot KV host residency ~=
32 GB; no silent code path exists - every fallback logs); the
user then restarted the whole service. Recommendation: drop
-np 2, keep the auto default. Live state changed by that
restart: NEW router on 0.0.0.0:8080 (PID not recorded), current
child 49822 on 127.0.0.1:49822 with the ORIGINAL config
(4 auto slots, unified KV, n_ctx_slot 262144); the user also
INSTALLED the latest build\bin (e78fe1b51, the timing-gate
binaries) into the install dir - the gated tg print behavior is
confirmed live in llama.log. F40 write-up still pending user
confirmation (analysis state below).

Previous refresh: 2026-10-28 (16:12) - BUILD RECOVERY GREEN after
the user's VS-2022-shell botched configure. Root cause
(confirmed by an isolated child-shell test): the user's window
was already a VS 2022 dev environment (Developer PowerShell
for VS 2022, or a terminal opened from VS 2022); with the VS
2022 env vars (VCINSTALLDIR/VSINSTALLDIR/PATH) preset, the
script's Launch-VsDevShell (VS 2026) import is a complete NO-OP
and still prints "imported successfully" - `where cl` stays at
VS 2022 14.44.35207. CMake's bare cl then resolved to VS 2022
!= cached VS 2026 cl -> the known CMake 4.4.4 cache wipe that
DROPS the -D args (a Debug + cl-as-CXX reconfigure started).
Ctrl+C mid-configure left build\CMakeCache.txt DELETED (never
rewritten), the prompt stuck in build\ (Push-Location not
popped), and the `&&` chain dead, so `-Portable -InstallOnly`
never ran. No source touched; objects + the old lock intact.
build-sycl.ps1 HARDENED (local, gitignored, NOT committed):
(1) after the VS import the script now FORCES the intended
VS's cl dir onto PATH (newest VC\Tools\MSVC\*\bin\Hostx64\x64
under the detected VS root, prepended; prints "Forced cl
directory: ..."), so a no-op import can no longer matter;
(2) the resolved cl full path is PINNED in .build-info-sycl.json
- a session pointing at a different VS now fails with a clean
lock mismatch BEFORE CMake (fix: -Clean) instead of triggering
the destructive wipe; locks written before the `cl` key existed
are tolerated. Recovery rerun from a clean shell: lock check
passed, fresh configure with the correct Release/VS 2026
cl/icx args (cache verified: C = cl 14.51.36231, CXX = icx);
the interrupted reconfigure had invalidated ninja's restat
state, so the fast TUs fully recompiled (247 steps) while the
slow SYCL TU obj survived (ggml-sycl.dll untouched, still the
03:24 VS 2026 one). build\bin back at e78fe1b51 code state
(HEAD is dff5d5cfe, docs-only). Rule for the user: run
build-sycl.ps1 from a PLAIN PowerShell window (start menu),
never a Developer PowerShell or a VS terminal - the script now
self-defends, but a plain shell is the zero-surprise path.
The throwaway proof script
(C:\Users\admin\AppData\Local\Temp\opencode\test-vsdevshell.ps1)
is deletable.

Previous refresh: 2026-10-28 (late) - timing-log gate committed +
built (e78fe1b51): the periodic tg line printed every 3 s
regardless of change (~200-350 lines per long task; 75% of the
current live llama.log - 65347 of 86566 lines) - now printed
only when tg or tg_3s moved >= 10% vs the last printed value,
and the prompt-processing tps line got the same gate. One file
(tools/server/server-context.cpp): TIMINGS_PRINT_CHANGE = 0.10
constant + per-slot last-printed fields (zeroed in reset()); the
3 s minimum interval and the final per-task summary are
unchanged. Incremental build GREEN (249 steps, same VS 2026
toolchain, build\bin @ e78fe1b51): only server-context.cpp.obj
recompiled (step 243) + llama-server-impl.dll / llama-server.exe
relinked (245-246); the rest is the build-info.cpp timestamp
cascade (harmless relink of the downstream targets). NOT deployed
AT THAT TIME (SUPERSEDED 10-28 afternoon: the user installed
these e78fe1b51 binaries into the install dir - the timing-gate
llama-server.exe is LIVE; see Services). If the 10% trigger
feels wrong after live use: one constant to edit, ~5 min
incremental rebuild. Also recorded:
first live 27B-scale shrink 262144 -> 163840 in 129 ms
(llama.log :168535-168537, 10-28 ~02:34, 30 s idle after the
73319-token conversation) - far under the 0.5-1 s estimate;
regrow still pending (needs a > 163840-token conversation,
largest so far 85807). F40 (bad-alloc) investigation ~80%, NOT
yet written into FINDINGS: root cause = machine commit
exhaustion (25.2 GB free commit now vs ~23 GB on 10-27); the
prompt-cache site (server_prompt_cache::alloc) CATCHES bad_alloc
and self-throttles 0.4x, so the decode-path bad_alloc is a host
new surfacing at an arbitrary site; every bad_alloc port was a
TENSOR-mode child (up to 62 lines on one port), all four layer
ports (52439/65299/56762/52420) 0 lines; the [NNNNN] child
prefix is the PORT, not a PID (F37/STATE PIDs to be corrected);
fit.cpp:219 throws on an MTP draft with SPLIT_MODE_TENSOR, and
the 10-27 tensor child with --fit-ctx 16284 still started at
n_ctx_slot = 262144.

Older refresh: 2026-10-28 - B17 HANDOVER DONE (user-executed
~02:32): install dir now ACTX (the child starts on --adaptive-ctx, which
the PRE-ACTX binary rejects), spawn line swapped to --adaptive-ctx
full, model moved Q5_K_S -> Qwen3.8-27B-UD-Q6_K. New live: router
PID 27804 @ 0.0.0.0:8080, child PID 2296 @ 127.0.0.1:65299 (WS
13.7 GB). Adaptive-ctx resolved: mode=full base=163840 max=262144
idle=30s; the full-mode load-fit settled at the MAX, so
n_ctx_slot = 262144. The idle shrink goes to BASE (163840), not
the 4096 auto-min (try_shrink_ctx_idle), so the 27B idle cycle is
262144 <-> 163840 = 98304 rows x ~27.5 KB/token = ~2.65 GiB
on-device copy per direction (supersedes the 262144->16284 ~6.3
GiB estimate). First post-handover decode: 91559-token
conversation tg 12.53 t/s single-stream, prompt eval 186.29 t/s -
the single-stream baseline for the F39 measurement. The user
omitted --adaptive-ctx-min/-max/-idle (auto resolution) and did
NOT add --cache-ram 0 (open call; the F37 bad-alloc risk zone is
LIVE: full 262144 ctx + 8 GiB default host prompt cache + 91k-
token conversations). Build recipe moved to VS 2026: the first
attempt failed because changing CMAKE_C_COMPILER on the existing
Ninja cache triggers a cache-wipe re-configure that DROPS the
original -D args (Debug + cl-as-CXX; the C2059 at
server-context.cpp:3848 was a cl artifact, not a source bug); the
-Clean rerun was GREEN (553 steps, C = cl 14.51.36231, CXX = icx
2026.1.1, Release, build\bin @ 2b10cfb42). Rule: after ANY
compiler change, rebuild with -Clean. CMake stays standalone
4.4.4 (VS 2026 bundles 4.3.1, fallback only). build-sycl.ps1 is
GITIGNORED by design (.gitignore:46 /build*) - machine-local, so
the recipe change is recorded here, not in a commit. F39
recorded: two-chat pipeline assessment - steady-state aggregate
EXACTLY ties the B14-DP aggregate (2/H), per-stream TG tied
(1/H), the "copy limited" premise FALSE for layer mode (only the
20 KB residual stream crosses per token, async in the meta
backend; the blocking q8_1 weight copies are tensor/row-mode
only); real win = ~2x KV headroom (~40 GB vs ~20 GB DP); real
cost = a third execution model upstream would not merge;
recommend MEASURE 2-slot TG vs the 12.53 t/s baseline first.
Carried: B15 COMPLETE (3 commits pushed, last 2822ce088; E2E on
the final binaries: first in-place shrink 16384->8192 in 68 ms,
grow content-verified, retained KV intact, over-cap clean); B14
still PARKED (priority raised, F37/F38); B16 PARKED.
NEXT: F40 write-up + 129 ms shrink record + port-not-PID
corrections into FINDINGS -> --cache-ram 0 user call (the F40
recommendation feeds it; the F37 risk zone is LIVE; the
timing-gate binary waits for the same stop window) -> F39
measurement (2-slot TG vs the 12.53 t/s baseline; recipe: two
concurrent chats, per-slot tg read from llama.log) -> B17
regrow timing (needs a > 163840-token conversation) -> B14
draft re-presentation -> extraArgs bisection outcome -> Q5_K_S
tensor bench (N17) when a card is free -> B7a -> B13 recon.

## Environment

- Machine: Windows, AMD Ryzen 9 5950X 16C, 64 GB RAM.
- GPU: TWO physical Intel Arc Pro B70 32 GB GPUs (SYCL0 + SYCL1) +
  64 GB RAM. GGML_SYCL_DEVICE is a NO-OP in this fork (F9). Pin:
  `-dev SYCL0` or `--split-mode none -mg 0`. Tensor mode INTENTIONALLY
  uses both (`-dev SYCL0,SYCL1`, F19/N14).
- Build tree: `build/` (Ninja, Release, shared libs, tests OFF).
  Configure ONLY via build-sycl.ps1 (GITIGNORED by design,
  .gitignore:46 `/build*` - a machine-local recipe; changes are
  recorded in this file, never committed). VS 2026 preferred (18.x
  first, 2022 fallback): C = MSVC 14.51.36231 (cl, VS 18),
  CXX = IntelLLVM icx 2026.1.1 (oneAPI 2026.1.1), Ninja 1.13.2
  from VS 18. CMake stays the standalone
  C:\Program Files\CMake\bin\cmake.exe (4.4.4) - VS 2026 bundles
  4.3.1 (fallback only); mixing CMake versions regenerates the
  build rules and forces a full rebuild (10-27 root cause of the
  repeated full builds). GOTCHA 10-28: changing CMAKE_C_COMPILER
  on the existing Ninja cache triggers a cache-wipe re-configure
  that DROPS the original -D args (observed: Debug + cl-as-CXX
  for both languages, then a C2059 at server-context.cpp:3848 -
  a cl artifact, the tree compiles clean under icx); after ANY
  compiler change, rebuild with -Clean. GOTCHA 10-28 (second
  trigger mode): a calling shell that is ALREADY a VS dev
  environment (Developer PowerShell for VS 2022, or a terminal
  opened from VS 2022) makes the script's VS 2026
  Launch-VsDevShell import a no-op (verified in an isolated
  test; it still prints success), so bare cl resolves to the
  session's VS -> same destructive wipe. HARDENED 10-28: the
  script now forces the intended VS's cl dir onto PATH after
  the import (prints "Forced cl directory:") and pins the
  resolved cl into .build-info-sycl.json, so a stale session's
  cl fails with a clean lock mismatch pre-CMake instead of a
  wipe. Still: run from a PLAIN PowerShell window. After a
  wipe, a rerun from a clean shell = fresh configure + full
  recompile of the fast TUs (the SYCL TU obj survives).
  Recipe: NOTES.md N3.
- Install tree: `C:\Users\admin\.local\3nary-sycl\` - the live
  child spawns from here. NOW holds the e78fe1b51 (timing-gate)
  binaries: the user copied build\bin over it on 10-28
  (afternoon, after the 16:12 recovery rebuild) - the gated tg
  print behavior is confirmed live in llama.log. Before: the
  ACTX binaries (2822ce088 tree) since the 10-28 ~02:32 B17
  swap. Swap procedure (for the next one): copy build\bin over
  this dir at a service stop window (running exe/DLLs are
  locked). The install dir is IN SYNC with build\bin.
  F30 note stands: the dlopen path is dead in this fork; SYCL works
  only via the static link in a SYCL=ON build.
- Workspace: working dir `C:\Users\admin\source\ai\llama.3nary`,
  workspace root `C:\Users\gorom\source\ai\llama.3nary` (SAME tree, one
  build, shared build\bin).
- Macube: `C:\Users\gorom\source\macube\` (DO NOT EDIT its config
  files). `TernaryModels.presets.ini` (model roots) = DO-NOT-EDIT.

## Services (macube)

- LIVE (do not disturb, user-owned): model Qwen3.8-27B-UD-Q6_K,
  CURRENT child 49822 on 127.0.0.1:49822 (layer mode, 4 auto
  slots, unified KV, n_ctx_slot 262144), NEW router on
  0.0.0.0:8080 (PID not recorded) - the user restarted the whole
  service after the F39 -np 2 test (old router died ~3.32 min,
  llama-error.log:16752; 49822 spawned at 10.27 min with the
  ORIGINAL config, :16786+). Child rotation since the handover:
  2296 -> 53777 -> 51396 (F39 test 1, default config) -> 57327
  (F39 test 2, --parallel 2) -> 49822. The child spawns the
  INSTALL-dir exe (C:\Users\admin\.local\3nary-sycl\llama-
  server.exe) - now the e78fe1b51 (timing-gate) binaries,
  user-installed 10-28 (afternoon). Spawn line VERBATIM
  (llama-error.log :15408-15473; re-confirmed WITHOUT --parallel
  for 51396 at :16565-16628 and for 49822 at :16786+):
  --adaptive-ctx full --api-key-
  file C:\Users\admin\source\macube\config\local\llama-api-keys.
  txt --alias Qwen3.8-27B-UD-Q6_K --batch-size 6144 --cache-type-
  k q5_1 --cache-type-v q5_1 --ctx-size 163840 --flash-attn on
  --fit on --host 127.0.0.1 --image-max-tokens 4000 --image-min-
  tokens 1024 --jinja --load-mode none --log-verbosity 3 --model
  E:\DataVolume\Models\Qwen3.8-27B-UD-Q6_K\Qwen3.8-27B-UD-Q6_K.
  gguf --mmproj E:\DataVolume\Models\Qwen3.8-27B-UD-Q6_K\mmproj-
  BF16.gguf --n-gpu-layers all --numa distribute --port 65299
  --presence-penalty 0.0 --reasoning-budget 4000
  --reasoning-preserve --rope-scaling yarn --split-mode layer
  --temperature 1.0 --threads 56 --threads-batch 28 --top-k 20
  --top-p 0.95 --ubatch-size 6144 --webui-mcp-proxy --min-p 0.0.
  CHANGES vs the 10-27 line: --fit-ctx 16284 --fit-ctx-max
  262144 DELETED, --adaptive-ctx full ADDED, model Q5_K_S ->
  Q6_K; the user OMITTED --adaptive-ctx-min/-max/-idle (auto) and
  did NOT add --cache-ram 0 (open call). Adaptive-ctx state at
  handover (child startup): `adaptive ctx: mode=full base=163840
  max=262144 idle=30s`, load-fit settled at the max ->
  n_ctx_slot = 262144. The idle shrink target is BASE (163840),
  not the 4096 auto-min (try_shrink_ctx_idle, server-context.cpp:
  1702-1770), so the 27B idle cycle = 262144 <-> 163840 = 98304
  rows x ~27.5 KB/token (q5_1 KV) ~ 2.65 GiB on-device copy per
  direction. First post-handover decode: 91559-token
  conversation tg 12.53 t/s (single-stream baseline for F39),
  prompt eval 186.29 t/s. F39 MEASURED 10-28 (afternoon) on this
  config: two concurrent chats batched (2 of the 4 auto slots,
  one 2-seq decode batch), per-stream tg_3s 10.7-10.8, SUM ~21.4
  t/s ~ 76% of the 2/H roofline vs solo ~14 at comparable ctx;
  the chats were 27.6K + 7K (NOT max-ctx-pinned) -> the
  pre-registered pipeline gate is not met (FINDINGS F39
  [measured]/[decision]). The extraArgs bisection: -np 2
  ANSWERED + root-caused (kv_unified silently false -> per-slot
  KV 2x + total adaptive-ctx cap 524288 by design; child healthy;
  the user's GPU2/CPU monitor read UNPROVEN - F39 [certain]);
  -ngl 99 / -c 262144 outcome still pending (prediction table in
  F31). The F37 risk zone is LIVE (full 262144 ctx + 8 GiB
  default host prompt cache + 82k-106k-token conversations); if
  the bad-alloc class recurs, add --cache-ram 0 to the spawn line
  at the next stop window (user-owned config). Future `adaptive
  ctx: KV cache shrunk/grown in place in X ms` timings land in
  macube/logs/llama.log - the first live shrink (129 ms,
  262144 -> 163840, in the previous refresh entry) + the first
  regrow still need to land in FINDINGS (closes B17; regrow
  needs a > 163840-token conversation, largest so far 106815).
  Re-derive PIDs: `Get-NetTCPConnection -LocalPort 8080 -State
  Listen` then the child via ParentProcessId (process details
  access-denied for non-service ports).
- History: (new router, PID not recorded)/49822 (10-28 afternoon,
  CURRENT - e78fe1b51 install, original config after the user's
  post-F39 restart), 27804/2296 (10-28 ~02:32 router/child -
  ACTX install, --adaptive-ctx full, Q6_K, the B17 handover;
  replaced by the user's post-F39 restart), 51396 (F39 test-1
  child, default config, 2-chat TG measured), 57327 (F39 test-2
  child, --parallel 2, healthy), 15516/22456 (10-27 evening
  router/child @ 52439, layer mode after the user's tensor
  revert; replaced 10-28 ~02:32 by the B17 handover), 15516/19760
  (10-27 9:11 AM router/child, tensor, replaced by the user's
  revert),
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

- branch `main`, pushed HEAD: `722ce512c` (docs : private sync -
  build recovery GREEN, VS-2022-shell root cause + build-
  sycl.ps1 hardening). Before it: `dff5d5cfe` (docs : private
  sync after the timing-log gate - build state, 129 ms shrink
  record, F40 status), `e78fe1b51` (server : gate periodic
  tg/tp log lines on a >= 10% rate change), `796c40945`
  (docs : private sync after B17 handover - F39 pipeline
  assessment, VS 2026 build recipe, live state),
  `2b10cfb42` (docs : private sync after B15 complete -
  F36/F37/F38, B17, handover ready), `2822ce088` (server : skip
  the idle KV shrink while retained slot KV does not fit),
  `5b8aa1fad` (llama : copy KV rows on device in the in-place
  cache regrow), `9c8772ed8` (llama : adaptive ctx, in-place
  runtime resize), `1011b639d` (fit : tensor-mode context
  fitting). The docs commit for THIS round (F39 outcome + -np 2
  root cause + the post-F39 service restart) is PENDING.
  Full list: NOTES.md N12.
- Working tree: `private/` modified (this refresh, uncommitted) +
  user's file (do not touch):
  `tools/ui/src/lib/stores/models/index.svelte.ts`.
- `private/` = this knowledge library; keep it committed+pushed.

## Binaries

- `build\bin`: 10-28 16:12 RECOVERY REBUILD GREEN via the
  hardened build-sycl.ps1 (post VS-2022-shell cache wipe; see
  header): fresh configure, full recompile of the fast TUs
  (247 steps), same VS 2026 toolchain (cl 14.51.36231 + icx
  2026.1.1, Release); ggml-sycl.dll untouched (the 03:24 wide
  rebuild obj survived). build\bin @ e78fe1b51 code state =
  the (late) entry below; DEPLOYED 10-28 (afternoon) by the
  user: the install dir now holds these e78fe1b51 binaries (the
  timing-gate llama-server.exe is LIVE; see Services).
- `build\bin`: 10-28 (late) INCREMENTAL build via build-sycl.ps1
  at HEAD e78fe1b51 (the timing-log gate), same VS 2026 toolchain,
  249 steps: only server-context.cpp.obj recompiled (step 243) +
  llama-server-impl.dll / llama-server.exe relinked (245-246);
  the other 246 steps are the build-info.cpp timestamp cascade
  (everything downstream of llama.dll relinked - known harmless
  behavior when build-info regens). NOT deployed: install dir
  still holds the 10-28 ACTX binaries (2822ce088 tree); the
  timing-gate llama-server.exe swaps in at the next stop window
  (full build\bin copy = established procedure).
- `build\bin`: 10-28 WIDE REBUILD GREEN via build-sycl.ps1 -Clean
  at HEAD 2b10cfb42 on the VS 2026 toolchain (C = cl
  14.51.36231, CXX = icx 2026.1.1, Release, 553 steps; llama-
  server.exe relinked 550/553, ggml-sycl.dll 492/553). NOT yet
  deployed: the install dir (C:\Users\admin\.local\3nary-sycl\)
  holds the 10-28 user-swapped ACTX binaries (built 10-27 evening
  at 2822ce088, the B17 swap source) - the live service runs
  those; same tree state, different toolchain, so a redeploy is
  not urgent (next stop window only, if ever needed).
- `build\bin`: 10-27 (evening) WIDE REBUILD via build-sycl.ps1 at
  HEAD 2822ce088 - the reconfigure auto-picked the new HEAD (cmake/
  build-info.cmake git rev-parse), 247 ninja steps, ~3 min, all
  targets relinked incl. llama-server.exe; build\bin in sync with
  the committed tree. This WAS the B17 swap source (installed
  10-28 ~02:32 by the user).
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
| Qwen3.8-27B-UD-Q6_K (LIVE SERVICE since 10-28 handover) | `E:\DataVolume\Models\Qwen3.8-27B-UD-Q6_K\` | gguf + mmproj-BF16. 65 blocks, 16 full-attn + 48 GDN, embd 5120. The user moved the live service from Q5_K_S to Q6_K at the B17 swap. First post-handover decode: tg 12.53 t/s single-stream (91559-token conversation). |
| Qwen3.8-27B-UD-Q5_K_S (previous live service) | `E:\DataVolume\Models\Qwen3.8-27B-UD-Q5_K_S\` | 17.38 GB + mtp-Q4_0 draft 1306 MiB + mmproj-BF16. 65 blocks, 16 full-attn + 48 GDN, embd 5120. Tensor mode proven at 160K ctx (31.9 t/s, N10). Still the N17 tensor-bench candidate. |
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

## Next (10-28)

1. F40 (bad-alloc) write-up into FINDINGS - analysis done (the
   "Previous refresh" entry above holds the state: machine
   commit-exhaustion root cause, the prompt-cache site CATCHES
   bad_alloc + self-throttles 0.4x, all bad_alloc ports were
   TENSOR-mode children, the [NNNNN] prefix is the PORT not a
   PID); write it on user confirmation (now or at the stop
   window).
2. [B] The user's call on --cache-ram 0 (the F37 risk zone is
   LIVE: full 262144 ctx + 8 GiB default host prompt cache +
   82k-106k-token conversations); the F40 recommendation feeds
   it; if the bad-alloc class recurs, add it to the spawn line
   at the next stop window (user-owned config).
3. DONE 10-28 (afternoon): F39 measurement - FINDINGS F39
   [measured]/[certain]/[decision]: 2 chats batched on the
   default config (sum ~21.4 t/s, ~76% of 2/H, chats 27.6K+7K,
   NOT max-ctx-pinned) -> pipeline gate not met, NOT building;
   -np 2 root-caused (per-slot KV 2x + total adaptive-ctx cap
   524288 by design; child healthy; monitor read unproven) ->
   recommended dropped, default (auto 4 slots + unified KV)
   kept.
4. Record the first live 27B-scale `adaptive ctx: KV cache grown
   in place in X ms` regrow timing from macube/logs/llama.log
   (262144 <-> 163840, ~2.65 GiB per direction) into FINDINGS -
   closes B17 (the 129 ms shrink is recorded in the previous
   refresh entry; regrow needs a > 163840-token conversation,
   largest so far 106815).
5. B14 draft re-presentation (priority RAISED by F37/F38 - the
   structural fix for BOTH reported issues; F39 keeps B14 as
   the capacity vehicle); implement only on explicit user
   confirmation.
6. [B] Await the user's extraArgs bisection outcome, remainder:
   -ngl 99 / -c 262144 (-np 2 is ANSWERED, F39) - fold into
   FINDINGS when reported (prediction table in F31).
7. [B] Await a free card (re-derive VRAM first) -> run the Q5_K_S
   tensor bench (N17; GDN correctness check first).
8. B7a (this fork, next code item): startup loud-fail on unfittable
   ctx/KV + closed -sm/-dev gate (the old tensor warn-then-continue
   target is GONE - F32; remaining: the non-tensor startup ctx/KV
   loud-fail + the parse gate).
9. B13 recon (read-only, clone hosts live service): diff fork vs
   clone meta backend (upstream PRs #26490/#27574), ternary enum
   mapping (fork non-standard PQ2_0=142/PTQ1_0=143), missing/
   divergent work map (F25), the seq_rm fix question (server-
   context.cpp ~:3444, relevant to F28). THEN port commit-by-
   commit on user-confirmed scope.
10. Refresh private/ + commit+push at each milestone.
