# NOTES - constraints, recipes, mappings, semantics, pitfalls

"Retain significant comments" file. Everything here cost something to learn.

## N1. Hard constraints (user rules)

- User stops/starts macube services. NEVER kill a PID the user owns.
  CURRENT (10-22, re-verify after reboots): 24252 router :8080 + 26876
  child :58452 (Q5_K_S + MTP). Killing PIDs of processes WE spawned is
  fine.
- No changes to macube service files (services.json, presets). Reading OK.
- `llama-api-keys.txt` untouched.
- All SYCL binaries need oneAPI setvars first (N2).
- MACHINE HAS 2 PHYSICAL Arc Pro B70 GPUs (F9). GGML_SYCL_DEVICE is a
  NO-OP in this fork (user's shell var, used with -mg; fork never reads
  it). To pin one card: `-dev SYCL0` (name-based) or
  `--split-mode none -mg 0`. Default (layer split) scatters across both
  GPUs - that is how 61850 is intended to run, leave it alone.
- Small changes, ASCII only (no emdash, no unicode arrows in code/commits).
- Commit trailer: `Assisted-by: Claude, Qwen3.8-27B`.
- No tests/* additions, no PRs, no reviewer replies (private fork).
- PREFERRED test model (user instruction): ternary PTQ1_0
  (`E:\DataVolume\TernaryModels\Ternary-Bonsai-2-27B-PTQ1_0\`), it fits the
  scarce free VRAM reliably.
- User does NOT use `--n-spec`.
- Tensor-mode runs INTENTIONALLY use both GPUs (`-dev SYCL0,SYCL1`,
  F19/N14) - that is the feature under development, not a pin
  violation.
- No DFlash2 content: the model dirs contain DFlash2 gguf files; never
  process, document, or commit anything about them.
- Small ctx tensor-mode debug runs are safe alongside the live service
  (deterministic warmup crash dies before long generation).

## N2. setvars recipe (pwsh, oneAPI for SYCL builds/runs)

```powershell
cmd /c """C:\Program Files (x86)\Intel\oneAPI\setvars.bat"" intel64 && set" |
  ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') {
    [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
  } }
# no $env:GGML_SYCL_DEVICE - not read by this fork (F9). Pin via -dev SYCL0.
```

## N3. Build recipe (full, each fresh pwsh)

- Tree: `build/` (Ninja, Release, shared libs, tests ON).
- 1. oneAPI setvars (N2).
- 2. VS env: parse `cmd /c '"C:\Program Files (x86)\Microsoft Visual
  Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" && set'`
  with the same regex trick as N2.
- 3. Prepend `C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\lib`
  to `$env:LIB`.
- 4. `cmake --build build -j 24` (in the repo root or `cd build`).
- Running the exe also needs the oneAPI env for SYCL.
- FULL builds exit 1 at pre-existing `test-dspark-forward` LNK2019 -
  HARMLESS. Verify the failing target is that one before worrying.
- No PDBs anywhere (stack traces via cdb need a RelWithDebInfo build).

## N4. Detached launch recipe (pwsh) - the only reliable way

```powershell
# N2 setvars + $env:GGML_SYCL_DEVICE='0' first
$p = Start-Process -FilePath llama-server.exe -ArgumentList $args `
  -RedirectStandardOutput X.log -RedirectStandardError Y.log `
  -WindowStyle Hidden -PassThru
# poll Y.log or the TCP port; curl; then Stop-Process -Id $p.Id (own pid only)
```

PITFALLS (learned the hard way):
- Logger writes to STDERR (Start-Process -RedirectStandardError captures
  everything); per-line unbuffered => log position after abort is RELIABLE.
- Background `& cmd | Tee-Object` shells get torn down at turn boundaries
  and can kill the child tree.
- Even detached Start-Process children DIE at turn boundaries (the ternary
  server died right after its request). => An E2E must COMPLETE WITHIN A
  SINGLE TURN (launch, wait, curl, kill, all in one tool-call sequence).

## N5. Verbosity / log-level semantics

- `-lv N` maps directly to the ggml level: 0=NONE 1=ERROR 2=WARN 3=INFO
  4=VERBOSE 5=DEBUG 6=TRACE (max seen in logs: 2147483647).
- SRV_TRC / LOG_TRC lines are INVISIBLE below trace. The assert repro used
  -lv 3 (INFO) - do not read absence of TRC lines in it as a crash marker
  (this cost a wrong window once, see F1).
- fit.cpp has a logger wrapper that demotes sub-min-level lines (fit
  measurement output can appear with an "I" prefix).
- SYCL-MEMTRACE logs FAILED device allocations ONLY (E level, always on).
  OK allocations are not logged by default. (F3 caveat.)
- Macube: llama.log = srv-target lines only; `[NNNNN]` prefix = model
  server PORT, not PID; timestamps are mm.ss.mmm and RESTART PER RUN
  (router clock vs per-run clock - never mix them across instances).
- llama-error.log = router srv lines + spawn args + "exited with status"
  ONLY. Model-server stderr (SYCL-MEMTRACE, ggml errors) is NOT in it.
  [PORT]-prefixed E lines the user sees come from the macube UI console.

## N6. Assert-repro args (the F1/F8 matrix)

```
--model E:\DataVolume\Models\Qwen3.8-27B-UD-Q5_K_S\Qwen3.8-27B-UD-Q5_K_S.gguf
--model-draft E:\DataVolume\Models\Qwen3.8-27B-UD-Q5_K_S\mtp-Qwen3.8-27B-Q4_0.gguf
--spec-type draft-mtp
--batch-size 6144 --ubatch-size 6144
--ctx-size 98304
--cache-type-k q5_1 --cache-type-v q5_1
--cache-type-k-draft q4_0 --cache-type-v-draft q4_0
--flash-attn on --threads 56 --threads-batch 28 --jinja
[--numa distribute]
--host 127.0.0.1 --port NNNNN -lv 3
```
- build\bin => port 18999; install-dir binary => port 18998 (+ --numa
  distribute). `--fit off` variant = same args + `--fit off`.
- Max-verbosity variant: `-lv 6` (or 2147483647) to see TRC lines before
  the assert.

## N7. PTQ1_0 dequant mapping (committed - do not rederive)

block = 28 B;
- e < 80: byte = qs[e&15], digit = e>>4
- 80 <= e < 120: byte = qs[16+(t&7)], digit = t>>3
- e >= 120: byte = qh[t&1], digit = t>>1
- q = (uint8_t)(byte * pow3[digit]); xi = ((uint16_t)q * 3) >> 8;
  value = (xi - 1) * d

## N8. Fork-specific mappings (NEVER use upstream)

- ggml type enum (ggml.h:389-417): F32=0, Q4_0=2, Q5_1=7, Q8_0=8,
  Q3_K=11, Q4_K=12, Q6_K=14, PQ2_0=142 (ternary; PTQ1_0 is the
  separate ternary type, F24).
- `src/llama.h` does not exist. Fork is far behind upstream (do not port
  by line number).
- MTP draft gguf quants: output/embd=Q3_K, ffn/eh_proj=Q4_K,
  attn projections=Q6_K, norms=F32.
- qwen35: gated-delta-net hybrid, block_count=65,
  full_attention_interval=4, embd 5120, n_ctx_train 262144, no SWA.
- Fork llama-cli: `-v` boolean, level via `-lv N`; it is a thin
  in-process-server wrapper. Interactive until stdin EOF (sweeps use
  `$null | & llama-cli.exe`).
- llama-gguf.exe: 0xC0000005 pre-output on the ternary gguf - do not trust.

## N9. Exit-code reference

- -1073740791 = 0xC0000409 = STATUS_STACK_BUFFER_OVERRUN = MSVC
  abort()/__fastfail (GGML_ASSERT aborts land here; so does any __fastfail).
- 0xC0000005 = access violation (llama-gguf on ternary).
- status 1 = clean exit after handled error (58458 OOM path).
- status 0 = clean stop.

## N10. Perf baselines (t/s)

| Model | ngl | pp512 | tg128 |
|---|---|---|---|
| Q5_K_S -sm tensor (SYCL0,SYCL1, -c 160000) | - | - | 31.9 |
| PTQ1_0 SYCL (B11 kernel) | 99 | 293.75 | 26.72 |
| PTQ1_0 SYCL (old scalar) | 99 | 296.39 | 6.90 |
| PTQ1_0 SYCL | 0 | 223.60 | 0.45 |
| PQ2_0 SYCL | 99 | 237.13 | 5.40 |
| PQ2_0 SYCL | 0 | 208.62 | 0.98 |
| Q6_K SYCL | 0 | 141.48 | 1.58 |

- 10-20 B11 (commit 5b80a3bb1): PTQ1_0 SYCL ngl 99 standalone tg128
  6.90 -> 26.72 t/s (3.9x); pp512 unchanged (293.75). Current PTQ1_0
  reference; the old scalar-kernel row is historical. F14 resolved.
- Ternary under 61850 pressure: pp 10.18 / tg 3.59 t/s (still serves).
- 61850 service MTP: tg 19.70 t/s (task 12627, healthy).
- 61850 service MTP long gen (task 42323, n_gen 3400+, router clock
  143+ min): tg 15.5-15.6 t/s = ~2.2x the non-MTP 6.90 t/s baseline.
  MTP confirmed working in the service.
- 10-19 user refresh (supersedes the table for current comparisons):
  Q6_K SYCL TG 13-18 t/s on this machine is the current reference.
  User memory: PTQ1_0/PQ2_0 CUDA ternary on RTX 3090 at similar ctx:
  TG well above 50, maybe 80 t/s. PTQ1_0 vs PQ2_0: no difference here.
- 52510 ternary service (PTQ1_0, n_ctx_slot 131072, kv_unified, 4 slots):
  pp ~300 t/s; tg 6.59 (55.7k ctx) -> 6.14 (62.4k) -> 7.07 -> 9.24 (30.0k
  ctx). Trend tracks KV size (effective GB/s ~constant), not warmup.
  See F14.

## N11. Key code locations (fork line numbers)

- server-context.cpp: :994 SRV_INF "loading model", :995 SRV_TRC "local
  path", :1016-1029+ mmproj mem estimate (has_mmproj && fit_params),
  :1072 `if (has_spec)`, :1093-1102 Part-3 fallback, :1104-1105
  `params_base.speculative.draft.ctx_dft = ctx_dft;`, :3584-3633 context
  ladder, :3589-3590 upstream TODO (preemption design ref).
- common/common.cpp: :397 common_params_print_info def; :1230-1464 READ
  10-19 (common_init_result ctor :1290+: fit gate :1295, "fitting params"
  COM_TRC :1296, common_fit_params call :1318, real load :1327+).
- src/llama.cpp: :157-312 llama_prepare_model_devices (-dev list as-is
  :159-183; default selection skips CPU :224-227, RPC+GPU list :274-284;
  single-GPU mode `-mg` branch :288-299, GPU-only index).
- common/arg.cpp: :1116-1136 parse_device_list (NAMES via
  ggml_backend_dev_by_name, CPU rejected); :1138 common_print_available_
  devices; :2727 -dev/--device; :2842 -mg/--main-gpu (env
  LLAMA_ARG_MAIN_GPU); :4150 draft -dev.
- common/common.h: :467 main_gpu default 0; :476 split_mode default
  LLAMA_SPLIT_MODE_LAYER.
- common/fit.cpp: :29-148 common_get_device_memory_data_impl (model
  no_alloc+NONE :55-57, ctx from *cparams :65); :178 common_params_fit_
  impl; :270-277 n_ctx_auto block; :453-455 explicit -c "no change";
  :458-460 throw (nd==0); :462-464 throw (n_gpu_layers user-set);
  :588+ get_memory_for_layers lambda (back-to-front dense fill).
- common/speculative.cpp: :2129-ish Part-1 edit; :3078-3127
  common_base_params_to_speculative (draft path swap:
  `result.model = params_spec.mparams`; n_outputs_max = n_parallel,
  n_outputs_max_per_seq = 1); :3138-3190 draft init ctor
  (ctx_type MTP :3152, n_ctx = llama_n_ctx(ctx_tgt) :3158,
  ctx_other = ctx_tgt :3161, model from params.model.path :3172,
  NO-throw on load failure :3173-3176); :3258 DRAFT_MTP gate on
  params.draft.ctx_dft != nullptr.
- ggml/src/ggml.c: :1628-1674 ggml_init (ctx alloc :1642 unchecked,
  ggml_aligned_malloc + GGML_ASSERT(ctx->mem_buffer != NULL) :1661).
- tools/server/server.cpp: :152-157 n_parallel auto => 4 + kv_unified.
- ggml/src/ggml-sycl/vecdotq.hpp: :987 VDR_PQ2_0=4, :1018 vec_dot_pq2_0_
  q8_1 (dp4a), :1066 VDR_PTQ1_0=4, :1075 vec_dot_ptq1_0_q8_1_multi
  (byte-parallel carry-safe trit decode + dp4a, x decoded once for up to
  3 y columns; B11 port of CUDA :808, F14), :1175 vec_dot_ptq1_0_q8_1
  (wrapper -> _multi<1>).
- ggml/src/ggml-sycl/mmvq.cpp: :141 mul_mat_vec_q (generic), :186
  mul_mat_vec_q_ncols (re-runs vec_dot per column), :1401 PTQ1_0 single-
  col launcher, :1423 PTQ1_0 fused multi-column kernel (x decoded once,
  up to 3 y cols), :1481 fused launcher, :1502 generic ncols launcher,
  :1523-1537 ncols switch (2-3 fused, 4-8 generic, 1 single),
  :2781-2788 dispatch, :2923 MoE launcher.
- ggml/src/ggml-cuda/vecdotq.cuh: :808 vec_dot_ptq1_0_q8_1_multi (byte-
  parallel trit decode + dp4a, decodes x once for up to 3 columns),
  :895 vec_dot_ptq1_0_q8_1, :955 vec_dot_pq2_0_q8_1.
- ggml/src/ggml-cuda/mmvq.cu: :719-742 PTQ1_0 special dispatch (multi
  ncols), :670-710 GB10-only prefetch path (not for B70).
- server-context.cpp (slot scheduling, F11): :55 slot_state enum,
  :2288-2324 slot admission + defer (:2297, waits indefinitely, no
  eviction), :3590-3639 decode-fail path (errors+releases ALL processing
  slots; TODO :3595), :1927-1935 send_error, :1580 try_clear_idle_slots,
  :2326-2342 prompt_save/prompt_cache, :2490-2533 restore-slot path.
- Tensor mode (F19-F24, 10-22):
  - ggml/src/ggml-backend-meta.cpp: :577-608 handle_mul_mat,
    :610-662 handle_reshape, :813-828 handle_gated_delta_net,
    :830-861 calculate_split_state (callback only non-compute non-view
    :834), :1066-1105 post-pass (div assert :1085-1087),
    :1181-1299 init_tensor_impl (local = global ne, split axis scaled),
    :1188 n_segments<=16 cap, :1410-1474 write path.
  - ggml/src/ggml-backend.h: :378-415 split-state struct/semantics,
    :498-505 split_states_equal (axis + per-device SUMS only).
  - src/llama-model.cpp: :368 split-state callback, :540-560 QWEN35
    segment table (ssm_out {{2048,3}} :541-542), :1190-1344 prism
    hadamard kvs, :1532-1541 mmap gate (SYCL => use_mmap=false),
    :1940 ml.load_all_data call, :1951-2109 hadamard builder (rot
    :2010-2034, signs :2043-2087 shared per width, transform +
    perm_hd/nk/rep :2089-2100).
  - src/llama-model-loader.cpp: :1551-1672 load loop (mmap-alloc :1582,
    mmap-set :1592, host read :1599, async chunked 1/64 MB pinned
    :1607-1658, plain read_buf->set :1660-1663).
  - src/llama-model-loader.h: :31 llama_model_loader (plain struct, NO
    split-mode knowledge), :33-50 llama_tensor_weight, :204
    load_all_data.
  - src/llama-graph.cpp: :1546-1603 build_lora_mm (memo :1556, perm
    gate :1561, perm ops :1563-1567, signs :1569-1571, rot :1572,
    mul_mat :1578). src/llama-graph.h: :23-32 llama_hadamard_transform
    (H-A adds `folded` here), :33 hadamard_rotations map.
  - src/llama-impl.h: :57-72 llama_mul_mat_hadamard.
  - src/models/qwen35.cpp: :89 ssm_out weight [K=6144, N=5120],
    :403-624 GDN block graph, :576 build_lora_mm(ssm_out, final_output,
    ssm_out_s). GDN hparams: head_k/v_dim=128, n_k_heads=16,
    n_v_heads=48, key_dim=2048, value_dim=6144, conv_dim=10240,
    ssm_d_conv=4, ssm_dt_rank=48, ssm_n_group=16.
  - ggml/src/ggml-quants.c: :113 quantize_row_pq2_0_ref, :494
    dequantize_row_pq2_0, :2183 quantize_pq2_0. ggml-cpu/quants.c:33
    cpu wrapper. ggml-common.h: :202-207 block_pq2_0 (34 B), :214-220
    block_ptq1_0.
  - ggml/src/ggml-sycl/fwht.cpp + fwht.hpp: FWHT fast path ONLY n in
    {64,128,256,512}; ggml-sycl.cpp:4528 HADAMARD-hint MUL_MAT dispatch.
  - GDN pairing (MODULO both backends): CPU ops.cpp:10849, SYCL
    gated_delta_net.cpp:41.

## N12. Git / repo state notes

- Branch main; committed+pushed (this project, recent first):
  48e9082ca (fit-params: include speculative draft model in the fit),
  835d728b3 (docs: close B11 in private library), 5b80a3bb1 (PTQ1_0
  SYCL byte-parallel kernel, B11), e844b224a (private library reopen),
  11c3d328a (closure docs), 7faf42ec4 (private/ refresh), 0589b1e1a
  (MTP 3-part fix), 02af668bf (private library), 355b4a2b3 (KV
  diagnostic), a3f36885c (KV guard), 033a6f0ad (PTQ1_0 prefill),
  bd993a9d7 (SYCL kernel).
- Upstream ref for Part 1: d1a92352c "speculative : only gemma4-assistant
  shares the target KV cache" (NOT in fork history).
- `private/` = this knowledge library; keep it committed+pushed.

## N13. Working assumptions to re-verify after compaction

(Refreshed 10-22; re-verify PIDs after any reboot.)
- LIVE service (do not disturb, user-owned): 24252 = llama-server
  router 0.0.0.0:8080; 26876 = llama-server child 127.0.0.1:58452
  (Q5_K_S + MTP). Check before VRAM work:
  `Get-NetTCPConnection -LocalPort 8080,58452 -State Listen`.
  (Old 52416/61850/52510/65463/54447 entries are history; the service
  was restarted, PIDs changed.)
- build\bin is CURRENT as of the 10-20 B11 build + 48e9082ca fit tool.
  Any tensor-mode (B12) build updates this.
- Install dir (C:\Users\admin\.local\3nary-sycl\): state unknown after
  the service restart; re-check before advising a redeploy.
- $env:TEMP\opencode\ logs persist on disk (re-read, don't assume
  memory). Key tensor-mode artifacts: pq2_meta_debug.txt (4379-line
  GGML_META_DEBUG trace; assert at :4378) + run_pq2_debug.ps1. B11
  logs: ptq10-b11-bench.log, ptq10-e2e2.log.
- Model paths (verified 10-22): C:\DataVolume\TernaryModels\Ternary-
  Bonsai-2-27B-PQ2_0\ (PQ2_0 gguf 6.71 GB + mmproj, the crashing
  model) and E:\DataVolume\TernaryModels\ (full family: 27B PTQ1_0
  5.54 GB + PQ2_0, 2B/4B/9B ternary, small MTP Q8_0 variants).
  TernaryModels.presets.ini in both roots = macube presets, DO-NOT-
  EDIT.
- C:\Users\admin\source\ai\llama.cpp = the UPSTREAM clone (B13 port
  target, F25); C:\Users\admin\... and C:\Users\gorom\... paths are the
  SAME tree for llama.3nary (one build, shared build\bin).

## N14. Tensor-mode repro (PQ2_0 assert, F20)

```
$env:GGML_META_DEBUG='1'   # SPLIT_STATE prints need BOTH of these
build\bin\llama-cli.exe -m C:\DataVolume\TernaryModels\Ternary-Bonsai-2-27B-PQ2_0\Ternary-Bonsai-2-27B-PQ2_0.gguf -mm C:\DataVolume\TernaryModels\Ternary-Bonsai-2-27B-PQ2_0\Ternary-Bonsai-2-27B-mmproj-BF16.gguf -ngl all -sm tensor -c 160000 -dev SYCL0,SYCL1 -lv 5 -p "What time is it"
```
- Deterministic crash at warmup forward, block 0 (assert
  ggml-backend-meta.cpp:1086) - safe to run alongside the live service
  (small ctx, dies before any long generation).
- The runner script in $env:TEMP\opencode\run_pq2_debug.ps1 wraps this
  with setvars (N3 env).
- Expected post-fix (B12/H-A): no assert; SPLIT_STATE for the ssm_out
  chain = {axis 0, {1024x3, 1024x3}} (CORRECTED 10-23, was written as
  2048x3 - the segment is 1024); output sane (slight degradation from
  the extra requant pass is EXPECTED). VERIFIED 10-23: see F26/F27.

## N15. Port-target clone (B13, F25)

- Path: `C:\Users\admin\source\ai\llama.cpp`. Clone of UPSTREAM
  ggml-org/llama.cpp (origin), branch master, HEAD 6fcaa16f4 (2026-09-21).
- Git state (verified 10-22): NO local user commits; the "[origin/
  master: ahead 53]" marker is a STALE fetch ref, not local work.
  Working tree has 3 uncommitted tracked-file mods from earlier
  sessions (DO NOT OVERWRITE): common/chat.cpp (tool-args JSON throw ->
  warn), scripts/ui-assets.cmake (npm.ps1 in find_program),
  tools/server/server-context.cpp (~:3444 recurrent-memory seq_rm
  refusal: clear seq + re-process full prompt).
- Build: `build-sycl.ps1` (in clone root). It runs oneAPI setvars,
  checks icx, `sycl-ls`, then cmake in build/ with Ninja:
  GGML_SYCL=ON, GGML_SYCL_F16=ON, GGML_OPENMP=ON, C=cl, CXX=icx,
  BUILD_SHARED_LIBS=ON, Release, LLAMA_OPENSSL (auto-installs via
  winget if absent). Flags: -Clean (wipe build/), -Portable (copy to
  ~/.local), -InstallOnly. Build: `cmake --build build -j 8`.
- Run: `run-llama-sycl.ps1 <tool> [args...]` (e.g. `run-llama-sycl.ps1
  cli -m model.gguf -ngl 99`). It re-runs oneAPI setvars and prepends
  ~/.local\llama-sycl to PATH (portable bin dir). Binaries also in
  clone build\bin\.
- Do NOT confuse with the 3nary repo remotes: 3nary's "origin" =
  goromil/llama.3nary (private), 3nary's "prism" = PrismML-Eng/llama.
  cpp. The clone's "origin" = upstream ggml-org/llama.cpp.
- 10-23: the LIVE macube service is now deployed from the CLONE build
  (compile-time path in its logs; child 50630 serving Q5_K_S in TENSOR
  mode, after 62743 died - F28). The old PIDs 24252/26876 are STALE;
  re-derive live PIDs via Get-NetTCPConnection -LocalPort 8080,58452.

## N16. Fork CLI facts learned 10-23 (B12 verification)

- llama-cli in this fork is a CLIENT-SERVER HYBRID (tools/cli/): it
  spawns an IN-PROCESS llama_server on a random 127.0.0.1 port
  (cli-server.h) and drives it over HTTP. Its log therefore contains
  `srv llama_server` + `slot print_timing` lines, and the "answer"
  arrives as streamed chat chunks (reasoning_content first for
  thinking models - 16-token budgets get EATEN by the thinking block,
  use >= 128 tokens to see the actual answer).
- Exit code from the hybrid is UNRELIABLE (observed 253 at REPL EOF
  after a fully successful run). Judge by the log, not the exit code.
- The small 10752-B tool exes in build\bin (llama-cli, llama-server,
  llama-bench, llama-completion, ...) are thin dispatch stubs; heavy
  code is in llama.exe / the shared libs. llama-cli.exe TIMESTAMP STAYS
  STALE across builds - judge by behavior, not mtime (the N3 note
  about incremental builds).
- NO sampling args in this fork's common args (no -temp in
  common/arg.cpp): server-side default temp = 0.80 (common.h:236).
  llama-server DOES take --temp (see tools/server/tests/utils.py:160).
  For a deterministic (greedy) comparison run llama-server + HTTP, or
  compare token SANITY, not exact tokens.
- -sm (split mode) values parse in common/arg.cpp:2804 (tensor there;
  none/layer/row standard). -sm none pins ALL layers to the first
  device - do NOT use it while the live service occupies SYCL0 (only
  ~5 GB free there); use -sm layer for a two-GPU baseline.
- Verify-run log capture: pwsh in-memory capture (`2>&1` into a
  variable) can be LOST when the runner dies (the verify2 log-missing
  incident). Direct file redirection survives: `cmd /c "exe args > file
  2>&1"` (runner: $env:TEMP\opencode\run_verify_direct.ps1; token
  dumper: dump_tokens.ps1).
- Live-service log lines: 5th column = level (D/I/W/E). LCP slot
  selection lines (`selected slot by LCP similarity, f_sim_best = ...`)
  are INFO, not warnings - they report warm-slot KV reuse.

## N17. Q5_K_S tensor bench (user runs it, needs a free card, 10-25)

Exact bench line for Qwen3.8-27B-UD-Q5_K_S in tensor mode (the BLOCKED
item; GDN correctness check FIRST, then speed):

    cmd /c "C:\Users\gorom\source\ai\llama.3nary\build\bin\llama-server.exe -m E:\DataVolume\Models\Qwen3.8-27B-UD-Q5_K_S\Qwen3.8-27B-UD-Q5_K_S.gguf --model-draft E:\DataVolume\Models\Qwen3.8-27B-UD-Q5_K_S\mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --cache-type-k q5_1 --cache-type-v q5_1 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 --ctx-size 131072 --batch-size 6144 --ubatch-size 6144 --threads 56 --threads-batch 28 --flash-attn on -fit off -ngl 99 -sm tensor -port 8123 -lv 5 > C:\Users\admin\AppData\Local\Temp\opencode\q5ks-tensor-bench.log 2>&1"

- Use build\bin (the 10-25 SYCL=ON build, F30); do not bench with the
  10-21 install tree's exe until its link state is re-verified. The
  original line used C:\Users\admin\.local\3nary-sycl\llama-server.exe
  with the same args.
- -ngl 99 + -sm tensor without -dev uses BOTH cards (tensor
  intentionally spans SYCL0+SYCL1, F19) - it needs the VRAM the live
  service holds (PID 212452 as of 10-25), so it runs only after the
  user frees a card / stops the service.
- GDN correctness first: short generation, sane output (no garbage /
  loops) before trusting the t/s.
