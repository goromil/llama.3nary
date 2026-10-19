# NOTES - constraints, recipes, mappings, semantics, pitfalls

"Retain significant comments" file. Everything here cost something to learn.

## N1. Hard constraints (user rules)

- User stops/starts macube services. NEVER kill a PID the user owns
  (61850: model 568492, router 421716). Killing PIDs of processes WE
  spawned is fine.
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

## N2. setvars recipe (pwsh, oneAPI for SYCL builds/runs)

```powershell
cmd /c """C:\Program Files (x86)\Intel\oneAPI\setvars.bat"" intel64 && set" |
  ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') {
    [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
  } }
# no $env:GGML_SYCL_DEVICE - not read by this fork (F9). Pin via -dev SYCL0.
```

## N3. Build recipe

- Tree: `build/` (Ninja, Release, shared libs, tests ON).
- `cmake --build . -j 24` in build/ after N2.
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
  Q3_K=11, Q4_K=12, Q6_K=14.
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
| PTQ1_0 SYCL | 99 | 296.39 | 6.90 |
| PTQ1_0 SYCL | 0 | 223.60 | 0.45 |
| PQ2_0 SYCL | 99 | 237.13 | 5.40 |
| PQ2_0 SYCL | 0 | 208.62 | 0.98 |
| Q6_K SYCL | 0 | 141.48 | 1.58 |

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
  q8_1 (dp4a), :1066 VDR_PTQ1_0=4, :1068 vec_dot_ptq1_0_q8_1 (scalar trit
  decode into local int8 q[128] + 128 scalar MACs = the slow path, F14).
- ggml/src/ggml-sycl/mmvq.cpp: :141 mul_mat_vec_q (generic), :186
  mul_mat_vec_q_ncols (re-runs vec_dot per column), :1401-1459 PTQ1_0
  launchers + ncols switch, :2836-2842 MoE launchers.
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

## N12. Git / repo state notes

- Branch main; committed+pushed (this project, recent first):
  11c3d328a (closure docs), 7faf42ec4 (private/ refresh), 0589b1e1a (MTP
  3-part fix), 02af668bf (private library), 355b4a2b3 (KV diagnostic),
  a3f36885c (KV guard), 033a6f0ad (PTQ1_0 prefill), bd993a9d7 (SYCL
  kernel).
- Upstream ref for Part 1: d1a92352c "speculative : only gemma4-assistant
  shares the target KV cache" (NOT in fork history).
- `private/` = this knowledge library; keep it committed+pushed.

## N13. Working assumptions to re-verify after compaction

- 61850 still alive (was at snapshot time). Check ports before any
  VRAM work: `Get-NetTCPConnection -LocalPort 61850 -State Listen`.
- build\bin is one \n-edit behind source (rebuild, PLAN step 2).
- Install dir still has OLD binaries (copy after 61850 stop, PLAN step 7d).
- $env:TEMP\opencode\ logs persist on disk (re-read, don't assume memory).
