# FINDINGS - research findings with evidence

Confidence: [certain] / [high] / [medium] / [open].
Every finding lists evidence and open questions. Newest first where relevant.

## F1. ggml.c:1661 assert (draft+spec path) - window PINNED

[certain] The assert is host-side: `ggml_init` (ggml/src/ggml.c:1628-1674):
ctx alloc at :1642 is unchecked, then `ggml_aligned_malloc(mem_size)` returns
NULL and `GGML_ASSERT(ctx->mem_buffer != NULL)` fires at :1661. It is a HOST
malloc failure, NOT a device OOM.

[certain] Crash window (corrected 2026-10-19):
- Assert log (mtp-server-98304.log, 8 lines) ends at
  `I srv load_model: loading model '...'` = server-context.cpp:994 (SRV_INF).
- The assert run used `-lv 3` = INFO level.
- server-context.cpp:995 `SRV_TRC("local path ...")` is TRACE level and is
  INVISIBLE at -lv 3. Its absence in the 8-line log proves NOTHING about
  :995. (Earlier assumption "crash between :994 and :995" was WRONG.)
- "fitting params to device memory ..." is INFO (common.cpp, common_init_)
  and NEVER printed in the assert run.
- THEREFORE the crash sits between server-context.cpp:994 and the
  "fitting params" print in common_init_. Contents of that window:
  server-context.cpp :995-:1013 (mmproj params setup; skipped when no mmproj),
  :1016+ (mmproj memory estimate, only if has_mmproj), :1030+ (UNREAD -
  presumably spec params setup and the common_init_from_params call),
  common_init_from_params entry, common_init_ entry up to the fit print.
- The no-draft run (same verbosity settings, same pressure) did NOT assert;
  it reached fit and died later at the real KV device alloc (1536 MiB).
  So the trigger correlates with the draft+spec code path inside the window.

[certain] Window narrowed (10-19, common.cpp:1230-1464 +
server-context.cpp:1030-1139 now READ):
- server-context.cpp:1047-1049 progress callback attach, :1051
  `llama_init = common_init_from_params(params_base);`
- common.cpp common_init_result ctor: :1292-1293 mparams/cparams
  conversion (pure), :1295 `if (params.fit_params)`, :1296 "fitting params
  to device memory ..." is COM_TRC (NOT INFO - hidden at -lv 3), :1304
  common_base_params_to_speculative (pure), :1311-1316 extra-model struct,
  :1318 common_fit_params, :1327+ real model load.
- fit.cpp:262 "getting device memory data for initial parameters:" is
  LOG_TRC (hidden at -lv 3).
- So at -lv 3 NOTHING inside the fit path prints before the first
  llama_model_loader INFO "loaded meta data ..." line. The 8-line assert
  log has none of those => the assert fired during/around fit's FIRST
  metadata load + measurement-ctx creation (the only ggml_init in the
  window), i.e. a host malloc failure inside fit measurement - but see the
  max-verbosity rerun below: same config survived fit and hung later.
[high] 10-19 max-verbosity rerun (mtp-server-98304-maxver-err.log,
122504 lines, rebuilt binary, warm page cache): NO assert. Fit ran ~37 ngl
iterations in ~1.6 s (warm cache; cold it took 28.32 s), then the process
HUNG 118 s at `sched_reserve: reserving ...` inside fit's FINAL
measurement ctx (candidate "offloaded 18/66 layers to GPU", tail layers
61-65 assigned to SYCL1, recurrent RS buffers CPU 1795 MiB + SYCL1
598 MiB). Killed at the 120 s timeout.
=> Under 61850 pressure the draft+spec 98304 config fails DIFFERENTLY per
run (host malloc assert vs allocation stall): pressure-dependent OOM, NOT
a deterministic code path. Which failure appears depends on timing and on
which physical GPU receives the tail layers (F9: no pin was active in any
run so far).
[open] Remaining experiments: rerun with real pin `-dev SYCL0` (a) under
pressure, (b) on free VRAM after user stops 61850, (c) `--fit off`
variant. If (b) loads clean => pressure theory confirmed, assert = host
commit exhaustion under dual-GPU scatter.

## F2. Fit loop runs even with explicit -c (28.32 s, 56 iterations)

[certain] With explicit `--ctx-size 98304`:
- fit.cpp:270-277 (n_ctx_auto block) is skipped; :453-455 logs
  "context size set by user to 98304 -> no change" (LOG_TRC) EXACTLY ONCE
  (verbose log :2442).
- fit STILL runs the step-3 ngl-candidate loop: 56 "memory for test
  allocation" iterations, 58 "loading model tensors (load_mode = none)"
  metadata loads, 1 "getting device memory data" (initial), total 28.32 s
  (verbose log :25 -> :135601). block_count=65 => consistent with a
  back-to-front ngl sweep.
- Per-iteration block shape (verbose :4814):
  "memory for test allocation by device:" / per-device
  "id=N, n_layer=..., n_part=..., overflow_type=4, mem=... MiB" /
  "filling dense layers back-to-front:" / metadata model load.
- First iteration showed id=0 mem=603 MiB, id=1 mem=0 MiB (nd=2: pinned
  SYCL0 + CPU under --numa distribute).

## F3. Fit ctx device-allocation behavior - empirically inert, source UNVERIFIED

[high] Fit measurement does NOT do visible real device KV allocs:
- Zero SYCL-MEMTRACE lines in the whole 56-iteration fit window.
- The only SYCL-MEMTRACE line in the 9.7 MB verbose log is the FINAL real
  load's `alloc FAILED: 1536.000 MiB direct` at :137874, AFTER fit done.
- If fit ctx had allocated the same 1536 MiB device KV under that pressure,
  it would have failed identically and thrown
  std::runtime_error("failed to create llama_context from model") - that
  message never appears in any log.
CAVEAT: SYCL-MEMTRACE logs FAILED allocations only (E level); OK allocs are
not logged by default (ternary-d-err.log has 0 lines despite a real load).
So absence of trace lines is not direct proof.
[certain] fit.cpp:55-57 sets model `no_alloc=true, load_mode=NONE` for the
measurement load; fit.cpp does NOT touch cparams.no_alloc anywhere (grep:
only :56). The measurement ctx is `llama_init_from_model(model, *cparams)`
(fit.cpp:65) with the CALLER's cparams.
[open] Whether the caller in common.cpp sets cparams.no_alloc before fit.
Read common.cpp fit call site to close this.

## F4. KV ring overcommit bug - ROOT CAUSE + fix committed

[certain] Server auto branch (tools/server/server.cpp:152-157):
n_parallel < 0 => n_parallel = 4 AND kv_unified = true.
With a preset `-c` (explicit ctx), per-slot n_ctx_seq = full -c, but the
shared unified ring is sized for full -c TOTAL => 4x overcommit => KV
halving/misbehavior. Upstream has the identical behavior with no fix
(issues #23652, #25566, #26558, #17316,
lmstudio-ai/lmstudio-bug-tracker#2201) => forking a guard is justified.
Fixed in fork: guard `a3f36885c` + diagnostic `355b4a2b3` (both
slot/ring-neutral), committed+pushed.
User's config fix E (remove `-c` from preset) is the user's own task.

## F5. MTP OOM 3-part fix - committed+pushed (0589b1e1a)

[certain] All three edits (common/speculative.cpp,
tools/server/server-context.cpp): rebuilt, committed+pushed as 0589b1e1a,
deployed to the install dir 10-19.
- Part 1 (speculative.cpp, old line ~2129): verbatim port of upstream
  d1a92352c "speculative : only gemma4-assistant shares the target KV cache":
  `char arch[64] = {0};` + `llama_model_meta_val_str(...,
  "general.architecture", arch, sizeof(arch));` +
  `is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt &&
  std::strcmp(arch, "gemma4-assistant") == 0;`
  keeping `chain_heads = n_mtp_layers > 1 && !is_mem_shared;`
- Part 2: doc comment above `cparams.n_ctx = llama_n_ctx(ctx_tgt);` +
  `GGML_ASSERT(llama_n_batch(ctx_dft) >= llama_n_batch(ctx_tgt) &&
  "MTP draft n_batch must be >= target n_batch");` (no behavior change).
- Part 3 (server-context.cpp :1093-1102, inside `if (has_spec)` at :1072):
  fallback when ctx_dft == nullptr: if has_draft => SRV_ERR + return false
  (stays fatal); else SRV_WRN "failed to create MTP context, continuing
  without speculative decoding" + spec_init.reset().
  `params_base.speculative.draft.ctx_dft = ctx_dft;` (:1104-1105) stays null
  => gated at speculative.cpp:3258. No dangling pointers.

[high] GAP: all service presets (Qwen3.8 AND gemma-4) use a draft gguf
(has_draft=true) => the Part-3 graceful fallback NEVER fires in service
runs; MTP ctx failure stays fatal there. Flagged to user; offered to extend
the fallback to the has_draft case.
[certain] Mode map post Part 1: Qwen3.8-27B => !is_mem_shared (mirror KV
active). gemma4-assistant arch WOULD be is_mem_shared (shares target KV) -
relevant if user runs gemma MTP on free VRAM.
[certain] Abort risk resolved: speculative.cpp:3258 gates DRAFT_MTP on
`params.draft.ctx_dft != nullptr`; the init-result ctor does NOT throw on
ctx failure; server spec call sites are null-safe.
[open] Part-3 fallback cannot trigger naturally on the Qwen target gguf
(it contains blk.64 nextn tensors). Trigger options: a model WITHOUT nextn
tensors, forced OOM, or skip with a note.

## F6. 58458 gemma-4 MTP service - clean device KV OOM (tracking DONE)

[certain] gemma-4-26B-A4B-it-UD-Q5_K_XL on OLD install binary,
`--model-draft mtp-gemma-4-26B-A4B-it.gguf --spec-type draft-mtp`,
ctx 131072, fit on, flash-attn on, load-mode none, KV q5_1 / draft KV q4_0,
--numa distribute, image-max-tokens 4000 / min 1024, min-p 0.0.
Spawned 46.25 (router clock), died ~2 min later (own clock 1.59.527):
first error = real target-ctx KV alloc `960.000 MiB direct` FAILED
=> "failed to allocate buffer for kv cache"
=> "exiting due to model loading error", status 1.
NOT the ggml.c:1661 assert (different failure layer: device buffer vs host
malloc). Fit did NOT fail under pressure (first error is the real KV alloc)
=> supports F3. Note: the E lines the user saw came from the macube UI
console ([58458] prefix); they are NOT in llama-error.log.

## F7. 58186 ternary service - 0xC0000409, cause not in logs (OPEN)

[certain] Old install binary. Spawn args (llama-error.log :4857-4914):
--model ternary PTQ1_0 gguf, --mmproj (BF16 gguf), --threads 56,
--threads-batch 28, --ubatch-size 6144, (more args in that span).
Spawned 48.38, `exited with status -1073740791` (= 0xC0000409, MSVC
abort/__fastfail) at 49.31.266.506 => died ~52 s in.
No error lines at all before exit (model-server stderr is not captured in
llama-error.log) => cause not determinable from logs.
Precedent: llama-error.log:4074 - an older Qwen3.8 service instance exited
with the same 0xC0000409 (that one was the assert repro as a service).
[open] If user asks: re-spawn ternary service with the NEW build\bin binary
and capture its own stderr (not via macube).

## F8. Assert repro matrix (MTP edits exonerated twice)

[certain] Three repro runs (logs in $env:TEMP\opencode\):
(a) build\bin draft+spec 98304 (-lv 3, port 18999) => 8-line assert ending
at "loading model".
(b) build\bin NO-draft, same pressure, max verbosity => clean GPU OOM at
real KV alloc (1536 MiB), fit completed first in 28.32 s.
(c) install-dir binary (no MTP edits) draft+spec 98304 + --numa distribute
(port 18998) => identical 8-line assert.
=> The assert exists WITHOUT the MTP edits; it is a pre-existing fork
defect in the draft+spec window (F1).

## F9. SYCL topology: TWO PHYSICAL Arc Pro B70 GPUs (corrected 10-19)

[certain] The machine has TWO physical Intel Arc Pro B70 32 GB GPUs,
enumerated SYCL0 and SYCL1, plus CPU. Proof: at the same instant the free
MiB differ (10626 vs 1051, maxver log L4-L5) - a duplicated enumeration of
one card would show equal free values. Under 61850 they hold ~21 GB +
~31 GB = ~52 GB in use total, impossible on a single 32 GB card.
61850 (no pin in its preset) spreads its model across BOTH GPUs.
The earlier "SYCL1 is a phantom" theory is WRONG.
[certain] GGML_SYCL_DEVICE is NOT read by this fork: the full SYCL env
list (ggml-sycl.cpp:313-342: GGML_SYCL_DEBUG, _ENABLE_OPT, _ENABLE_GRAPH,
_DNN, _FA_ONEDNN, _ENABLE_VMM, _FUSION, _ESIMD, _USM_SYSTEM, _MEMTRACE,
... ) has no device selector. Every earlier `$env:GGML_SYCL_DEVICE='0'`
was a NO-OP (it is the user's own shell variable, used with -mg; the fork
code itself never looks at it).
[certain] Verified device-selection semantics (10-19, user pointed out
the index distinction - CONFIRMED):
- `-mg/--main-gpu N` (env LLAMA_ARG_MAIN_GPU): index into the prepared
  GPU-ONLY device list (CPU is explicitly skipped, llama.cpp:224-227),
  and it only applies with `--split-mode none` (llama.cpp:288-299).
  So -mg 0 = SYCL0, -mg 1 = SYCL1 - a different index space than raw
  backend indices (which include CPU).
- `-dev/--device` (arg.cpp:2727, parse_device_list arg.cpp:1116-1136):
  takes device NAMES (ggml_backend_dev_by_name, e.g. "SYCL0"), not
  indices, and rejects CPU entries.
- Default split_mode = LLAMA_SPLIT_MODE_LAYER (common.h:476) and the
  61850 preset passes neither -dev nor -mg => both GPUs stay in the
  device list and layers scatter across them (this is BY DESIGN for
  61850, and it is why our repro runs landed tail layers on SYCL1).
- Real pin options: `-dev SYCL0`, or `--split-mode none -mg 0`, or
  ONEAPI_DEVICE_SELECTOR=level_zero:0 (oneAPI runtime filter).
[open] Our repro runs scattered tail layers (61-65) onto the nearly full
SYCL1 (1051 MiB free) and stalled there (F1 rerun). Rerun repros with a
real pin (`-dev SYCL0`) to isolate single-GPU behavior.

## F10. Ternary standalone E2E - PASS (closed)

[certain] New build\bin, standalone server, ternary PTQ1_0 + mmproj, under
61850 pressure: listening at 12.7 s, task 0, prefill 59 tok 5.80 s (10.18
t/s), decode 32 tok 8.62 s (3.59 t/s), total ~14.4 s (curl 14.515 s),
"stop processing: n_tokens = 90, truncated = 0" = clean slot release.
Process exited AFTER the request = harness process-tree reclamation at turn
boundary, NOT a bug. The exit-code-1 mystery is closed.
Standalone baselines (no pressure): PTQ1_0 SYCL ngl=99 pp512 296.39 t/s /
tg128 6.90 t/s.

## F11. Preemption design (Option C, user-chosen, vLLM-style)

[certain] Upstream TODO at server-context.cpp:3589-3590 is the design
reference; the context ladder is at :3584-3633. Plan:
- C0: on slot-alloc failure, error ONLY the failing slot (not the whole
  server).
- C1: pick victim = largest active slot, prompt_clear + rebuild its
  resource claim, retry the failing slot, abort the victim with an explicit
  "preempted" reason.
- C2 (optional): silent resume of the preempted slot.
[certain] Located 10-19 (no server-slot.cpp in fork; everything is in
server-context.cpp): slot_state enum :55 (IDLE, WAIT_OTHER, STARTED,
PROCESSING_PROMPT, DONE_PROMPT, GENERATING); slot admission :2288-2324
(get_available_slot, no slot => queue_tasks.defer :2297 - the request waits
indefinitely, there is NO eviction today); decode-fail path :3590-3639
(currently errors + releases ALL processing slots; TODO :3595 says terminate
only the largest); send_error :1927-1935; try_clear_idle_slots :1580;
prompt_save/prompt_cache :2326-2342; restore-slot path :2490-2533.
C1 (preempt, explicit reason): when a deferred task still cannot get a slot
or llama_decode fails at :3594 (ret==1, KV full), pick the victim = the
GENERATING slot with the largest KV claim, send_error(slot, "preempted")
on its response stream, slot.release() + slot.prompt_clear() to free the
ring cells, then re-run get_available_slot for the deferred task and launch
it. The victim client gets an explicit error and re-sends its prompt.
C2 (silent resume): instead of erroring the victim, prompt_save it into the
existing prompt_cache (:2326), mark the slot preempted (new slot_state or
flag), and after the preempting request finishes, re-inject the victim task
into the queue: re-prefill from the prompt cache (or the restore-slot path
:2533) and continue generation. The client sees a stall, not an error.
Needs: task re-injection plumbing and re-attaching the client response
stream (queue_results) to the resumed slot.

## F12. Fork vs upstream divergences (never mix mappings)

[certain] Fork ggml type enum is NON-STANDARD (ggml.h:389-417):
F32=0, Q4_0=2, Q5_1=7, Q8_0=8, Q3_K=11, Q4_K=12, Q6_K=14.
Fork is far behind upstream (upstream line 1414 ~ fork line 2129 for the MTP
ctor). `src/llama.h` does NOT exist in the fork.
MTP draft gguf quant types: output/embd=Q3_K(11), ffn/eh_proj=Q4_K(12),
attn projections=Q6_K(14), norms=F32.
qwen35 arch: gated-delta-net hybrid, block_count=65,
full_attention_interval=4, embd 5120, n_ctx_train 262144, NO SWA.
Fork llama-cli is a thin in-process-server wrapper: `-v` is a boolean, log
level is `-lv N`.
llama-gguf.exe crashes 0xC0000005 pre-output on the ternary gguf -
tool-specific; do not trust llama-gguf here.
Actions (10-19):
- This section is the never-mix list: any upstream patch that touches
  GGML_TYPE_* values, ggml-quants.c block layouts, or server code that
  assumes the upstream type table is a NO-GO until re-audited against the
  enum table above.
- Prefer per-feature merges over a whole-history rebase: the fork is far
  behind, and a bulk rebase would drag in exactly the enum/layout changes
  above.
- After every upstream sync: re-verify the enum table + the MTP draft
  quant list, and re-run the ternary E2E repro (N7) before trusting the
  result.

## F13. Pre-existing / environmental issues (do not chase, note in report)

- test-dspark-forward LNK2019: full builds exit 1 on this pre-existing
  link error; harmless, ignore unless user asks.
- `W resolve_fused_ops: layer 0 is assigned to device CPU but fused Gated
  Delta Net (chunked) is assigned to device SYCL0 ... not supported, set to
  disabled` - pre-existing, seen in Qwen (55463 run) and ternary runs.
- ggml-sycl NULL-copy crash - noted earlier, not reproduced since.
- nvidia-smi exists but fails with a permission error (Intel box, irrelevant).
- This session's own local model is Qwen3.8-27B-UD-Q5_K_S (same family as
  the test model) - never confuse session logs with service logs.

## F14. Ternary (PTQ1_0) SYCL kernel gap vs CUDA - root cause located

[certain] The 52510 service tg trend (6.14 t/s at 62.4k ctx tokens ->
9.24 t/s at 30.0k) is KV-size driven, not warmup: the effective read
bandwidth comes out ~constant for both tasks. Decode reads weights + KV
once per step; the KV read grows with context.
[certain] Weight-byte math: PTQ1_0 ~1.75 bpw ~ 5.9 GB of 27B quantized
weights (+ F16/F32 norms/embd/head). Q6_K ~6.6 bpw ~ 22 GB. If both were
bandwidth-bound, PTQ1_0 should be ~3-4x FASTER than Q6_K, not slower.
[certain] Observed: Q6_K 13-18 t/s (user, 10-19) = ~37-52% of the 768 GB/s
peak; PTQ1_0 6-9 t/s = ~8-15%. The PTQ1_0 kernel is ALU/latency bound,
not bandwidth bound.
[certain] Kernel diff, per 128-weight block per output column (read both,
10-19):
- CUDA (ggml-cuda/vecdotq.cuh:808 vec_dot_ptq1_0_q8_1_multi): byte-
  parallel carry-safe trit decode (4 lanes packed in one int32: x3 +
  0x00FF00FF mask, 5 iters/byte, vsub4 -> {-1,0,+1}) fused into dp4a
  accumulation (32 dp4a per block). No local trit array.
- SYCL (ggml-sycl/vecdotq.hpp:1068 vec_dot_ptq1_0_q8_1): scalar base-3
  decode per trit (pow3 table, uint8*3, uint16*3>>8) materialized into a
  local int8_t q[128] (128 B/thread, spill-prone), then 128 scalar int32
  MACs against scalar-loaded q8_1 bytes (128 byte loads, no dp4a).
- Multi-column: CUDA decodes the x block ONCE and dots up to 3 y columns
  (ggml-cuda/mmvq.cu:719-742); SYCL mul_mat_vec_q_ncols (mmvq.cpp:186)
  re-runs the full vec_dot, including the whole decode, per column.
- PQ2_0: SYCL already uses dp4a; only gap is the 2-bit->int8 expand
  (shifts+ORs vs CUDA __byte_perm) - small.
- Not applicable to B70: CUDA GB10 prefetch path (mmvq.cu:670-710,
  GB10-only) and Hopper sm_90a wgmma prefill (3090 is sm_86).
[high] The 3090 CUDA ternary number (50-80 t/s, user memory) is the same
generic MMVQ + the fast vec_dot; 768 vs 936 GB/s peak cannot explain a
7-10x gap, so the SYCL vec_dot rewrite is the expected fix, not a
hardware limitation.
Fix sketch (plan, not written): port the CUDA _multi approach into
vecdotq.hpp - packed-lane x3/mask trit decode + dpct::dp4a, drop the
local q[128]; add a multi-column variant and special-case PTQ1_0 in
mul_mat_vec_q_ncols (mirror mmvq.cu:719). Verify: standalone pp/tg
before/after (N7) + existing host unit test. See B11.
