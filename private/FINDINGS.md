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
F32=0, Q4_0=2, Q5_1=7, Q8_0=8, Q3_K=11, Q4_K=12, Q6_K=14, PQ2_0=142.
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

RESOLVED 10-20 (B11, commit 5b80a3bb1):
[certain] The sketch was ported as written:
- vecdotq.hpp:1075 vec_dot_ptq1_0_q8_1_multi<ncols_dst>: byte-parallel
  carry-safe trit decode of the packed 16-bit lanes (x3 + 0x00FF00FF
  mask, no cross-lane carry) fused into dpct::dp4a; x decoded ONCE per
  block for up to 3 y columns; local q[128] gone. :1175 vec_dot_ptq1_0_
  q8_1 = wrapper to _multi<1>.
- mmvq.cpp:1423 fused kernel (row check, i stride WARP_SIZE, per-column
  xor reduce, literal iqs 0) + :1481 launcher; :1523-1537 switch:
  cases 2-3 fused, 4-8 generic, 1 single (mirrors mmvq.cu:719-742).
[certain] Result, standalone (no service pressure, ngl 99, t 56, b/ub
  6144, ctk/ctv q8_0, fa on): tg128 6.90 -> 26.72 t/s (3.9x); pp512
  296.39 -> 293.75 (unchanged, prefill path untouched). New tg128 is
  ~1.7-2x the Q6_K 13-18 t/s reference with ~3.8x fewer weight bytes.
[certain] Verification: host dot test T1-T5 (1029385 checks) + element
  map (2560000 positions) + cuda-dot exact int checks: 0 mismatches.
  Deterministic SYCL vs CPU 192-token completion: TEXT IDENTICAL,
  answers correct (temp 0, seed 42).
Debugging note (do not re-derive):
[certain] The early T1/T2 host-test failures were a TEST-ONLY store-
  index bug in the test kernel_decode: out[120 + t + l] with t in {0,2}
  made the t=2 group write 122..125 (overwrote 122/123, never wrote
  126/127) - fixed to out[120 + 2*t + l] in test-ptq1_0-sycl-dot.cpp.
  The DEVICE KERNEL WAS CORRECT FROM THE START: it stores nothing
  (dp4a accumulation), which is exactly why T3/T4/T5 passed while T1
  failed. The stale-exe hypothesis was DISPROVEN - the binary was always
  faithful to the source.
[certain] sim_qh.py (Python emulation) is BUGGY (unbounded shifts,
  broken sign-fill, print mismatch); use the compiled C sim
  (sim_qh.cpp, icx) for C-arithmetic emulation of the trit decode.

## F15. 52416 user depth curve: tg tracks ctx, "9.4x collapse" CLOSED

[high] (Entry reconstructed 10-22 from STATE.md refs + 65463/54447 log
blocks; the original write-up was lost in a compaction.)
- 54447 (UD-Q5_K_S + MTP) tg 1.47-1.50 looked like a 4.5-9.4x collapse
  vs 65463's 6.31 at 71.3K ctx. WRONG framing: 54447 had fused GDN
  disabled (F16), and the comparison confounded small-ctx vs long-ctx
  operating points.
- 65463 raw curve: tg 6.31 rock-stable at 71271 prompt tokens. The raw
  depth curve predicts ~6.6 at the 54447 depth with fused GDN enabled.
- B11 PTQ1_0 standalone (26.72 tg128) was verified at production ctx by
  the user depth curve (52416 class runs).

## F16. 54447 tg collapse: fit placed layer 0 on CPU mid-race, GDN
fused off for ALL 48 layers

[certain] 54447 spawned at router-clock 29.43 while 65463 was still
resident. fit placed layer 0 on CPU. resolve_fused_ops is all-or-
nothing: one CPU-placed layer disables the fused Gated Delta Net
(chunked) for ALL 48 linear-attention layers. Warnings at macube
llama.log :79831-79833 ("layer 0 is assigned to device CPU but fused
Gated Delta Net (chunked) is assigned to device SYCL0 ... not
supported, set to disabled"). Result: tg 1.47-1.50 (4.5x collapse).
Lesson: the F13 "pre-existing warning" is actually the F16 poison pill
- it is the SMOKE for a broken fit placement, not a harmless note.
User cancelled the 54447 task at 10.23.

## F17. Co-residency: NO room for a second 27B-class while 52416 runs

[certain] Fit probes 10-20 (52416 resident, draft-aware tool from
48e9082ca; logs $env:TEMP\fitprobeA/B/C.txt, 12 MB at -lv 5, grep
"printing fitted|^-c |free|MiB|target="):
- A = 52416 clone (Q5_K_S + mtp draft): projected 51523 MiB vs 10906
  free => fit gives -c 131072 -ngl 6 (90% CPU, unusable).
- B = PTQ1_0 plain q8_0 KV: -c 131072 -ngl 20 (~1-2 t/s class).
- C = PTQ1_0 --spec-type draft-mtp: -c 131072 -ngl 17 (same, bad).
- Big consumers at ctx 131072 / batch 6144: weights ~17.8 GB + KV ~3 GB
  + graph/sched compute reserve ~20-27 GB (comparable to the weights).
  Reserve scales with ctx x batch (65536 probe not yet run).
[certain] KV math CORRECTED (supersedes the old 6.27 GB / 9.7 GB
estimates): qwen35-27B has 16 full-attn layers (full_attention_interval
= 4 of 65 blocks). KV q5_1 @131072 = 3072 MiB (K 1536 + V 1536). Draft
KV q4_0 = 144 MiB. GDN recurrent state 598.50 MiB (64 layers, NO
per-token growth).
[high] Only a small model (4B-class ternary, ~1.3 GB weights) could
plausibly co-reside. Idle-unload is the only way to make room for a
second 27B (user accepted that as a future router option, 10-20).

## F18. MTP mechanics mapped (stub - detail lost in compaction)

[high] (Original detail lost; key retained facts, cross-referenced.)
- MTP = ONE nextn block (blk.64, nextn_predict_layers = 1), iterated
  for depth (mtp-Qwen3.8-27B-Q4_0.gguf, 18 tensors, 1306 MiB).
- Mode map (F5 Part 1): Qwen3.8-27B => is_mem_shared = FALSE (mirror KV
  active); only gemma4-assistant arch shares the target KV.
- Draft ctx: ctx_type MTP, n_ctx = llama_n_ctx(ctx_tgt),
  ctx_other = ctx_tgt (N11 speculative.cpp refs).
- MTP multiplier in service: ~2.0x raw decode (52416: tg 13.2-13.6 at
  73.6K ctx vs ~6.5 raw estimate).
- MTP under tensor mode (-sm tensor) WORKS (verified 10-22, Q5_K_S).
- [open] Full mechanics write-up (acceptance loops, chain_heads) was
  lost; re-derive from N11 code refs if needed.

## F19. Tensor mode (-sm tensor) facts

[certain] The fork has a 4th split mode, EXPERIMENTAL: one Meta device
wrapping the real backends; weights AND KV are split across devices;
per-subgraph all-reduce (vLLM-TP style).
- Run: `--split-mode tensor --fit off -ngl all -dev SYCL0,SYCL1`.
  An all-zero tensor_split = even split. `--fit off` is effectively
  mandatory (fit does not understand the meta device).
- MTP under tensor mode works. GDN output QUALITY in tensor mode is
  correct for the non-ternary Q5_K_S (user runs fine, 31.9 t/s at 160K
  ctx, 10-22).
- This finding set (F19-F25) targets making the ternary PTQ1_0/PQ2_0
  models tensor-mode compatible; the blocker is F20.

## F20. PQ2_0 tensor-mode assert - ROOT CAUSE (confirmed)

[certain] Crash: deterministic, warmup forward, block 0, the GDN
`build_lora_mm(ssm_out)` hadamard perm.
`GGML_ASSERT(split_state.ne[j] % div == 0)` at
ggml/src/ggml-backend-meta.cpp:1086.
- The failing node is the RESHAPE of the memo-MISS hadamard perm in
  build_lora_mm (llama-graph.cpp:1546-1603):
  `ggml_reshape_4d(x, 128, 16, 3, T)` on the GDN output (K = 6144).
- GDN output split state: {axis 0, ne {2048,3}, nr 1, n_seg 1},
  per-device extent 3072 rows (derived by the post-pass from srcs, see
  F21).
- handle_reshape (ggml-backend-meta.cpp:610-662) returns {axis 2, scale
  1}: the first dim where the cumulative OUT extent is a multiple of
  6144 is dim 2 (128*16 = 2048, x3 = 6144).
- The post-pass (:1066-1105) then computes div for that axis:
  3072 (per-device extent) x 3 (rep size) / 6144 = 1.5 => not integral:
  9216 % 6144 = 3072 => assert fires at :1086.
- T-INDEPENDENT (T cancels). UNREPRESENTABLE at ANY even split:
  dev0's row union = {(hd, nk in 0-7, rep 0-2)} = 1.5 rep-slabs of the
  6144-row axis; no segment table can express a half-slab.
Row/order math (both orderings, do not re-derive):
- TILED (GDN output / activation) row r: hd = r%128,
  nk = (r/128)%16, rep = r/2048, vhead = nk + 16*rep.
- GROUPED (weight K axis) row k: t = k/128, nk = t/3, rep = t%3.
- dev0 (eff rotation 0) holds vheads {0-7, 16-23, 32-39}, local order
  [v0-7 | v16-23 | v32-39]; dev1 = mirrored, +8 offset.
- H = W3 x W7 Kronecker (1024-row blocks = 8 vheads); H_local =
  P x H_bd x P^T, device-invariant (u = (3h+c) mod 8 is
  device-independent).
[certain] GDN kernel pairing is MODULO in BOTH backends (CPU ops.cpp
:10849, SYCL gated_delta_net.cpp:41) => the earlier "silent GDN
corruption" finding is VOID: no corruption, only the perm reshape is
unrepresentable. Plain {3072,1}/{2048,1} contiguous splits of the GDN
output are WRONG (they break the nk/rep interleaving). A rotation-at-
load of the weight alone is a NO-OP (it must travel with the perm +
signs to be meaningful).

## F21. Meta backend execution model (ggml-backend-meta.cpp)

[certain] init_tensor_impl (:1181-1299): creates a per-device local
tensor with GLOBAL ne but the split axis SCALED:
`ne[split_dim] = sum_s ne[s*n_bufs + j] * nr[s]`; nb rescaled
proportionally; the SAME op is enqueued on each device with srcs
rewired to local tensors. Consequence: every op must be
shape-consistent per device on the scaled tensor.
Split-state semantics (ggml-backend.h:378-415): axis + n_segments;
segment s = nr[s] reps, each rep split across devices, device j holds
ne[s*n_bufs + j]. Invariant: `sum_s (sum_j ne[s,j]) * nr[s] ==
ne[axis]`. `split_states_equal` (:498-505) compares axis + per-device
SUMS only. Axis enum: 0..3, MIRRORED = 10, PARTIAL = 11, NONE = 98,
UNKNOWN = 99. Hard cap n_segments <= 16 (ggml-backend-meta.cpp:1188).
- handle_gated_delta_net (:813-828): all-6-srcs-MIRRORED => MIRRORED
  out; else asserts srcs 0-4 axis 1, src5 axis in {0,1,2}, and returns
  a placeholder {axis 0, ne 0, nr 1, n_seg 1}; the real per-device
  extents (3072/dev, 2048x3 structure) are filled by the post-pass from
  srcs.
- calculate_split_state (:830-861): the model callback is consulted
  only for non-compute non-view tensors (:834).
- QWEN35 segment table (llama-model.cpp:540-560): ssm_out = {{2048,3}}
  (:541-542); split-state callback registered at :368.
- Assert + post-pass: :1066-1105 (div check :1085-1087).

## F22. H-B (runtime activation-side hadamard) is DEAD

[certain] Proof that no representable graph subgraph applies H on the
activation side in tensor mode:
- (a) Direct matmul H[3072,3072] MIRRORED x xs[6144,T]{axis 0}: ggml K-
  mismatch (3072 != 6144) + meta abort (MIRRORED src0 x axis-0 src1 is
  unsupported in handle_mul_mat :577-608).
- (b) Reshape round-trip [6144,T] -> [3072,2T] -> matmul -> [3072,2T]
  -> [6144,T]: the middle matmul IS representable (src1 axis 1 + src0
  MIRRORED => out {axis 1, T}; local [3072,T]; local compute =
  H_local x x_local = EXACTLY correct), but the FINAL reshape fails:
  T=1 => meta local shape [1536,1] != real [3072,1] => downstream
  split_states_equal assert; T>1 => post-pass assert (ne = T/2 vs div
  = 2T).
- No other form works: the only ops representable on xs[6144,T]{axis 0}
  are elementwise ops with matching states and TP K-reduction matmuls.
- H-C (fuse H into the kernel) = SYCL + CPU kernel rewrite;
  rejected as too invasive for the gain. Now documented in full as the
  fallback option: F29.
Context: the SYCL FWHT fast path (ggml_sycl_op_fwht, fwht.cpp/.hpp)
only serves n in {64,128,256,512} (n = src ne[0]); the 1024-wide
hadamard falls through to generic dense matmul EVEN in single-device
mode. So H-B would have cost only ~1.5x the existing dense H matmul -
but it is dead regardless.

## F23. H-A (LEAD FIX, APPROVED by user 10-22, IMPLEMENTING): load-time
weight fold W''

[certain] Design: tensor mode only; at load time
fold the ENTIRE transform (perm + signs + rot) into the ssm_out weight:

    W''[r,n] = s[p^-1(r)] * ( sum_k H[k, p^-1(r)] * W[k,n] )

r = tiled row, j = p^-1(r) = grouped row; H = 1024-block normalized
Sylvester (same as the on-spot rot, scale 1/sqrt(1024)); s = signs in
GROUPED order.
Implementation steps:
- Dequant PQ2_0 -> f32 (125.8 MB scratch per layer).
- FWHT butterfly per 1024-row H block: WHT-1024 = W8 x W128 (iterative
  butterfly), ~15.1 GFLOP over all 48 layers. NOT a dense GEMM (that
  would be ~3 TFLOP).
- Gather 128-row blocks grouped -> tiled: tiled vhead b <- grouped
  t-unit t_b = 3*(b%16) + b/16; the H block for it is t_b/8.
- Per-row sign flip in f32 BEFORE requant (the codec is asymmetric
  {-1,0,+1,+2} x d with NO -2 level; flipping in the quant domain is
  impossible).
- Requant with the EXISTING quantize_row_pq2_0_ref (F24).
- ggml_backend_tensor_set with the EXISTING {2048,3} segment table.
- Weight size UNCHANGED (401 MB total across 48 layers). Precision:
  one extra PQ2_0 pass (error ~ sqrt(2) x single pass).
- The graph then skips perm + signs + rot entirely: res = mul_mat(W'',
  x).
Loader hook design:
- llama_model_loader is a plain struct (llama-model-loader.h:31) with
  NO split-mode knowledge; load_all_data(ctx, bufs, lmlocks,
  progress_cb, ud) at :204.
- Add an optional pre-transform callback member:
  `std::function<bool(const ggml_tensor *, const llama_tensor_weight &)>`
  (return true = handled), set by the model BEFORE load_all_data.
  The load loop branch forces the read_buf path for target tensors and
  skips the normal set. No-op when the callback is unset (non-tensor
  modes untouched).
- Load-loop data paths (llama-model-loader.cpp:1551-1672): mmap-alloc
  :1582 (zero-copy, NOT taken - SYCL has no mmap_support so use_mmap =
  false via llama-model.cpp:1532-1541), mmap-set :1592, non-mmap host
  read :1599, non-mmap async chunked upload via
  ggml_backend_tensor_set_async in 1/64 MB pinned chunks :1607-1658
  (taken when device has async + host_buffer + events caps; SYCL likely
  qualifies), non-mmap plain read_buf -> set :1660-1663.
Gating (the hadamard builder runs AFTER ml.load_all_data,
llama-model.cpp:1940 -> builder :1951-2109, so the loader-side hook
cannot read the built transforms):
- Use hadamard_weight_blocks (name -> block_size map) +
  hadamard_gdn_v_grouped flag + split mode, all available before the
  load loop.
- Signs: ONE shared signs[6144] f32 tensor for all 48 ssm_out layers
  (keyed by (width, buft); hadamard_sign_data.find(width)); rot: one
  rot[1024,1024] f32 per (block_size, buft), computed on spot.
  Transform emplaced per weight :2100.
Graph gate:
- build_lora_mm (llama-graph.cpp:1546-1603): memo key (cur, t.rot)
  :1556; perm gate t.perm_rep > 1 :1561 (reshape_4d -> cont(permute
  (0,2,1,3)) -> reshape back :1563-1567); signs mul :1569-1571;
  llama_mul_mat_hadamard :1572; memo store :1573; ggml_mul_mat(w,
  cur_mm) :1578.
- Add `bool folded = false;` to llama_hadamard_transform
  (llama-graph.h:23-32); the builder sets it in tensor mode;
  build_lora_mm skips the whole transform block when folded (cur_mm =
  cur). Signs/rot tensors still get created in tensor mode (harmless:
  24.5 KB + 4 MB mirrored) - only their graph use is skipped.
- split_states_equal HARDENING: DROPPED from scope - with H-A both
  matmul srcs carry identical states {axis 0, 2048x3, 1024/1024}; keep
  the change minimal.
- If the requant noise ever proves unacceptable, the kernel-level
  fallback (H-C) is documented in F29.

## F24. PQ2_0 / PTQ1_0 codec facts (committed knowledge)

[certain] PQ2_0 codec: 2-bit codes {00=-1, 01=0, 10=+1, 11=+2};
d = amax per 128-group stored as fp16; quant `q = round(w/d) + 1`
clamped to [0,3]; quant runs along ne[0] (K axis, contiguous blocks,
layout data[r + 6144*n] for ssm_out).
- Full existing pair: quantize_row_pq2_0_ref (ggml-quants.c:113),
  dequantize_row_pq2_0 (:494), quantize_pq2_0 (:2183), cpu wrapper
  (ggml-cpu/quants.c:33). H-A needs NO new quant code.
- block_pq2_0 (ggml-common.h:202-207): QK = 128, ggml_half d +
  uint8_t qs[32] = 34 B per block. block_ptq1_0 at :214-220.
- ssm_out weight [K=6144, N=5120] PQ2_0: 245760 blocks x 34 B =
  8.36 MB per layer, 401 MB total.
- Fork ggml type enum: PQ2_0 = 142 (extends the F12 table; PTQ1_0 is
  the separate ternary type used by the PTQ1_0 family).

## F25. Port-target clone located (the "min llama.cpp fork" question)

[certain] `C:\Users\admin\source\ai\llama.cpp` exists: a CLONE of
UPSTREAM ggml-org/llama.cpp (remote origin, branch master, HEAD
6fcaa16f4 - recent upstream). It is NOT a 3nary-derivative and NOT the
PrismML-Eng remote (that lives in the 3nary repo as remote "prism").
- Contents: full upstream tree + build/ + build-gtx/, user scripts
  (build-sycl.ps1, run-llama-sycl.ps1/.bat/.sh, build-gtx.ps1,
  build-nv.ps1, build-vulkan.ps1, disable-vulkan-validation.ps1,
  PORTABLE-SETUP.md), session notes (session-ses_f2a3_MTP_vs_
  concurrency.md, session-ses_f381.md, session-ses_f458.md).
- The user proposed (10-22) to work directly on this clone as the port
  target ("keep the copy for us and create a true fork" later if
  needed).
[certain] Git state verified 10-22: branch master, HEAD 6fcaa16f4
(2026-09-21, recent upstream). NO local user commits - the "[origin/
master: ahead 53]" marker is a STALE local fetch ref, not local work.
Working tree HAS 3 uncommitted tracked-file modifications from earlier
sessions (DO NOT OVERWRITE, F25 user deltas):
- common/chat.cpp: tool-call args JSON parse failure downgraded from
  throw to LOG_WRN (keep string).
- scripts/ui-assets.cmake: npm.ps1 added to the npm find_program list.
- tools/server/server-context.cpp (~:3444): recurrent-memory (hybrid/
  GDN) seq_rm refusal handling - if rollback fails, clear the sequence
  and re-process the full prompt.
[open] The port delta is UNMEASURED: the clone is recent UPSTREAM while
the 3nary work sits on a fork far behind upstream (F12). Upstream has
its own meta / tensor-split path (check its ggml-backend-meta.cpp) that
DIVERGED from the 3nary one. Before any porting: diff upstream meta
backend vs 3nary meta backend, confirm upstream type enum values for
PQ2_0/PTQ1_0 (upstream has ternary types; numeric values likely differ
from the 3nary fork - F12 never-mix rule), and map which 3nary work
(ternary kernels, GDN hadamard builder, prism hadamard, tensor-mode
options) is missing or divergent.

## F26. :537 UNKNOWN-axis assert - RESOLVED (10-23)

[certain] The `ggml-backend-meta.cpp:537` UNKNOWN-axis assert hit after
B12 (first failing node: blk.0 ffn_down signs-MUL) is PRE-EXISTING, not a
fold regression: tensor mode cannot represent axis-split activations whose
K extent is not a whole number of Hadamard blocks on each device, and
ffn_down K=17408 = 8.5 x 1024 is one.
[certain] FULL 401-weight audit (GGUF metadata dump, prism_names.py):
exactly THREE weight families receive axis-split activations in tensor
mode; all other 273 rotations take MIRRORED activations (runtime transform
fine):
- ssm_out x48  [K=6144, N=5120], split {1024x3, 1024x3} -> folded (B12).
- ffn_down x64 [N=5120, K=17408], split {8704, 8704} on axis 0 = 8.5
  blocks per device -> :537 assert (blk.0, first node after ssm_out).
- attn_output x16 [5120, 5120] -> LATENT, same assert at blk.3 (the first
  full-attn block; no tensor run ever reached it before).
Break mechanics (attn_output, dump+code verified): flash-attn op q is 4D
[n_embd_head, T, n_head, n_stream] (build_attn_mha permute 0,2,1,3, heads
axis 2); handle_flash_attn_ext (:773) returns axis 1 for the 4D out;
reshape_2d (llama-graph.cpp:2712) to [5120, T] gives axis-0 {2560, 2560};
the sigmoid gate is a view of the QG-joint Qcur_full [10240, T]
(interleaved head slots of 320 rows: q 160 + gate 160) so it carries the
same split; the gated MUL keeps it. 2560 = 2.5 blocks.
[certain] Option B (re-placement) is DEAD for attn_output: 5120 = 5 whole
blocks cannot be split into whole-block counts across 2 devices (2.5
each). Option A (meta-backend rules / sliced views) is dead at the
rotation reshape step (F22). Folding (option C) is the ONLY viable shape.
Fold formula for the new families (no perm): the graph transform is
T(x) = H (s . x) (build_lora_mm order verified: perm-if-ssm_out -> signs
MUL -> rotation), so W'' = W H diag(s): per output row dequant -> in-place
FWHT per 1024 block -> multiply signs[k] (global K index) -> requant.
Weight-domain order = FWHT THEN signs (transpose of activation-domain
signs-then-H).
Scope decision (user-approved 10-23): fold ONLY ssm_out + ffn_down +
attn_output = 128 weights. Do NOT fold the other 273: their activations
are MIRRORED, and folding all would add requant noise to every FFN/proj
weight for zero gain.
Total folds in tensor mode after B12 fix + extension: 128 (48 + 64 + 16).
Expected SPLIT_STATE on folded leaves: ssm_out {1024x3, 1024x3} axis 0,
ffn_down {8704x1, 8704x1} axis 0, attn_output {2560x1, 2560x1} axis 0.
(Exit criterion CORRECTION: ssm_out is {1024x3}, not {2048x3} as F23/
STATE said.)

## F27. ssm_out fold gather bug (PROVEN) + B12 fix (10-23)

[certain] The B12 (H-A) fold as first implemented had a PROVEN bug: the
pre-gather `gcol[l] = col[perm_t[l]]` is a SPURIOUS permutation of the
weight rows. W K-storage is GROUPED (F20 row/order math + kernel evidence:
gated_delta_net.cpp:41 `iq1 = fastmodulo(h_idx, neqk1)` pairs v head i
with k head i%16; kernel output written at `h + 128*h_idx` = TILED
order), so the dequantized row is already in grouped order and must NOT
be permuted before the FWHT.
Proof (numeric mini-model, $env:TEMP\opencode\fold_order_test.py;
n_hd=4 n_k=2 n_rep=4 K=32 block 8, true Sylvester H, cross-block perm):
- code as implemented (gather p^-1 + scatter p): max err 5.58 WRONG
- F23 design-doc formula (no gather, scatter p^-1): err 4.62 WRONG
- FIXED (no gather, per-block FWHT, signs, scatter p = perm_g): EXACT
  (diff 0.000000) vs required W'' = W H diag(s) P.
The bug was LATENT: every tensor run died at :537 (blk.0 ffn_down) before
any token, so the wrong 48 ssm_out folds never produced output; and the
verifier (llama-context.cpp:36-120) is topology-only (folded = no
activation transform), it cannot catch a numeric fold error. perm p is
NOT self-inverse (p(128)=384, p(384)=1152 in the real 128/16/3 geometry),
so "swap the gather direction" was also wrong - the gather must be DELETED.
Fixed formula (implemented in fold_hadamard_weight, llama-model.cpp:1519):
per row: dequant (grouped) -> in-place FWHT per 1024 block -> multiply
signs (grouped index) -> ssm_out only: out[r] = col[perm_g[r]]; requant.
ffn_down/attn_output: same pipeline minus the scatter (identity).
Rotation-tensor build (load_tensors :2158-2171, popcount parity x
1/sqrt(block_size)) verified == the fold's FWHT matrix exactly.
B12 verification (10-23, PQ2_0 27B, -sm tensor, -c 4096): 128 fold lines
(48 perm + 80 signs), "loaded 402 Hadamard-folded weight(s) (1
inverse-lookup) using 1 rotation(s) and 3 sign vector(s)", SPLIT_STATE
ssm_out {1024x3, 1024x3} / ffn_down {8704x1, 8704x1}, NO :537 assert,
prompt eval 64 tok 20.68 t/s, 16+ sane reasoning tokens, full-attn blocks
3,7,... first exercised (attn_output folds fire on blk.3/7).
Formula pinned by unit test tests/test-hadamard-fold.cpp (added 10-23,
user-approved): transcribed FWHT/signs/scatter pipeline checked against
an independent dense Sylvester reference (butterfly == dense at n=4/8/
16/1024; pipeline == W H diag(s) at small scale and == W H diag(s) P at
the real 6144/1024 geometry, exact 0.0; perm bijection + spot values +
not-self-inverse). Self-contained host test (pattern:
test-ptq1_0-element-map.cpp), no model load. Scope note: it pins the
formula contract, not the 20-line function body (private member, no
public hook); the body is covered by the live B12 runs + this mini-model.
PTQ1_0 tensor 128-tok: Paris found in thinking + temp-0.8 repeat loop;
`-sm layer` baseline clean Paris. User 10-23: production sampling is
temp 1.0 and never loops -> loop = sampling artifact, fold exonerated,
greedy isolation dropped.

## F28. Live service (CLONE build) state restore failure, 88K tok (OPEN)

[correction 10-27] The "CLONE build" part is STALE: the live
child now spawns the INSTALL-dir exe (C:\Users\admin\.local\
3nary-sycl\llama-server.exe, port 56762, spawn log :14435) - the
compile-time path in the 10-23 error line did not match the spawn
exe. Still open: whether the bad-allocation state-save failure
mode exists in the 3nary build (fold into the deferred bad-
allocation analysis).

[observed] 10-23, user pasted live-service log: the macube service is now
deployed from the CLONE tree `C:\Users\admin\source\ai\llama.cpp`
(compile-time path in the error line; old child PID 62743, new child
50630, LCP lines from 59148). It runs Qwen3.8-27B-UD-Q5_K_S in TENSOR
mode (fit warning "llama_params_fit is not implemented for
SPLIT_MODE_TENSOR" at child boot - so the clone HAS the tensor mode enum
+ meta backend: de-risks B13, but recon must confirm scope).
Failure sequence (slot 3, task 15109, n_tokens = 88212):
`E slot iterate: got exception: bad allocation` -> `state_seq_set_data:
error loading state: bad allocation` -> `common/common.cpp:2350:
checkpoint size mismatch: expected 102694248, got 0`. 102694248 B =
97.94 MiB. This is the sequence-state (recurrent memory) save/restore
path - the same area as the user's clone fix in
tools/server/server-context.cpp (~:3444, seq_rm refusal: on rollback
failure clear the sequence and re-process the full prompt).
[certain] Interpretation: a state SAVE must have failed (bad allocation)
and left a 0-byte checkpoint; the subsequent restore then hit the size
mismatch. Root cause of the save failure NOT yet pinned: candidates are
device-side allocation pressure at 88K tokens (KV + GDN state + a ~98 MiB
state blob copy) or a code path bug in state save/restore. The child
recovered (50630 relaunched, model loaded, serving).
[constraint] Do not disturb the service. Investigation is read-only
(macube logs: C:\Users\gorom\source\macube\logs\llama.log +
llama-error.log, grep by PID/port) until the user says otherwise.
[open] Whether the same failure mode exists in the 3nary build (the
clone's server-context.cpp seq_rm fix is a USER MOD, F25 - the 3nary
fork may lack it or differ). Relevant to B13 porting scope.

## F29. H-C kernel-level fusion - DOCUMENTED FALLBACK (10-23, user
request: keep it documented in any case)

Escalation path for B12 if the fold's one extra requant pass on the 128
weights ever proves unacceptable (user clause 10-23: "if this proves
[defect] above acceptable we must look for other solution"). NOT
adopted now: B12 (H-A, 7cb134632) ships the load-time fold and is
verified. H-C removes the requant entirely: the 128 weights stay
verbatim from the GGUF; the activation transform runs at runtime, fused
inside the ternary matmul kernel.

Problem shape (why this is possible now, and what it must not break):
- The meta backend only ever failed to place the STANDALONE signs +
  hadamard ops on axis-split activations (F22: an axis-split tensor can
  only flow through elementwise + K-reduction matmul; block-mixing is
  neither). A plain matmul on the RAW activation IS placeable - the
  folded B12 graph proves it (same shapes, same splits).
- So H-C keeps the exact same graph as B12: res = mul_mat(W, x_raw).
  No new placement rules, no meta backend change. All of the work is in
  the SYCL ternary dot kernel + how it learns a given weight is
  rotated.

Design:
1. Graph: build_lora_mm (llama-graph.cpp:1546-1603) skips the transform
   block exactly as it does today for folded weights (B12 `folded`
   flag). The only difference: the weight is NOT pre-transformed at
   load (no dequant -> FWHT -> signs -> scatter -> requant), and a
   "rotated" marker (the B12 `folded` set, renamed/repurposed) is
   propagated to the op.
2. Op/kernels: the matmul op must carry a rotation attribute (1024-block
   FWHT + signs + optional ssm_out scatter) so backends apply it.
   Representation options, in increasing invasiveness:
   (a) a ggml op attribute / extra src on ggml_op_mul_mat set by the
       graph builder;
   (b) a thin dedicated op (e.g. ggml_op_mul_mat_rotated) that
       ggml-backend-meta.cpp maps 1:1 to the mul_mat placement rules
       (one new case, same split-state outputs: N-split -> axis-0 out,
       K-split -> mirrored out with the existing cross-device
       reduction).
   (a) is smaller; (b) is cleaner if the rotation ever grows more
   parameters.
3. SYCL kernel (the actual work): inside the ternary dot kernel (the
   B11 byte-parallel PQ2_0/PTQ1_0 x q8_1 family), per 1024-block of the
   K (activation) axis:
   - cooperatively load s * x_block (1024 f32 = 4 KB local memory);
   - in-place FWHT (10 stages, n*log2(n) = 10240 ops);
   - dot the transformed block against the (dequantized) W row chunk.
   The transform is amortized per work-group over all N rows the group
   covers - NOT re-run per row. The ssm_out scatter (grouped <-> tiled)
   is applied to the x_block indices at load time (cheap index gather,
   per 128-row group), ffn_down/attn_output use the identity.
4. Straddling blocks under an EVEN split (the one real hazard - this is
   the F22/F26 geometry): the in-kernel transform needs the FULL 1024
   block, but under the current even split ffn_down devices hold 8704
   (8.5 blocks) and attn_output 2560 (2.5 blocks) - one block straddles
   the device boundary and needs 512 elements from the other device.
   Two mitigations:
   (i)  PREFERRED: block-aware UNEven split - ffn_down 8192/9216 (8/9
        blocks), attn_output 2048/3072 (2/3), ssm_out stays 3072/3072.
        Whole blocks per device, ZERO cross-device traffic. Cost: 12.5%
        / 20% load imbalance on those weights only (4.2 vs 4.7 MiB
        f32-equiv for ffn_down; trivial), plus a placement change in
        llama-model.cpp (segment computation must emit uneven
        block-aligned segments; SPLIT_STATE already supports arbitrary
        per-device ne). NOTE: this also revives F26 option B in the
        form actually needed - F26 only ruled out resplitting an EVEN
        split (5 whole blocks cannot stay 50/50 as whole blocks); an
        odd 2/3 split was never considered.
   (ii) keep the even split + a boundary-block exchange: each device
        fetches the other's 512 x T f32 for the straddling block
        (2 KB at T=1 per layer) and transforms the full block locally.
        Tiny volume, but per-token P2P latency + fence complexity on
        every ffn_down/attn_output call. Rejected unless (i) proves
        problematic.
5. Cost estimate (runtime, per token, transform only): 10 flops/element
   (log2 1024): ffn_down 17408 x 10 x 64 layers = ~11.1 M, attn_output
   5120 x 10 x 16 = ~0.8 M, ssm_out 6144 x 10 x 48 = ~3.0 M; total
   ~15 MFLOP/token - comparable to or smaller than the dequant + MAC
   work the same kernels already do; expected to hide behind the
   existing memory bandwidth, to be MEASURED (tg + t/s vs the B12 fold
   run, same prompt, temp 1.0 for quality comparison).
6. Trade vs B12 (the decision axis):
   - B12 (shipped): +1 requant pass on 128/401 weights (PQ2_0/PTQ1_0
     error ~ sqrt(2) x single pass, F23); zero runtime cost; zero
     kernel/graph changes beyond the flag.
   - H-C: zero added quantization noise (weights verbatim); cost =
     kernel work above + graph op plumbing + (i) placement change.
   - Decision criterion (user 10-23): adopt H-C only if the B12
     fold's requant noise shows up in real workloads (temp 1.0,
     long-context, the user's traffic). Quality A/B: same prompts,
     -sm tensor (B12) vs the H-C build vs -sm layer baseline; judge
     task accuracy, not token identity (temp 1.0 is non-deterministic).
7. Scope estimate if ever started: graph flag propagation (small), op
   attribute or thin op + meta case (small), SYCL kernel transform
   (the bulk - one kernel family, 3 weight families), placement
   uneven-split (small-medium), CPU/other-backend fallback = REFUSE
   rotated matmul on non-SYCL backends in tensor mode (fail loud, like
   B7a) since only the SYCL ternary kernel implements the transform.
8. Port note (B13): the clone's SYCL ternary kernels diverge from the
   3nary fork's (B11 byte-parallel rework is fork-only, F25). H-C would
   be designed per-tree, not ported.

## F30. "SYCL=OFF OOM bypass" = CPU-only runtime, no visible error
   (root cause of the 10-25 "no usable GPU"; FIXED)

The 10-21/10-23 build state ran with -DGGML_SYCL=OFF as a bypass for
commit-headroom OOMs during the heavy SYCL TU compile. Consequence found
10-25 while verifying the fit grow work: the runtime had NO SYCL device
at all (fit reported nd == 0), with no error anywhere in any log.

[certain] Chain (all verified in source 10-25):
1. Entry points call ggml_backend_load_all() unconditionally (common/
   arg.cpp:253/:1125/:1142/:1168, src/llama.cpp:103).
2. The loader tries to dlopen [lib]ggml-sycl-.dll next to the exe
   (ggml/src/ggml-backend-reg.cpp:480-560, load_best; base-name
   fallback :543-557).
3. The dlopen contract needs ggml_backend_score (optional, :229-235)
   and ggml_backend_init (REQUIRED, :237-243). Those exports are only
   emitted with GGML_BACKEND_DL=ON (ggml/CMakeLists.txt:86, default
   OFF - and this fork's proven build config keeps it OFF).
4. Both the 10-23 build tree's ggml-sycl.dll and the 10-21 install
   tree's (C:\Users\admin\.local\3nary-sycl\) export only
   ggml_backend_sycl_reg + the SYCL API (verified 10-25: dumpbin
   export lists + P/Invoke probes). So a SYCL=OFF build can NEVER
   dlopen SYCL in this fork.
5. Every load failure is swallowed under NDEBUG: silent = true
   (ggml-backend-reg.cpp:567-571). Release builds log nothing.

Net effect: SYCL=OFF build = CPU-only runtime; -ngl 99 silently does
nothing; fit sees nd == 0. SYCL works ONLY via static registration in a
SYCL=ON build: the ggml target links ggml-sycl PUBLIC and the registry
ctor does #ifdef GGML_USE_SYCL register_backend(ggml_backend_sycl_reg())
(ggml-backend-reg.cpp:126-128).

Fix (DONE 10-25): one-time reconfigure -DGGML_SYCL=ON (GGML_BACKEND_DL
stays OFF = the proven 10-23 config), build ONLY the ggml-sycl target
with -j 1 (105/105 steps; needs ~24 GB commit headroom), then the
downstream rebuild (297/297). build-3nary.bat (N3) is permanently
switched to SYCL=ON with a comment; future builds are incremental (the
SYCL TU is already compiled, so only invalidation of it reruns the
heavy compile). build-sycl-tu.bat / build-downstream.bat in
$env:TEMP\opencode\ hold the one-time recipe.
[constraint] Any future redeploy from this tree must come from a
SYCL=ON build state. If commit headroom is ever short again, SYCL=OFF
may be used to finish the non-SYCL parts, but the deployable binary
must then be completed with the SYCL=ON recipe before it is used.

## F31. KV-exhaustion wall: no-crash at the wall + fit-time auto-size
   (BOTH COMMITTED 10-25)

Problem: a decode that could not allocate a KV slot after the half-batch
retries used to fail EVERY active slot and abort the whole server
context - one client hitting the -c 262144 wall took down the others.
Also: fit could only SHRINK to fit, never grow a pinned -c into the
free memory that is demonstrably available.

Stage 1 - no-crash (commit 6a30f9157, "server : end only the failing
slot when the KV cache is exhausted"):
- decode() signature bool -> int; file-local enum decode_result
  { DECODE_OK = 1, DECODE_RETRY = 0, DECODE_SKIP = -1 } in
  server-context.cpp; single caller = the batch loop (~:2770).
- Handles ONLY n_batch == 1 && ret == 1 (the KV wall). On n_batch == 1
  the failing token belongs to a single slot:
  1. try_clear_idle_slots(); if a slot is freed -> DECODE_RETRY (the
     same token, next iteration).
  2. else identify the failing slot by slot.id ==
     batch_view.seq_id[0][0] (llama_batch.seq_id is llama_seq_id **,
     one pointer per token; in the server seq id == slot.id), send_error
     EXCEED_CONTEXT_SIZE to that client, release() + prompt_clear() the
     slot, DECODE_SKIP.
- Result: that client gets a clean error; the other slots keep
  generating; the server keeps running. NO preemption of active clients
  in v1.

Stage 2 - fit-time auto-size (commit db5cf4d80, "fit : grow the context
into free memory up to an opt-in cap (-fitm)"):
- OPT-IN -fitm / --fit-ctx-max N (env LLAMA_ARG_FIT_CTX_MAX), field
  common_params.fit_params_max_ctx (0 = off), units = total ctx like -c.
  The grow block is a no-op when the flag is absent: the default fit
  behavior is unchanged.
- Grow block in common_params_fit_impl (common/fit.cpp :310-360):
  gate on n_ctx_max_user > 0 && nd > 0 && cparams->n_ctx > 0; cap =
  min(fitm, n_ctx_max); measure dmds_cap ONCE at the cap (save/restore
  n_ctx around common_get_device_memory_data_impl); ctx_rate[id] =
  (cap_ctx - cur_ctx) / (cap - cur), skip if <= 0; budget = free -
  mb.total() - margin; budget < 0 -> limit = 0, break (already over
  budget, no grow); else limit = min(limit, cur + budget / rate);
  grown = floor(min(limit, cap) to lcm(16384, n_streams)) - the 16K
  grow step; if grown > cur: set n_ctx and adjust
  dmds_full[id].mb.context += ctx_rate * (grown - cur) in place so the
  step-1 margin check applies to the grown context; sync
  extra->cparams->n_ctx afterwards (add_extra_memory had left it at the
  cap). fit.h contract doc extended.
- In-place KV ring growth is REJECTED (the KV pool is fixed at context
  creation) - auto-size happens at FIT time, before the context exists.

Verification (3 scenarios, 10-25, live service holding the cards; fit
loads with no_alloc + LLAMA_LOAD_MODE_NONE, safe alongside the service):
- (a) host-only, nd == 0: grow skipped, fit OK (fitm-test2.log).
- (b) GPU over budget, -c 131072 -ngl 99 -fitm 262144: budget < 0 ->
  no grow; fit aborts correctly (projected 29501 MiB > free 29008 MiB;
  pinned -c/-ngl reported; exit 1) (fitm-test3.log).
- (c) GPU GROW, -c 32768 -ngl 40 -fitm 65536: grew 32768 -> 65536,
  margin check passed on BOTH devices (S0 free 7132 / S1 free 6040 vs
  target 1024 MiB), printed -c 65536 -ngl 40, exit 0 (fitm-grow.log +
  fitm-grow.bat). Grow path proven on-device.
- Fit math reference (10-25, service holding the cards): at -ngl 40 /
  -c 32768 / -np 2: S0 free 13759, self 6751 (model 4742 + ctx 1629 +
  compute 379); S1 free 15749, self 8582. At -ngl 99 / -c 131072 /
  -np 2: S0 ctx 7766, S1 ctx 9814, projected 29501 vs free 29008.
- Relation to the user's extraArgs bisection (pending user report):
  preset tail -ngl 99 -np 2 -c 262144; predicted KV per the 10-25
  table: full tail 2 x 131072 = 6144 MiB; minus -c: 2 x 81920 =
  3840 MiB; all removed: 1 x 163840 = 3840 MiB. Fold the outcome into
  this finding when the user reports it.

## F32. Tensor-mode ctx fit: real-load measurement (SHIPPED 10-25)

Problem: common_params_fit_impl could not fit SPLIT_MODE_TENSOR.
The ngl loop is meaningless there (the weight split is fixed by the
model, decided at allocation time) and the projection helpers assume
per-device layer placement. Tensor mode fell through to
warn-then-continue (the old B7a target site, fit.cpp:182-183), so a
tensor-mode run with a pinned -c that does not fit failed late and
opaquely.

Design (approved 10-25; user constraint: NO alloc probes - real
context creations only, no graph build):
- common_params_fit_tensor_impl (fit.cpp :214-434), dispatched from
  common_params_fit_impl (:440-443) when split_mode ==
  SPLIT_MODE_TENSOR; a literal sibling of common_params_fit_impl
  (the -fit family naming precedent).
- Real model load with mparams as given (the tensor split is only
  known at allocation time); per-card free memory before/after the
  load = used_model.
- Two real context creations, create+free, at n_ctx 2048 and 32768
  (tensor_measure_ctx, :192-212); per-card free-memory deltas
  (used_lo, used_hi). If the high point does not fit, halve it; if
  even the low point fails, the model+min ctx do not fit (throw);
  if only the low point fits, binary-search the max instead of the
  linear solve.
- Per-card linear solve: rate = (used_hi - used_lo)/(hi - lo)
  bytes/token, fixed = used_lo - rate*lo; budget = free0 -
  used_model - fixed - margin; n_ctx_fits = min over cards of
  budget/rate (0 when a card is over budget at the minimum context).
- n_ctx is TOTAL ctx (per-seq = n_ctx / n_seq_max); n_ctx_max =
  n_ctx_train * n_streams (total cap, n_streams = 1 for unified KV,
  else max(1, n_seq_max)).
- Alignment: floor to lcm(16384, n_streams), EXCEPT when the cap is
  the limiting bound (cap == n_ctx_max keeps the trained cap EXACT,
  n_ctx_from_cap :392-398).
- -c 0 sentinel: n_ctx_min == UINT32_MAX means full context
  requested; effective min = n_ctx_max (no reduction allowed).
- Explicit -c above n_ctx_fits: fit_exception with the max fitting
  printed (the client sees "does not fit, max fitting is N").
- -fitm (n_ctx_max_user): grow a pinned -c to
  min(n_ctx_fits, n_ctx_max_user, n_ctx_max) with the same
  alignment, only when it grows (n_ctx_orig saved).
- Extra (draft/MTP) model in tensor mode: throw (unsupported).
- Margins: margins_s are ordered by the meta device's card order
  (explicit -dev list, or the default non-CPU selection), matched
  via the three new meta accessors.

Dependency (same commit): ggml_backend_dev_is_meta /
ggml_backend_meta_dev_n_devs / ggml_backend_meta_dev_simple_dev
were file-static in ggml-backend-meta.cpp; de-static + declared in
ggml-backend.h (:407-409) so fit.cpp can map the meta device back
to its cards. The model's single meta device comes from
llama_model_n_devices/llama_model_get_device (llama-ext.h:88-90,
llama-model.cpp:3514-3523 - in-tree, NOT in include/llama.h).

Verification (27B ternary PTQ1_0 5.54 GB, q5_1 KV, -sm tensor
-ngl 99, 2x Arc Pro B70 32 GB, co-resident with the live service;
runner + logs in $env:TEMP\opencode, fit-verify-*.log):

  A-auto (no -c):        exit 0, -c 262144 -ngl 99
                         (max fitting 910054, trained cap exact)
  A2-auto -np 2:         exit 0, -c 524288 -ngl 99
                         (max fitting 903787; 2 x 262144 per-seq;
                         cap == n_ctx_max so exact, no alignment)
  B-explicit -c 1000000: exit 1, "the user context size of 1000000
                         does not fit, max fitting is 909759"
  C-full-ctx -c 0:       exit 0, -c 262144 -ngl 99
                         (sentinel path; max fitting 909508)
  D-grow -c 8192 -fitm
  65536:                 exit 0, "growing the context size from
                         8192 to 65536", -c 65536 -ngl 99

Per-card trace (A): model 2707 MiB/card; free before/after 14577/
11870 (SYCL0), 16234/13527 (SYCL1); ctx@2048 = 164 MiB/card;
fit wall time ~97 s (one model load + two context creations).

Findings:
1. The KV cache does NOT scale with n_seq_max in this fork: 1 seq
   32768 = 768 MiB (32768 cells, 16 layers q5_1, 1/1 seqs) vs 2 seqs
   32768 = 768 MiB (16384 cells, 2/2 seqs) - the same shared n_ctx
   ring (the pool F31's no-crash work addresses). Only the GDN
   recurrent state grows per seq (149.62 -> 299.25 MiB, R+S f32).
   This is why max fitting moved only 910054 -> 903787 between A
   and A2.
2. hadamard-v2 loader gate: 10/20 ternary re-exports (2B/4B/9B +
   MTP) carry prism.hadamard.version = 2 and are rejected by this
   tree (llama-model.cpp:1198-1201 accepts v1 only); only the 9/20
   27B files (PTQ1_0, PQ2_0) load. The 27B was therefore the
   verification vehicle; 4B/9B tensor work is blocked on this gate.
3. Tensor mode force-enables flash_attn ("enabling flash_attn since
   it is required for SPLIT_MODE_TENSOR") - no --flash-attn needed.
4. Max fitting varies per run (903787-913441 observed) with the
   service's memory footprint at fit time - expected (per-card free
   at load), not a bug.
5. Co-residency: post-restart the 27B tensor fit ran fine alongside
   the live service (per-card free after model load ~11.6/13.5 GiB);
   pre-restart the same fit hit the 0xC0000409 host-malloc assert
   (F1/B4 family) under the older service footprint. Re-runs can
   fail if the service footprint grows; re-derive VRAM first.
6. Build quirk (local, documented, not chased): VS-bundled ninja
   1.12.1 exits 0 after the "Provisioning UI assets" custom target
   while ninja -n shows 6 pending edges, so the exes were not
   relinked after the helper rename. Not blocking: the 16:59 exes +
   18:28 llama-common.dll are ABI-compatible (the rename is a
   file-static function; fit.cpp lives in llama-common, not the
   -impl dlls). See STATE.md Binaries.

Commit: 1011b639d (fit.cpp +260, ggml-backend.h +4,
ggml-backend-meta.cpp 8 +/-), pushed 10-25.

## F33. Runtime context growth (design CONFIRMED 10-25, implementing)

Goal: when a slot hits the KV wall (F31 stage 1 ends it) and the
cards have headroom, grow the context IN PLACE instead of ending
the slot. Runtime counterpart of -fitm (F31 stage 2, fit-time): the
model stays loaded, no restart, no slot loss, no output reset.

CORRECTION 10-25 (pivot, verified in tree): the original F33 draft
assumed PER-SLOT contexts (slot.ctx_tgt/ctx_dft, "recreate one
slot's ctx", "the other slots keep decoding"). That premise is
WRONG:
- ONE shared ctx per server: ctx_tgt/ctx_dft are created exactly
  once in load_model (server-context.cpp:1057-1060) via
  common_init_result; slots hold only pointers (server-context.
  cpp:203-204). n_ctx = llama_n_ctx(ctx_tgt) (:1074) is the TOTAL;
  n_ctx_slot = llama_n_ctx_seq(ctx_tgt) (:1170, capped at
  n_ctx_train) is the per-slot share.
- "Other slots keep decoding" during a swap is UNACHIEVABLE: all
  slots decode through the same shared ctx. All active slots pause
  briefly (sub-second) during the regrow.

The recreate design (save host state, free ctx, re-create,
restore) is SUPERSEDED by the design below.

FINAL DESIGN (user agreed 10-25): in-place tail extension of the
shared ctx ring, via per-layer data-tensor regrow. No shift, no
re-pointing, no host save, no ctx recreation. The ring just gets
more cells at the tail; everything else keeps the same object
identity.

Why in-place is safe (all verified in tree):
- Bookkeeping: llama_kv_cells.used is a std::set of cell indices
  (llama-kv-cells.h:462); find_slot (llama-kv-cache.cpp:899-1096)
  scans for empty cells, so holes are tolerated; cell indices stay
  stable under all normal ops (shifts are pos-based; K-shift
  re-encodes in place). Appending empty cells at the tail is a
  valid extension; no existing data moves.
- Data side: K/V tensors are per-buft today (one ggml ctx + one
  buffer per device, llama-kv-cache.cpp:111-134, 278-296).
  Regrowing per-device would need 2x per-device peak (~21 GiB on
  ~10.7 GiB of per-device KV at 27B scale) - does not fit. Fix:
  restructure the ctor to per-(layer, buft) ggml ctx + buffer, so
  resize() can free each layer's old buffer right after its D2D
  copy. Peak = old total + one layer's K+V (27B: 16 attn layers,
  ~21.4 GiB KV total at ~910k tokens -> ~0.7-1.4 GiB extra per
  card), which fits the 1.6-2.7 GiB free/card headroom.
- GDN recurrent memory is NOT n_ctx-dependent: rs_size =
  max(1, n_seq_max) (llama-model.cpp:2944); ring size = cell count
  = n_seq_max (llama-memory-recurrent.h:70). No GDN grow needed.
- MTP/draft ctx (mtp_on_hybrid_qwen, llama-model.cpp:2870-2873)
  is a standalone plain llama_kv_cache (filter il >= n_layer,
  mem_other = nullptr: own cells, own tensors) - the server
  driver grows it too, same size; rollback = resize back.
- No re-pointing after grow: common_memory (common.h:988-998)
  holds only the ctx pointers; the prompt cache holds HOST buffers
  (prompt_save :259-283, LLAMA_STATE_SEQ_FLAGS_NONE) keyed by
  position - entries stay valid after a tail extension, no
  invalidation needed.
- The per-seq state export/import (llama_state_seq_get/set_data_
  ext) is still the right tool for the draft rollback path and for
  debugging, but the happy path needs no host save at all.

Design (final, agreed 10-25):
1. Core (stage 1):
   - llama_memory_i: virtual bool resize(uint32_t new_size)
     { return false; } (precedent: get_cell_usage, llama-memory.
     h:107).
   - llama_kv_cells: keep-existing resize (append empty cells; the
     existing resize() at :63-70 resets state and is NOT used).
   - llama_kv_cache: ctor restructure to per-(layer, buft) ggml
     ctx + buffer; resize(new_size) = per layer: allocate new K/V
     tensors -> ggml_backend_buffer_copy_tensor rows [0, min(S,
     S')) (ggml-backend-impl.h:79) -> free the old buffer -> swap
     the k/v pointers, rebuild the k_stream/v_stream views, re-
     point shared-layer entries (map_layer_ids); then the
     keep-existing cell resize; clamp v_heads; handle the no_alloc
     (dummy) path; returns false when other != nullptr (shared-
     cell consumers - the GEMMA4_ASSISTANT share path - are not
     supported in stage 1).
   - llama_memory_hybrid::resize: delegates to mem_attn only
     (mem_recr is n_ctx-independent, see above).
   - ctx-level public API (llama_context_resize or similar,
     include/llama.h + llama-context.cpp): compute the new
     n_ctx_seq mirroring the :376-392 derivation (unified:
     n_ctx_seq = n_ctx; else n_ctx / n_seq_max padded to 256,
     n_ctx adjusted back), call memory->resize, and on success
     update cparams.n_ctx/n_ctx_seq (non-const member,
     llama-context.h:300; n_ctx()/n_ctx_seq() read cparams,
     llama-context.cpp:867-873).
2. Server (stage 2):
   - OPT-IN cap flag mirroring -fitm (name TBD, e.g. --grow-ctx-
     max; 0 = off) in common_params + arg.cpp + common.cpp print
     (pattern: -fitm at arg.cpp:2910-2916, common.h:472,
     common.cpp:1323). SUPERSEDED 10-27: the flag landed as the
     4-flag adaptive-ctx family (F34); --grow-ctx-max was never
     shipped.
   - Trigger (a) task start (:3097 non-splittable / :3108
     splittable): if the request > slot.n_ctx and the cap allows,
     grow before rejecting.
   - Trigger (b) decode wall (:3632-3674): after
     try_clear_idle_slots() fails (:3637), grow, then
     DECODE_RETRY.
   - Both sites are between decodes (no graph in flight): (a) is
     pre-launch, (b) is after llama_decode returned ret == 1.
   - Growth target: round up to a multiple of n_pad (1 for
     hybrid), step-based (e.g. max(4096, current/4)), capped at
     the flag.
   - Grow ctx_dft too when present; rollback both via
     resize(old) if the draft regrow fails.
   - On success: update n_ctx_slot + every slot.n_ctx (they all
     read the same shared ring).
3. Verify (stage 3): build (bat recipe, ninja quirk, SYCL=ON); on
   27B PTQ1_0: grow at the wall (small -c + cap), confirm
   continuation without slot reset, output consistency, failure
   path (cap too low -> normal EXCEED_CONTEXT_SIZE), MTP case if
   feasible.

Parked (future optimization, BACKLOG B16): the VA-page version -
reserve the full max ring's virtual address space up front, then
page-map physical memory as tokens arrive (ideal zero-copy
growth, no regrow at all). Not reachable on the SYCL stack: it is
a driver/memory-API surface question, no device kernels involved,
but the needed mapping API is not exposed there. Revisit only if
the per-layer regrow cost proves too high in practice.

Status: Stages 1+2 implemented and BUILT GREEN 10-26; unit tests
100% green (F34). Stage 3 (verify on 27B PTQ1_0) awaits the user's
install. B14 remains PARKED behind this path.

## F34. B15 implementation: core resize, server triggers, unit tests (10-26)

Implementation of F33 stages 1+2 (uncommitted at refresh).

Stage 1 (core), as designed:
- llama_kv_cache ctor restructured to per-layer ggml ctx + buffer
  (layer_ctxs/layer_bufs, llama-kv-cache.h:289-290); regrow_layer
  (llama-kv-cache.cpp:450) = fresh no_alloc ggml ctx with the new
  k/v + stream views -> allocate buffer on the layer's own buft ->
  zero-clear -> copy rows [0, min(old,new)] stream by stream via a
  host staging buffer (get/set resolve the possibly-split tensors)
  -> atomic swap of k/v + views + ctx + buf (old released with it).
  Atomic: the layer state is swapped only after full success.
- llama_kv_cells::resize_keep (llama-kv-cells.h:75): grow appends
  pos=-1 cells; shrink validates ALL dropped cells before mutating
  (a failed check touches nothing).
- llama_kv_cache::resize (llama-kv-cache.cpp:385): no-op if equal;
  refuse other/n_shared (shared-layer consumers); GGML_ASSERT
  new_size % n_pad; cells first, then per-layer regrow with
  rollback (see bug below).
- llama_memory_hybrid::resize (llama-memory-hybrid.cpp:147):
  mem_attn only (GDN recurrent is n_ctx-independent).
- Public API: llama_context_resize (llama-context.cpp:4028) ->
  memory_resize (:955): reject n_ctx < 256, GGML_PAD(n_ctx, 256),
  no-op if unchanged after padding, n_ctx_seq derivation mirroring
  the ctor (kv_unified: n_ctx_seq = n_ctx; else n_ctx / n_seq_max
  padded to 256), memory->resize, update cparams.n_ctx/n_ctx_seq,
  gf_res_prev->reset() + graph re-reserve (the k/v tensors have new
  addresses; the worst-case graph is sized by the full cache).

ORDERING BUG FOUND + FIXED in resize() (unit test catches it):
- The first draft regrew the data tensors BEFORE the cell-usage
  check. A failed shrink destroys the tail rows (the row copy is
  one-way: the new buffer only holds min(old,new) rows), then the
  rollback regrow copies only min(new,old) rows back, so the used
  tail cells are left marked in use with ZEROED data while
  resize() returns false. Any continuation decode would attend over
  the zeroed keys.
- Fix: resize the cells FIRST (a failed shrink fails fast with
  nothing touched), save v_heads (clamp to 0 if >= new_size), then
  regrow the layers; on a layer failure roll back the layers AND
  the cells (resize_keep(old_size) always succeeds - only empty
  cells were dropped or added) and restore v_heads.
- Mutation-verified: temporarily restoring the old ordering makes
  test case (d) fail (max |logit diff| 0.918 vs the control); the
  fixed code passes at 0 (byte-exact).

Stage 2 (server), as designed; the flag surface was later
redesigned by the user into the final 4-flag adaptive-ctx family
(the initial grow-only flag below was hard-replaced, not kept):
- FINAL FLAGS (locked 10-27): -actx/--adaptive-ctx [none|half|full]
  (default none), --adaptive-ctx-min N (default -1),
  --adaptive-ctx-max N (default -1), --adaptive-ctx-idle 0|N (default
  30); env LLAMA_ARG_ADAPTIVE_CTX[_MIN|_MAX|_IDLE]. Old flags
  --fit-ctx/-fitc, --fit-ctx-max/-fitm, --grow-ctx-max are HARD
  DELETED: a live spawn line using them will not start the server
  until it is swapped at the next install.
  Semantics: auto min = the load-time base the fit settled before
  growing (fit_ctx_base, common.h:482; explicit min wins and becomes
  the shrink floor); max -1 with mode=full = grow-to-fit capped by
  the training ctx (server-context.cpp:1088-1093); idle 0 = grow-only
  (shrink disabled); the idle shrink is a single jump to the floor,
  gated by t_idle_start_ms (server-context.cpp:870) set when the last
  slot goes idle. Per-resize duration lines: "adaptive ctx: KV cache
  grown/shrunk in place in %lld ms".
  Without --fit the same flags work: base falls back to n_ctx and max
  must be explicit (fit_ctx_base stays 0).
- try_grow_ctx(n_min) (server-context.cpp:1640): cap check;
  slot capacity capped at n_ctx_train; step target =
  max(4096, cur/4) or enough to fit (n_min), capped at
  min(cap/(unified ? 1 : n_seq_max), n_train); resize ctx_tgt,
  then ctx_dft (roll back the target on draft failure); on success
  n_ctx + n_ctx_slot + every slot.n_ctx + prompt_cache->limit_
  tokens follow the context.
- Triggers: task start (~:3100-3130, grow before rejecting an
  over-cap request) and the decode wall (~:3640, exactly where the
  shipped try_clear_idle_slots + DECODE_RETRY sits, F31). Both run
  on the loop thread between decodes.

Thread safety (in-process) - VERIFIED:
- All context ops happen on the server's single loop thread:
  server_queue::start_loop (server-queue.cpp:278) alternates
  process_new_tasks -> callback_update_slots (decode at
  server-context.cpp:3662) -> wait; both grow triggers fire there
  between decodes. HTTP threads never touch the ctx: /metrics
  (server-context.cpp:4667) and /slots (:4729) post tasks
  (SERVER_TASK_TYPE_METRICS / SLOT_GET); get_metrics (:825)
  returns a struct copy; get_meta/get_llama_context are
  main-thread-only by documented contract;
  common_memory_breakdown_print (server.cpp:545) runs post-
  shutdown after the joins.
- llama_kv_cache::resize is lock-free like every other KV mutation
  (caller-serialized). A one-line doc note was added to
  llama_context_resize (llama.h:580-584); precedent: the
  tokenization section ("The API is thread-safe.", llama.h:1172).

Unit tests: tests/test-context-resize.cpp (NEW, user-requested;
C API only + real GGUF, registered in tests/CMakeLists.txt after
the test-thread-safety block with FIXTURES_REQUIRED test-download-
model; model = tinyllamas/stories15M-q4_0.gguf, ~9 MB, already in
build/). n_batch=1024, n_ubatch=128, n_seq_max=1; deterministic
tokens = one fixed prompt tiled. Cases:
  (a) grow preserves KV: ctx 512, decode 400, resize(1536),
      decode 144 more; final-position logits vs a fresh 1536
      single-pass reference -> max |logit diff| 7.49e-05 + argmax
      match (tol 1e-4).
  (b) grow on an EMPTY cache + 256 rounding: resize(1500) -> true,
      n_ctx == 1536, then a 300-token decode succeeds.
  (c) no-op: resize to the current size -> true; and an empty
      cache shrinks (512 -> 256) -> true.
  (d) shrink dropping USED cells: ctx 1024, decode 800,
      resize(512) -> false, n_ctx stays 1024, and the continuation
      (800..900) is BYTE-EXACT (diff 0) vs a same-split control
      continuation without the resize call. This is the regression
      test for the ordering bug above.
  (e) shrink dropping only the EMPTY tail: decode 320, resize(512)
      -> true, n_ctx 512, continuation (320..470) byte-exact vs
      the same-split control - the kept rows survive the buffer
      copy.
Results: ctest 100% (test-download-model fixture + test-context-
resize, ~8.4 s); deterministic across runs (identical diff
values). The small llama model exercises the plain
llama_kv_cache path; the hybrid (GDN) path is covered by stage 3
on the 27B.

Why C API + real GGUF (not the synthetic-model machinery):
- The synthetic tests (test-kv-mean-center, test-llama-archs) are
  excluded on WIN32 + BUILD_SHARED_LIBS (tests/CMakeLists.txt:172,
  if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)); the block is from
  upstream commit 7f4fbe518 ("allow building all tests on windows
  when not using shared libs"). Cause: llama_model_saver is
  internal (src/llama-model-saver.h, not exported from the DLL).
  llama_model_init_from_user IS exported (llama.h:508) - only the
  saver blocks that path on this platform.
- The plain llama arch of stories15M hits the same
  llama_kv_cache path the server grow uses (the 27B adds the
  hybrid wrapper, verified in stage 3).

Empirical note (test design driver, not a bug): on this SYCL
build the ubatch split is NOT fp-invariant for stories15M-q4_0
(FA disabled: "Flash Attention not supported, set to disabled"):
a 2-decode continuation (800|100) diverges from the single-pass
900 by max |logit diff| 0.448, while (400|144) vs 544 is
7.49e-05. Same split, no resize: diff 0. So (d)/(e) use a
same-split control (the resize is the only difference) and only
(a) crosses splits (tol 1e-4).

Other:
- llama_n_ctx/llama_n_ctx_seq/llama_n_seq_max return uint32_t
  (llama.h:573-577): the server trigger code needed (int) casts at
  the std::min sites (build error fixed in the stage 2 pass).
- Multi-instance (router) semantics: the adaptive-ctx bounds are
  PER INSTANCE (set per child, e.g. via the env for router children).
  Each instance owns its pool; no cross-process state, no lock
  needed. Growth is an allocation: on failure the instance keeps
  its current size and the request gets the normal
  EXCEED_CONTEXT_SIZE rejection (no corruption, no effect on
  siblings). The per-instance cap bounds the worst-case KV
  footprint per child, so total VRAM can be planned as the sum
  over children of (weights + KV at cap).

Status (10-27): all stages built green (llama-server + test), ctest
100% (9.72 s), adaptive-ctx flag family in place, T2 (full actx set
on gemma-4-E4B) PASS incl. the load-time grow 8192 -> 32768. E2E
timed matrix (shrink/regrow pause vs the ~2 s gate) is in progress;
blocked by live-service VRAM fragmentation (F35). Commit + the live
spawn-line swap (--fit-ctx/--fit-ctx-max -> --adaptive-ctx full /
--adaptive-ctx-max 262144 / --adaptive-ctx-idle 30) are pending.

## F35. gemma-4-E4B E2E bisection: init-time crash signatures under live-service VRAM pressure (10-27)

Context: adaptive-ctx E2E (gemma-4-E4B-it-Q4_0, E:\DataVolume)
alongside the live 27B Q6_K service. Both cards held by the live
child (~12-14 GiB free each, fragmented; the child restarted 3x in
3.5 h, see the bad-allocation thread). All runs: build\bin\
llama-server.exe; BASE set = `--fit on -c 8192 -actx full
--adaptive-ctx-max 32768 --adaptive-ctx-idle 5 --cache-type-k q5_1
--cache-type-v q5_1 --flash-attn on --threads 56` (no
split-mode/batch-size).

Bisection matrix (single flag added to BASE; T2/B1/B2 ran against
the OLDER live child, B3 + the E2E re-runs against the 3:37 AM
child 17416):
- T2 BASE: PASS. Prints `adaptive ctx: mode=full base=8192
  max=32768 idle=5s`; n_ctx_slot 8192 -> 32768 at load (the fit
  grow-to-cap, fit.cpp:573-624).
- B1 + --batch-size 2048: PASS (no-op: the default is already 2048,
  common.h:451).
- B2 + --flash-attn on: PASS (no-op for gemma4: gemma4.cpp never
  sets f_attn, FA is auto-on everywhere).
- B3 + --split-mode tensor: 3 load-time crash variants (sig 3-4).
- T1 BASE + all 3 perf flags: loaded, then CRASH in warmup decode
  (sig 1).
- BASE re-runs (the E2E set) vs the new child: run 1 = silent
  0xC0000409 after a ~26 s slow load, zero stderr (sig 5); run 2 =
  clean SYCL alloc failure at model load (sig 2). Same args PASSed
  (T2) and fail depending on the child's VRAM layout: non-
  deterministic under this VRAM state.

Signatures (all PRE-EXISTING; our fit/actx code exonerated - T2 PASS
the full actx set, and every crash site is at/after model load,
before the server's first `adaptive ctx:` line):
1. 64-pool assert (T1): warmup-decode graph build (window:
   common.cpp:1417 threadpools.init -> common.cpp:1520-1556 warmup,
   11-13 ms after the threadpool line) hits ggml.c:1822
   GGML_ASSERT allocating a 65th tensor into a no_alloc-style pool
   of exactly 23552 B = 64 x 368 (ggml_tensor_overhead(), ggml.c:
   1483); needed 23920. Owner NOT identified: every ggml_init_params
   site checked - kv per-layer pool = 2*(1+n_stream)*368 (would need
   n_stream=31, excluded: kv_unified forces 1), k_bar ctxs =
   layers*368 (too small), recurrent = 2*n_layer*368 (gemma4 has no
   recurrent layers), loader/model ctx (load-time, too big), sched
   ctx (graph-sized), sampler probe (1024), meta reduce
   (n_reduce_steps=0), graph_copy (CPU-only), hadamard (1).
2. Fit layer-split contiguity failure (E2E run 2): `alloc FAILED:
   1123.035 MiB direct` on SYCL0 (1177587584 B) at model load,
   clean exit=1. Mechanism: fit's layer mode gives each device a
   CONTIGUOUS slice of the model (fit.cpp:804-805,
   set_ngl_tensor_split_tbo :864) and the allocator groups
   consecutive same-buft tensors into ONE buffer alloc, segment-
   capped by the buft max_size (ggml-alloc.c:1168-1206, the
   :1187 max_size split) - one ~1.12 GiB segment that fragmented
   SYCL0 cannot serve. Free totals unchanged vs the T2 pass
   (~12.1 GiB): contiguity, not capacity.
3. B3 #1: load-time CPU malloc NULL, ggml.c:1661 GGML_ASSERT
   (ctx->mem_buffer != NULL) (ggml_init, mem_buffer = NULL path).
4. B3 #2 + re-run: load-time SYCL0 alloc failure 1652.622 MiB
   (1732899584 B) + ggml-backend-meta.cpp:1744 GGML_ASSERT
   (meta_buf_ctx->bufs[i]) - the meta/tensor-split path needs an
   even bigger per-device contiguous block for gemma, so the
   production-parity `--split-mode tensor` variant is NOT a
   workaround in this state.
5. Silent 0xC0000409 (E2E run 1): ~26 s slow load (the other runs
   load in ~3 s; even the CPU-only minimal control took ~25 s -
   system-wide commit pressure from the 20.47 GiB live child), then
   abort with ZERO stderr. Do not assume a silent 0xC0000409 is a
   GGML_ASSERT; buffered stdout W/I lines are lost on abort.

Placement notes (checked while unblocking):
- Non-fit `-ngl all` uses the SAME contiguous-ratio split
  (llama-model.cpp:1736-1748, upper_bound on cumulative splits) -
  NOT a workaround for sig 2.
- Tensor-mode fit never reassigns weights: "the split of the
  weights is fixed by the model ... only the context size is
  fitted" (fit.cpp:179-183) - the live production line
  (--fit on --split-mode tensor, 27B) does not take the
  layer-placement path of sig 2; it is exposed to sigs 3-4 instead,
  which have been observed on this box with that flag combination.
- The CPU-only minimal control (no fit, no -ngl) loads fine in the
  same state: binary/model healthy.

E2E status: the timed matrix (b -> 8 s idle shrink -> d regrow ->
restart capped -> e over-cap) is BLOCKED on the above; T2 + the unit
tests are the current evidence base.

Two OUR-side code bugs found during the first full E2E run (both
fixed, uncommitted):
1. Idle hook never fires on a purely idle server:
   server_queue::start_loop handles the 1 s wait_for timeout inside
   the inner loop (server-queue.cpp:357-370) without re-entering the
   outer loop, so callback_update_slots (the all-idle branch with the
   shrink hook) only ran when a task re-entered the queue. Fix: on
   timeout, unlock the lock, tick callback_update_slots, relock, and
   shift time_last_task by the tick duration (same rule as
   :311-317: tick time is not idle). Verified invariants: callbacks
   always run without mutex_tasks held (outer-loop call site :310
   runs before the lock at :314); the all-idle branch is idempotent
   (server-context.cpp:2854-2872 early return); metrics_flush_idle
   no-ops when n_prompt_queued == 0 (:4217-4224), so the ~1/s
   cadence is free.
2. iswa wrapper had no resize override: llama_kv_cache_iswa derives
   from llama_memory_i (a wrapper around TWO llama_kv_cache
   sub-caches: kv_base non-SWA + kv_swa SWA ring) and did not
   override resize, so every SWA model (gemma3n, gemma4, ...) hit
   the silent base default return false (llama-memory.h:118) ->
   the observed 6 us `E memory_resize: failed to resize the memory
   to 8192 cells`. Stage 1 covered llama_kv_cache +
   llama_memory_hybrid, missed the iswa wrapper. Fix: store swa_full
   + the unpadded ring cap n_swa*(unified ? n_seq_max : 1) + n_ubatch
   as ctor members; resize = ring first (swa_full ? new_size :
   GGML_PAD(min(new_size, n_swa_cap), 256), mirroring the ctor
   sizing), then base; roll the ring back (with an error log if the
   rollback fails) when the base fails. Ordering rationale: each
   sub-resize is atomic (resize_keep fails before mutating; the
   layer regrow rolls back on failure), and a base failure leaves a
   ring SMALLER than the base, which still wraps safely (never OOB),
   while the reverse would leave the base smaller than cparams
   (OOB). The dsa-iswa sibling (llama-kv-cache-dsa-iswa.h) has the
   same gap; out of scope, noted only.
   SWA ring row-copy safety (checked before writing the fix):
   find_slot allocates free cells from the head with wrap-around
   (llama-kv-cache.cpp:1083-1242), overwrite safety is per-cell via
   is_masked_swa position masking, so the regrow row copy is correct
   for the position-indexed base (rows = positions) and for the ring
   (grow: used-block row indices are invariant; shrink:
   resize_keep fails if any dropped cell is used).

Log forensics note: the .out file loses buffered stdout (all I/W
lines) on a force-kill; only stderr (E lines) survives. Timestamps
are event-time: the 6 us gap between the shrink line and the error
is real elapsed time, i.e. an immediate early return, not a slow
allocation failure. The E2E run's hook firing 532 s after the last
task (t=48.47 min) was because a /slots HTTP poll posted a task that
re-entered the outer loop - it proved the tick gap, not a
replacement for the fix.

Known limitation (no fix, production line unaffected): an
--sleep-idle-seconds shorter than --adaptive-ctx-idle makes the
queue enter the sleep state before the shrink can fire (the
sleep-state wait, server-queue.cpp:336-338, blocks the update_slots
cadence until a task wakes it). The production spawn line does not
set --sleep-idle-seconds (default -1), so this is only a trap for
future flag combinations.

Third OUR-side bug (fixed, uncommitted): the idle shrink retried
every second after a failure (E-line spam for the whole idle
period). Root cause: retained slot KV. The server keeps a finished
slot's KV until the NEXT task lands on that slot (release() only
prompt_clears child slots; n_cache_reuse is a separate KV-shift
feature, default 0; t_last_used only drives slot-steal selection
when a new task arrives, so nothing frees retained cells during a
pure idle). A shrink below the max retained cell boundary then
correctly fails (resize_keep, llama-kv-cells.h:75, is
all-or-nothing), and the old code retried every tick. Fix:
actx_shrink_blocked member (server-context.cpp:871) - on a failed
attempt set the flag and skip further attempts until a new task
arrives (the else branch at :2883 clears it, same place that resets
t_idle_start_ms). Rationale: while idle the cell state cannot
change, so a retry within the same idle period can never succeed.
The no-op case (already at the floor) also sets the flag, which is
harmless: it is cleared by the same new-task event that is required
to push the cache above the floor again (a grow) or free retained
KV.

E2E on the final binary (gemma-4-E4B, port 18100, all far under the
2 s gate; times in ms):
- load 32768 (mode=full) -> 5 s idle shrink 32768->8192: 19 ms
  (4-slot run), 21 ms (--parallel 1 run)
- phase d (9616-token prompt) grow 8192->12288: 19 ms (both runs)
- post-d idle shrink 12288->8192: correctly BLOCKED by the slot's
  9623 retained cells, exactly ONE attempt (I + 2 E lines), then
  silent for 20+ s of further idle (pre-fix binary: 1 attempt/s)
- --parallel 1 full cycle: phase f (small task, same slot) frees
  the retained KV at task start -> next 5 s idle mark shrinks
  12288->8192 in 26 ms -> post-shrink requests OK (prefix reuse
  still works: f_keep = 0.958 on an identical-prompt follow-up)
- note: with N>1 slots a shrink only succeeds once every slot that
  retains KV above the floor has received a new task; with
  LCP/LRU slot selection that takes one request per such slot. No
  slot-pinning API exists in this build (POST /slots/:id is the
  slot-save feature only).
- production-scale extrapolation (27B Q6_K, 262144->16284, ~6.3
  GiB q5_1 KV, copy-bound): order ~0.5-1 s, still under the gate.

Retained-KV shrink policy fork (OPEN, presented to user):
- A: on idle shrink, clear the idle slots' retained KV first, then
  shrink to the floor (full VRAM reclaim; a resumed long session
  pays re-prefill of its whole history)
- B: shrink only to the retained boundary (max used_max_p1() over
  streams, llama-kv-cells.h:124) - needs a new core accessor;
  iswa/dsa-iswa lack get_cell_usage overrides (base returns {})
- C (shipped interim, recommended): never touch retained KV; one
  attempt per idle period (the fix above). Still reclaims
  262144->16284 (~4 GiB at production scale) once traffic has
  cycled through the retaining slots.

## F36. B70/SYCL single-GPU reference data (4 URLs, fetched 10-27) vs the live dual-B70

[certain] Card specs (web-verified, consistent across all 4 sources):
Intel Arc Pro B70, BMG-G31: 32 GB GDDR6, 256-bit bus, 608 GB/s, 32 Xe
cores / 256 XMX engines, TBP 160-290 W.

Single-GPU reference numbers (dense Q4-class, llama.cpp SYCL):
- jeffgrover/b70-setup (USB4 eGPU, Q4_K_S, 256K ctx): pp8K ~1068 t/s,
  decode 12.4-25.9 t/s.
- sergiiob.dev full recipe (dense 27B Q4, one card): pp4096 936 t/s,
  decode 21.3 t/s. Their env recipe:
  SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0,
  SYCL_CACHE_PERSISTENT=0, SYCL_DEVICE_FILTER=level_zero,
  ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE, ONEAPI_DEVICE_SELECTOR=level_zero:0,
  ZE_AFFINITY_MASK=0.
- snailium/llama.cpp-sycl-intel-b70 (AOT
  -DGGML_SYCL_DEVICE_ARCH=bmg-g31): pp 470-490 t/s, decode 41-46 t/s.
  Mandatory env: ONEAPI_DEVICE_SELECTOR=level_zero:0,
  SYCL_CACHE_PERSISTENT=0, ZES_ENABLE_SYSMAN=1. The pp spread across
  sources (470-1068) reflects prompt size + AOT-vs-JIT + eGPU-vs-discrete.
- indiekings: prefill is compute-bound (XMX), decode is bandwidth-bound
  (608 GB/s ceiling); strip debug symbols.

Framing vs the live setup: ALL refs are single-GPU. The live child =
dual B70 (layer split) + Qwen3.8-27B-UD-Q6_K 20.47 GiB + q5_1 KV +
ctx 163840 + MTP draft + flash-attn + yarn. The 41-46 t/s single-card
decode (snailium, same 27B class) is the useful floor: the live
dual-card single-chat decode should be AT or ABOVE it; a live single-
chat TG below that points at driver/env or split-mode P2P overhead
(F38), not the hardware. The 12.4-25.9 t/s (jeffgrover) is USB4 eGPU
+ 256K ctx (long-ctx decode penalty), not comparable to our
163840-base, discrete, dual-card setup.

[open] No dual-B70 llama.cpp SYCL reference in these sources; the
split-mode overhead on the live child is the F38 mechanism.

## F37. Tensor-mode "bad allocation" decode failures (live service,
10-27 evening)

[certain] Symptom (user report + macube logs):
`update_slots: decode() failed: bad allocation` (e.g. [57315]
194.19.001.745, task 96704) happens ONLY under `--split-mode tensor`,
is random, and may need a server restart; the user reverted to
`-sm layer`. ALL 5 bad_alloc children are TENSOR mode (59129, 55422,
57315, 65508, 54879; 54879 = MTP variant, ctx 262144); the only known
layer child (52439, current live) is clean. Pattern: long decode
(26379 / 96802 tokens) fails once, retries of the same conversation
fail for minutes, then recover or hard-crash.

[certain] "bad allocation" = std::bad_alloc's DEFAULT what() (MSVC STL
under icx) = a HOST-side allocation failure. The SYCL backend prints
device OOM with its own text; a raw host throw is what surfaces.

[certain] Fatal variant (child death + router respawn): raw
`bad allocation` + abort in ggml_sycl_op_mul_mat
(ggml-sycl.cpp:3332) via CHECK_TRY_ERROR
(ggml-sycl/common.hpp:77-88, prints e.what()) -> ggml_sycl_error ->
GGML_ABORT (common.hpp:145). The same throw caught around
llama_decode = the non-fatal server-level `decode() failed` line.

[certain] PROVEN host bad_alloc site = the prompt cache:
server_prompt_cache::alloc (server-task.cpp:1711) does a host
std::vector::resize of the full KV state. Caught in the logs:
[65508] "failed to allocate memory for prompt cache state: bad
allocation" -> self-throttle to 1961.495 MiB; [54879] "prompt state
size 10828.796 MiB exceeds cache size limit 8192.000 MiB, skipping".
Default cache_ram_mib = 8192 (common/common.h:632); the live spawn
line has NO --cache-ram. At the live KV size (~27.5 KB/token q5_1), a
full 163840-ctx state is ~7-10 GiB of HOST memory per cache entry -
the 10.8 GiB skip above matches.

[high] Machine state at check (10-27 evening): 64 GB RAM (39.4 GB
free), pagefile 60.9 GB, commit limit 123 GB, ~23 GB free commit
(~100 GB committed). Per-process commit charge not readable
(Get-Process CommitVirtual = access denied; WS only).

[open] The tensor-decode-specific host allocation site is NOT pinned:
the split mul_mat staging (ggml-sycl.cpp:3100-3388) uses per-op
buffers from ctx.pool(i) = DEVICE memory (src0_dd/src1_ddf/src1_ddq/
dst_dd); dst stays on device 0 via dpct::async_dpct_memcpy dev->dev.
ggml_sycl_host_malloc (common.cpp:26, pinned sycl::malloc_host) has no
call sites in ggml/. Working hypothesis: host COMMIT exhaustion
surfacing at an arbitrary `new` during decode, aggravated in tensor
mode because fit is a no-op there (llama_params_fit is not
implemented for SPLIT_MODE_TENSOR - warning at load) so the server
always runs the full 163840 ctx and exports GiB-scale prompt-cache
states, while layer mode + fit keeps the ctx (and the cache states)
smaller.

Mitigations (ranked, for the handover):
1. `--cache-ram 0` (or a small value) in the macube spawn line -
   user-owned, one flag, kills the proven host OOM class without
   touching model behavior.
2. Stay on `-sm layer` short-term (the clean child).
3. B14 twin-servers + router as the STRUCTURAL fix: each child is
   single-card (no split-mode P2P waits, F38; tensor mode not needed
   to use both cards), so both reported issues (balance + tensor
   bad_alloc) disappear by construction.

## F38. Layer/row-split two-chat balance: per-layer blocking P2P waits,
no slot pipelining (10-27)

[high] User report: with `-sm layer` (maybe row too), two concurrent
chats are not balanced - the devices wait for each other per token and
TG halves. User expectation: phased load (device 2 does chat#1 token i
while device 1 does chat#2 token i).

[certain] Code evidence for the per-token wait (ggml-sycl.cpp split
mul_mat :3100-3388): the split path issues per-layer q8_1 weight
copies dev->dev with BLOCKING .wait()s (:3288-3292), dev2dev_memcpy
(:3298), and event barriers (:3361-3366, :3371-3387); dst accumulates
on device 0 (dpct::async_dpct_memcpy dev->dev, :3335-3358). Every
layer therefore serializes both cards: a token cannot progress until
the P2P copies + barriers of that layer complete on BOTH devices.

[certain] No slot pipelining exists in one llama-server: slots are
evaluated SEQUENTIALLY through one shared pipeline (update_slots,
server-context.cpp:2844+). Two chats sharing one server alternate
through the same serialized per-layer split path; there is no
mechanism to run chat#2 on card 1 while chat#1 is on card 2. The
halved TG under two concurrent chats is the expected consequence, not
a misconfiguration.

The user's "phasing" = DATA PARALLELISM, which is exactly B14
(twin-servers + router, one child per card). B14 sidesteps BOTH
reported issues: no split-mode P2P waits (each child is single-card)
and no tensor mode (so the F37 tensor-only bad_alloc class does not
apply). Status: B14 remains PARKED (draft pending user); this finding
raises its priority.

## F39. Two-chat pipeline proposal: assessment (10-28)

[proposed] Follow-up to F38: a steady-state DECODE PIPELINE across
the two cards - the graph is split at the device boundary and the
next token's half-graph runs on the other card while the current
one finishes, so two concurrent chats advance in a staggered
stream (the "phasing" the user described; user phrasing 10-28:
"2 chats, 1 token per half round"). DRAFT PROJECT written 10-28
(evening) at DESIGN-alternating-pipeline.md on the user's request
(design only, no code; BACKLOG B18).

[certain] Steady-state throughput: with a half-graph cost
T_half = H/2 (H = the current serial per-token cost of the layer
split), the pipeline emits 1 token per T_half = 2/H aggregate -
EXACTLY the B14 data-parallel aggregate (2 x 1/H). Roofline: a
system token reads the full weight set W either way (serial layer
split, two DP streams, or the pipeline), so nothing beats 2/H
without reducing the W reads. Per-stream TG: the pipeline serves
each chat at 1/H, TIED with DP. The pipeline buys NO throughput
over B14.

[likely false] The "copy limited" premise (a pipeline would hide
the P2P copies) does not hold for LAYER mode: only the residual
stream crosses the device boundary per token (embd 5120 f32 = 20
KB), and the meta backend already runs that asynchronously
(subgraph split only at PARTIAL tensor boundaries,
ggml-backend-meta.cpp:2176; per-subgraph async enqueue :2418-2424;
boundary copy via the butterfly, ggml_backend_tensor_copy_async
:2343, no host-side sync bubble). The large BLOCKING q8_1 weight
copies (ggml-sycl.cpp:3288-3387) belong to the tensor/row split
path, which is not active in layer mode. F38's per-layer wait is
the sync structure of the layer split itself, not a copy stall a
pipeline would overlap.

[certain] What the pipeline WOULD buy: KV headroom. DP is capped
by one card's free VRAM per stream (~10 GB/card here => ~20 GB
total for two chats); the pipeline shares both cards' KV across
the SAME chat pair (~20 GB/card => ~40 GB total) - ~2x the usable
context for the two-chat case at the same 2/H aggregate.

[certain] What it would COST: a third execution model in this
tree - the graph split at the device boundary, a two-phase async
decode API on the server, a stage scheduler that pairs the two
chats, stage-imbalance handling (the 48 GDN + 16 full-attn hybrid
does not split evenly by device), and MTP/speculative breakage
(the draft model + reasoning-budget path assumes one blocking
decode per token). Upstream would not merge it: no mainline
dual-card SYCL pipeline exists; it is a fork-specific capacity
hack layered on the meta backend.

[recommendation] Do NOT build it now. B17 is done (10-28); next
MEASURE: 1-stream TG vs 2-slot TG on the live dual-card setup
(single-stream baseline recorded 10-28: 12.53 t/s on a 91559-
token conversation, layer mode). If the measured 2-slot TG lands
near the 1/H expectation and the workload genuinely pins BOTH
cards at max ctx simultaneously, the ~2x KV headroom justifies the
pipeline; otherwise B14 (DP + router) delivers the same aggregate
with existing execution models and half the risk.

[measured 10-28] The MEASURE step, run on the live dual-card
service with the DEFAULT spawn line (no --parallel):
- The default config IS a batching config: the server auto branch
  (tools/server/server.cpp:152-157) sets n_parallel = 4 and
  kv_unified = true when --parallel is absent. Two concurrent
  chats landed on separate slots (3 and 1) and decoded in ONE
  batch of 2 seqs (llama.log :178425-178437, child 51396):
  per-stream tg_3s 10.7-10.8, SUM ~21.4 t/s (user-reported ~20).
- Solo finals, same child: 20.02 @ 8.5K, 14.03 @ 34.8K, 15.17 @
  53.4K, 13.99 @ 67.5K, 12.90 @ 84.8K (the 12.53 @ 91559 baseline
  continues the curve).
- Verdict: the 2-chat sum is ~76% of the 2/H roofline (solo ~14
  at the comparable ~35K ctx) - the 2-seq batch step costs ~93 ms
  vs the ~71 ms solo step at that ctx. Batching works without any
  flag; the gap to 2/H is per-step 2-seq overhead, not a config
  problem.
- The overlapped chats were 27.6K + 7K tokens: NOT pinning both
  cards at max ctx. The pre-registered gate (pipeline justified
  only if the workload pins BOTH cards at max ctx simultaneously)
  is NOT met. Concrete boundary: the unified 262144-row ring holds
  two ~131K chats; the largest live conversation so far is 106815.
  The capacity question becomes real only above that.

[disputed 10-28 (evening)] The user tested 2-chat throughput
several times: the sum is FAR from 2x the solo TG - the 76% above
is one overlap sample (27.6K + 7K) and the verdict is disputed,
not settled. The per-step numbers pin down the structure of the
gap:
- The 2-seq step (93 ms) is NOT 2x the solo step (71 ms; else
  >= 142 ms): the weight set IS read once for both seqs - the
  weight sharing the user expects is real and already happening.
- Decomposition: solo ~= W + A, 2-seq ~= W + 2A, A ~= 22 ms (one
  seq's attention/KV/GDN-state share, A/H ~= 0.31). The measured
  1.53x is AT the batching ceiling 2/(1 + A/H) ~= 1.52x: default
  batching is not misconfigured, it is already doing the right
  thing.
- The ceiling's cause is STAGE IDLE, not per-seq waste: in layer
  mode the 2-seq batch is one graph - [both seqs' stage 1 on
  SYCL0] -> [both seqs' stage 2 on SYCL1] - so SYCL1 idles for
  the whole stage-1 work of the pair (and vice versa); the pair
  takes (W/2 + A) x 2 = W + 2A = H + A. The alternating pipeline
  (DESIGN-alternating-pipeline.md) overlaps stage 2(token N) with
  stage 1(token N+1): both cards continuously busy, pair takes H,
  aggregate 2/H = 28.2 t/s, +31% over the measured 21.4. The per-
  seq A is unchanged (each seq still reads its own KV); only the
  idle bubble disappears.
- Open: a controlled same-ctx 2-chat run (both ~35K, equal load)
  to pin A and check for a per-seq fixed overhead beyond KV
  (GDN state update, per-seq sampling, non-fused attn kernels).

[certain 10-28] The -np 2 anomaly (child 57327), root-caused in
code:
- Explicit --parallel 2 skips the auto branch, so kv_unified keeps
  its default FALSE (common/common.h:575) -> per-slot KV:
  n_stream = n_seq_max (llama-kv-cache.cpp:86, K/V tensors are
  3D [n_embd_gqa, kv_size, n_stream], :231-232). Total KV doubles:
  2 x 262144 rows ~= 14.4 GiB (~7.2 GiB per card) vs ~7.2 GiB
  unified.
- The adaptive-ctx TOTAL cap doubled accordingly BY DESIGN:
  n_ctx_max = n_ctx_train * n_streams (common/fit.cpp:456-458,
  :528) = 262144 x 2 = 524288 (child log :178613); the no_alloc
  fit measured it and granted it (the per-slot share stays capped
  at n_ctx_train). "The adaptive ctx does not account for -np" is
  half right: it DOES account - by doubling the total.
- SUPERSEDED (10-28 evening, user CONFIRMED): the child was NOT
  serving - the model itself was unable to generate a single
  token (57327 launched task 0 and produced zero timing lines),
  and with -np 2 the MODEL + KV sit on the HOST, not the GPUs
  (the GPU2 full->0 + CPU ~+30 GB monitor read was correct). The
  arithmetic holds: 27G model + 14G KV = 31G across two 32G cards
  should leave ~half a card free; the failed trial showed the
  cards almost empty. After the first unload the model+KV come
  back onto GPU2 and it circles, very slow; it does this every
  time lately. One historical exception (user memory): one
  -np 2 -c 256k run worked - both GPUs full AND CPU memory at
  100% all day, TG ~5 (model+KV co-resident on the GPUs AND the
  host). +30 GB ~= model ~17.5 GB + per-slot KV 14.4 GiB ~= 32
  GB matches the confirmed host residency.
- New log evidence from the pre-178327 -np 2 trials (found this
  round): 58561 + 62981 (Qwen3.6-27B-Q6_K, -np 2) loaded in
  2.26/2.16 min and were killed on "exit command"; 53059
  (Qwen3.8-27B-UD-Q5_K_S, 17.38 GB) was a MISTAKE load (user:
  loaded by mistake, since unloaded; Q6_K is the standing
  preference) and never printed a "model loaded" line; 54674
  (Q6_K, -np 2, started 2 ms AFTER 53059 - a concurrent double
  load) HARD FAILED (llama.log :178227-178243): `alloc FAILED:
  696.000 MiB direct` -> `can't allocate 729808896 Bytes ... on
  device` -> `failed to allocate buffer for kv cache` ->
  `exiting due to model loading error` - the co-resident 53059
  left SYCL1 with < 696 MiB free for 54674's first per-slot KV
  slice. So -np 2 KV placement dies at load under ANY VRAM
  contention (~0.7 GiB unavailable on one card is fatal).
- 53777 (Q6_K, -np 2, clean single-model load, warm page cache,
  loaded in ~46 s) is the "cannot start a single chat" trial:
  task 0 (slot 1) prompt eval 22.66 -> 19.17 -> 11.16 t/s, task 2
  (slot 0) 0.49 t/s (124.5 s / 61 tokens) then 13.54 t/s -
  CPU-class decode with ZERO alloc errors in its whole section.
- SYCL-MEMTRACE semantics (ggml-sycl/memtrace.cpp): FAILED device
  allocs log at E level ALWAYS; OK allocs need the env
  GGML_SYCL_MEMTRACE=1 (per-type totals) or =2 (per-alloc). Zero
  MEMTRACE lines in the 53777/57327 sections => no device alloc
  ever FAILED in those children. Combined with "every fallback
  logs" (pool-malloc ERROR ggml-sycl.cpp:1674-1678, FA buffer
  GGML_ABORT fattn-buffers.cpp:39-42, reorder host-fallback WARN
  :3913-3919, pinned-host fallback WARN :1528-1531 - none
  present), the host placement in 53777/57327 has NO logged
  trigger: no code path in this tree is known to move model+KV to
  host silently. The user's observation is the ground truth; the
  mechanism that places -np 2 model+KV on the host (and the
  unload -> reload-on-GPU2 circle) is still unproven in code.
- Repro to catch it (needs a stop window, user-owned): ONE Q6_K
  model, no co-resident child, -np 2, spawn env
  GGML_SYCL_MEMTRACE=1 - the OK-alloc trace shows, buffer by
  buffer, which device (SYCL0/SYCL1/host) each model tensor and
  the per-slot KV lands on; capture the user's monitor read at
  the same time.
- The service then fully restarted (user): old router died ~3.32
  min (llama-error.log:16752), new router, child 49822 respawned
  at 10.27 min with the ORIGINAL config (no --parallel) - not a
  child crash.
- Recommendation (unchanged): drop -np 2 from the bisection.
  Per-slot isolation buys nothing for 2 chats (unified KV already
  interleaves them on one ring) and costs 2x KV memory + a 2x
  total adaptive-ctx cap, and now also shows host placement + a
  load-time fragility. Keep the default (auto 4 slots + unified
  KV).

[decision 10-28] Pipeline: NOT building. The measured 2-chat
workload (27.6K + 7K) does not pin both cards at max ctx, so the
~2x KV headroom - the pipeline's only real win - is not on the
table. B14 (DP twin-server + router) was REJECTED by the user
10-28 (evening): "I will never start. It was dead end." - it is
no longer the capacity vehicle; the capacity path for > ~131K
per concurrent chat is the in-place ring growth (B15, shipped) +
the unified ring, with the alternating pipeline (DESIGN-
alternating-pipeline.md, B18) as the drafted - and only
remaining - 2x-aggregate design on record. The 76% batch
efficiency is DISPUTED by the user (repeated tests far from 2x;
see [disputed] above) - the stage-bubble analysis is the current
explanation; a controlled same-ctx 2-chat run is the open
measurement.

## For consideration (not approved, no code written)

Deferred design options for the KV/memory wall, in the order discussed
10-24/10-25. The two already-shipped pieces (F31) are the no-crash
landing and the fit-time auto-size; these are what is left on the table:

1. Degradation signature (observed, unexplained): 50588 went from ~18
   t/s to ~1.1-1.5 t/s on BOTH slots with zero child-log events; 60117
   went 18 -> 0.09 t/s. Time/load-correlated => system-level, not
   per-request. Candidate causes to check when next reproducible:
   driver/VRAM fragmentation after long ctx, commit charge pressure,
   SYCL runtime stalls. No fix proposed.
2. LCP router thrash: the macube router (59148) LCP lines showed the
   router re-baselining sessions; combined with (1) this is the likely
   source of the "sudden slow" episodes. Needs a macube-side look
   (user-owned; read-only at most).
3. Twin-server + router (data parallel, deferred - see BACKLOG B14):
   two llama-server children pinned -dev SYCL0 / -dev SYCL1 behind a
   router that aggregates /v1/models and routes by session. Draft
   pending user confirmation (registry config shape, budget via the
   fit.cpp projection, idle-offload + pin/lock).
4. KV ring design options: fit-time auto-size (shipped, F31 stage
   2) handles the startup sizing. NEW 10-25: in-place runtime
   growth is now the CONFIRMED path (F33, user agreed 10-25) - the
   old "REJECTED (KV pool fixed at context creation)" note is
   SUPERSEDED: the per-(layer, buft) restructure makes the regrow
   affordable (peak = old total + one layer). The VA-page version
   is parked as BACKLOG B16 (future optimization).
