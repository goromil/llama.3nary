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
4. KV ring design options (if per-slot growth is ever needed):
   in-place ring growth is REJECTED (KV pool fixed at context creation);
   alternatives are fit-time auto-size (shipped, F31 stage 2), a larger
   per-slot reservation at startup, or per-slot ring rebalance on
   evict - none approved.
