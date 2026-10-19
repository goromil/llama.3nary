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

[open] What exactly in the window calls ggml_init with a failing host alloc?
NEXT: read common.cpp ~:1230-1460 (common_init_from_params / common_init_
entry + the two "failed to create context" sites :1398/:1449) and
server-context.cpp :1030-1090 (spec params setup). Also rerun repro with
`--fit off` to exculpate/confirm the fit path, and with max verbosity to see
the last lines before the assert.

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

## F5. MTP OOM 3-part fix - applied, built, NOT committed

[certain] All three edits are in the working tree (common/speculative.cpp,
tools/server/server-context.cpp), build GREEN at 9:43 AM; one `\n` edit at
server-context.cpp:1100 applied AFTER the build => REBUILD PENDING.
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

## F9. SYCL topology: two SYCL devices visible, use ONLY device 0

[high] common_params_print_info device_info lists SYCL0 AND SYCL1, both
"Intel(R) Arc(TM) Pro B70 Graphics (31906 MiB)" (free: 8776 / 684 MiB at
that moment, under 61850 pressure) + CPU (Ryzen 5950X, 65445 MiB).
SYCL1 having only 684 MiB free makes no sense for a second physical card
(someone would have to hold ~31 GB) => same physical GPU enumerated twice
or a partition. Fit saw nd=2 (id=0 = pinned SYCL0, id=1 = CPU with
--numa distribute). Always pin GGML_SYCL_DEVICE=0.
[open] Confirm whether SYCL1 is a real separate resource (e.g. sycl-ls) -
low priority, do not touch it.

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
[open] server-slot.cpp does NOT exist in this fork; the slot state machine +
abort/response-stream plumbing lives in server-context.cpp + server-*.h and
must be located before C0.

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
