# BACKLOG - open items (ordered)

Each item: what / why / how / exit criteria / status.

> PROJECT CLOSED 2026-10-19 by user decision, then REOPENED the same day
> (F5 status, F12 actions, F11 C1/C2 plan, ternary slowness -> F14).
> Closure disposition of the original items (still valid):
> - B3 CLOSED: leave the MTP ctx failure fatal on the has_draft path,
>   no code change.
> - B4 CLOSED as explained: the ggml.c:1661 assert is a pressure-dependent
>   host-malloc OOM (F1), not a deterministic code path; the pinned reruns
>   were not pursued.
> - B5: the binary copy is DONE 10-19 (89/89 SHA256 match, see STATE.md);
>   the rest of the free-VRAM suite PARKED (services now run the new
>   binaries anyway).
> - B6/B8/B10 PARKED: Part-3 trigger test, PQ2_0 sweep, SYCL docs.
> - B9: this banner is the accounting.
> B11 DONE 10-20 (5b80a3bb1: PTQ1_0 tg128 6.90 -> 26.72 t/s, F14
> resolved); B7 REOPENED with a ready plan (F11), then RESCOOPED 10-22
> (user decision: twin-server data parallel DROPPED; runtime preemption
> C0/C1 DROPPED => B7-C0/C1 CLOSED; B7 startup-side loud-fail + closed
> --dev gate APPROVED for implementation in 3nary, see B7a below).
> 10-22 PIVOT: user direction = make ternary code tensor-compatible in
> THIS fork first (B12, H-A design APPROVED by user 10-22, implement in
> 3nary then port), then port the ternary work to the upstream clone at
> C:\Users\admin\source\ai\llama.cpp (B13, target confirmed 10-22).
> 10-25: B12 DONE (7cb134632); KV wall shipped in two stages (no-crash
> 6a30f9157, fit-time auto-size -fitm db5cf4d80, F31); SYCL build
> finding F30 (SYCL=OFF = CPU-only runtime, build recipe now SYCL=ON);
> B14 (twin-server + router) drafted, pending user.
> Restart recipe: STATE.md, then NOTES.md (N2 setvars, N3 build, N4
> detached launch, N7 repro args, N14 tensor-mode repro), FINDINGS.md
> for conclusions.

## B1. Commit+push the private/ knowledge library
- what: `git add private/ && git commit && git push` (ONLY private/).
- why: user wants the knowledge base in the private repo as a restart anchor.
- how: ASCII message e.g. `docs : add private knowledge library` + trailer
  `Assisted-by: Claude, Qwen3.8-27B`.
- exit: pushed on main; does NOT touch the MTP-edited files or the user's
  svelte file.
- status: DONE 10-19 (commit 02af668bf; keep the library refreshed).

## B2. Rebuild + commit+push MTP 3-part fix
- what: rebuild build/ (one \n edit at server-context.cpp:1100 pending),
  then commit+push common/speculative.cpp + tools/server/server-context.cpp.
- why: fix is verified (build green, no-draft OOM is expected under
  pressure, abort risk resolved) but uncommitted and one edit behind.
- how: NOTES.md N2 (setvars) + N3 (build). Commit: concise ASCII, trailer.
- exit: HEAD pushed; install-dir copy still deferred until 61850 stop.
- status: DONE 10-19 (rebuilt, commit 0589b1e1a).

## B3. Flag has_draft fallback gap to user
- what: tell user that Qwen3.8 AND gemma-4 service presets have has_draft
  = true, so the Part-3 graceful MTP fallback never fires in service runs.
- why: otherwise MTP ctx failure stays fatal in exactly the service
  scenarios the user runs (58458 style).
- how: message; offer to extend the fallback to the has_draft case
  (small edit in server-context.cpp Part-3 block).
- exit: user decides (extend now / later / leave fatal).
- status: ANSWERED 10-19. Explained: has_draft is the PRE-EXISTING server
  flag ("--model-draft was passed"), not something we added; Part-3 soft
  landing only covers MTP ctx failure WITHOUT a draft file. Both service
  presets pass a draft => failure stays fatal there (existing behavior,
  unchanged). Latent, not active: 61850 runs fine with MTP (~15.5 t/s).
  Recommendation given: leave fatal (fail loudly if an explicitly requested
  draft model cannot load). CLOSED 10-19: user confirmed, leave fatal,
  NO code change.

## B4. Close the assert window (F1)
- what: read common.cpp :1230-1460 + server-context.cpp :1030-1090 (DONE);
  max-verbosity repro (DONE); remaining: reruns with a real device pin
  (`-dev SYCL0`) under pressure AND on free VRAM, plus a `--fit off`
  variant.
- why: root-cause the ggml.c:1661 host-malloc assert on the draft+spec
  path; it is pre-existing (F8) and kills the Qwen MTP service config.
  New 10-19 data: failure is pressure-dependent (assert in one run,
  sched_reserve stall in the other, F1) and our runs scattered layers onto
  the nearly full SYCL1 (F9) - a pin was never actually active.
- how: PLAN.md step 5-6; repro args in NOTES.md N7; detached launch N4.
- exit: the exact ggml_init call site in the window identified OR the
  pinned / free-VRAM reruns change the outcome (then re-window).
- status: OPEN (reads + maxver repro done, see F1; pinned reruns pending).

## B5. Free-VRAM verification suite (BLOCKED on user stopping 61850)
- what: (a) draft+spec 98304 on FREE VRAM (decisive pressure test),
  (b) standalone MTP E2E 98304, (c) 262144 short gen, (d) copy 6 files to
  install dir, (e) user starts service + 2-client repro, (f) grep
  llama.log for "kv cache stream" at next FAILED_PREPARE (expect
  used == size).
- why: MTP fix + KV guard need service-level verification; 61850 holds the
  VRAM and the user controls it.
- how: user stops 61850 -> PLAN.md step 7.
- exit: all sub-steps green; service running new binaries.
- status: PARKED 10-19 (sub-step d done: binaries deployed 89/89 SHA256
  match; a/b/c/e/f not run - services run the new binaries anyway).

## B6. Part-3 fallback trigger test
- what: prove the "continuing without speculative decoding" path once.
- why: cannot trigger naturally on the Qwen target (it has blk.64 nextn
  tensors).
- how: pick: model without nextn tensors (different arch path) / forced
  OOM (tiny -c or VRAM starvation) / skip with a note in the final report.
- exit: one successful fallback run OR documented skip.
- status: OPEN (needs B5 VRAM for the OOM option).

## B7. Option C preemption (C0 -> C1 -> C2)
- what: vLLM-style preemption for the 4x-KV-overcommit scenario.
- why: user values vLLM behavior; guard (a3f36885c) only prevents
  corruption, it does not give fairness under contention.
- how: PLAN.md phase 2; slot state machine LOCATED 10-19 (server-context.
  cpp :55 enum, :2288-2324 admission+defer, :3590-3639 decode-fail);
  upstream TODO server-context.cpp:3589-3590 = design reference; C0/C1/C2
  plan in F11.
- exit: C0 committed+pushed; C1 committed+pushed; C2 optional.
- status: CLOSED 10-22 (user decision: runtime preemption C0/C1 DROPPED
  => this item closed; twin-server data parallel also dropped). Plan in
  F11 preserved for the record. The B7 STARTUP-SIDE work (loud-fail on
  unfittable ctx/KV + closed --dev gate) is tracked separately as B7a
  and was APPROVED 10-22.

## B7a. Startup-side loud fail + closed --dev gate (APPROVED 10-22)
- what: (1) at startup, LOUDLY fail (clear error, not silent degrade)
  when the requested ctx/KV does not fit the device; (2) closed --dev
  gate: reject unknown/invalid device names instead of silently
  ignoring. This is the class of failure F16 (silent bad placement ->
  GDN fused disabled) was.
- why: user approved 10-22; directly prevents silent mis-placement
  like F16. Small and independent of B12.
- how: locate the fit/placement + device-parsing sites (common.cpp
  common_init_* / common_fit; -dev parse in common-params), add the
  loud error paths. Read the existing code first; keep it minimal.
- exit: a bad ctx/KV and a bad --dev each produce a clear startup error
  (no silent degrade); committed+pushed.
- status: OPEN, APPROVED 10-22 (B12 has landed; next candidate in this
  fork). NOTE 10-25: the runtime KV work shipped (F31, 6a30f9157 +
  db5cf4d80) is additive and is NOT B7a - B7a remains the startup-side
  loud-fail + closed --dev gate.

## B8. PQ2_0 ctx sweep (fit auto)
- what: standalone PQ2_0 load WITHOUT -c (fit auto): read computed ctx +
  load time; then run pq2-ctx-sweep.ps1.
- why: validate fit-auto behavior + complete the ctx sweep matrix.
- how: service down; setvars inside the script (already fixed).
- exit: sweep table appended to script comments / report.
- status: BLOCKED (service down required).

## B9. Final report
- what: single report with every open/closed item (list in PLAN.md step 18).
- why: user asked for a complete accounting.
- exit: report delivered.
- status: OPEN (last).

## B10. Small follow-ups (low priority)
- 58186 re-spawn with new binary if user asks (F7).
- Blocker B (ngl=0 comparison basis): user choice (a)/(b)/(c) - PLAN 15.
- Stray artifacts delete decision - PLAN 16.
- SYCL docs (user-facing) - PLAN 17 (must use the corrected 2-GPU facts,
  F9; the old "never use SYCL1" wording is WRONG).
- SYCL1 phantom confirmation - CLOSED 10-19: SYCL1 is a real second GPU
  (F9). Do not act on the old wording.
- User's own config fix E (remove -c from preset) - NOT agent-tracked.

## B11. PTQ1_0 SYCL kernel speedup (F14)
- what: port the CUDA byte-parallel trit decode + dp4a accumulation into
  SYCL vec_dot_ptq1_0_q8_1 (ggml-sycl/vecdotq.hpp:1068), plus a fused
  multi-column variant for the _ncols path (mirror ggml-cuda/mmvq.cu:719).
- why: PTQ1_0 TG 6-9 t/s vs Q6_K 13-18 t/s despite ~3.8x fewer weight
  bytes; the kernel is ALU/latency bound (scalar base-3 decode into a
  local int8 q[128], 128 scalar MACs, no dp4a).
- how: F14 fix sketch; verify with the standalone pp/tg repro (N7)
  before/after; keep the existing host unit test green.
- exit: PTQ1_0 standalone tg128 materially above the 6.90 baseline
  (target: close most of the per-byte gap to Q6_K); committed+pushed.
- status: DONE 10-20 (commit 5b80a3bb1, pushed). CUDA byte-parallel
  decode + fused multi-column dot ported (vecdotq.hpp:1075/:1175,
  mmvq.cpp:1423-1537). Standalone tg128 6.90 -> 26.72 t/s (3.9x), pp512
  unchanged (293.75). Host tests T1-T5 + element map + cuda-dot: 0
  mismatches; deterministic SYCL vs CPU completion text-identical.
  Test-only store-index bug found+fixed in lockstep (F14 debugging
  note).

## B12. Make ternary (PTQ1_0/PQ2_0) `-sm tensor` compatible (10-22 LEAD)
- what: fix the deterministic PQ2_0 tensor-mode assert
  `GGML_ASSERT(split_state.ne[j] % div == 0)` (ggml-backend-meta.cpp:1086,
  warmup forward, block 0, GDN `build_lora_mm(ssm_out)` hadamard perm).
  Root cause + the chosen fix design are in F20 (root cause), F21 (meta
  model), F22 (H-B dead), F23 (H-A lead fix), F24 (codecs).
- why: user wants ternary models to run in tensor mode in THIS fork, as
  the prerequisite for porting the ternary work to the upstream clone
  (B13). The perm reshape on the GDN output is unrepresentable at any
  even split; H-A folds the perm+signs+rot into the ssm_out weight at
  load time so the graph skips the perm entirely.
- how (F23, APPROVED by user 10-22, implement in 3nary first then port):
  (a) `llama_model_loader` pre-transform callback member + load-loop
      branch (force read_buf path for target tensors, skip normal set,
      no-op when unset);
  (b) model-side transform: dequant PQ2_0 -> f32, per-1024-row FWHT
      butterfly (W8 x W128), gather 128-row blocks grouped -> tiled
      (t_b = 3*(b%16) + b/16) with per-row sign flip, requant via
      EXISTING quantize_row_pq2_0_ref, ggml_backend_tensor_set with the
      EXISTING {2048,3} segment table; gated on tensor mode +
      hadamard_gdn_v_grouped + `.ssm_out.` names;
  (c) `bool folded = false;` on llama_hadamard_transform (llama-graph.
      h:23-32) + build_lora_mm (llama-graph.cpp:1546-1603) skips the
      whole transform block when folded.
  Then rebuild (N3) + verify with N14 (assert gone; SPLIT_STATE for the
  ssm_out chain = {axis 0, 2048x3}; K pairing consistent); test PQ2_0
  THEN PTQ1_0 in tensor mode; refresh private/ + commit+push.
- exit: PQ2_0 AND PTQ1_0 27B run `-sm tensor` on SYCL0,SYCL1 with no
  assert and sane output (expect ~sqrt(2)x quant error from the extra
  requant pass); committed+pushed; private/ updated.
- status: DONE 10-23, committed 7cb134632 (pushed; 7 src files + the
  tests/test-hadamard-fold.cpp unit test). H-A as designed: load-time
  fold of the 128 axis-split weights (48 ssm_out + 64 ffn_down + 16
  attn_output); the ssm_out gather bug was found + fixed on the way
  (F27). PQ2_0 AND PTQ1_0 27B verified in tensor mode, no assert, sane
  output; formula pinned by the unit test. H-C (kernel fusion) remains
  the documented fallback (F29) - adopt only if the fold's requant
  noise shows up in real workloads.

## B13. Port ternary work to the upstream clone (C:\Users\admin\source\ai\llama.cpp)
- what: port the tensor-compatible ternary kernels + tensor-mode support
  from THIS 3nary fork into the recent UPSTREAM clone the user wants to
  use as the forward working copy (F25).
- why: user (10-22) will keep the local copy for us and "create a true
  fork" later if needed; working on the clone avoids fork-inconvenience
  now. The 3nary fork is far behind upstream (F12), so the port delta is
  large and UNMEASURED.
- how (RECON FIRST, no porting until recon is done + user confirms scope):
  1. diff upstream ggml-backend-meta.cpp vs the 3nary meta backend
     (the 4th/tensor split mode) - they DIVERGED;
  2. confirm upstream ggml type enum values for PQ2_0/PTQ1_0 (upstream
     ternary types; numeric values likely differ from the 3nary fork -
     F12 never-mix rule);
  3. map which 3nary work is missing/divergent in the clone: ternary
     SYCL/CUDA kernels (F14/B11), GDN hadamard builder, prism hadamard,
     tensor-mode options, the H-A fold (B12) once landed;
  4. only then stage the port commit-by-commit.
- exit: recon notes written to private/ (F25 expansion); user confirms
  port scope; port staged.
- status: OPEN. (a) UNBLOCKED 10-23: B12 landed in this fork
  (7cb134632) - the port source is the FIXED ternary, not the asserting
  one. (b) recon NOT started (read-only: the clone hosts the live
  service).

## B14. Twin-server + router (data parallel, DEFERRED - draft pending user)
- what: two llama-server children, one pinned -dev SYCL0 and one
  -dev SYCL1 (one physical card each, -fit off + explicit -ngl, no
  cross-card split), fronted by a router that aggregates /v1/models and
  routes by session (LCP-style stickiness). Registry config shape: one
  entry per child (binary, device pin, -ngl, port, model); per-child
  budget via the fit.cpp projection (F31 math); idle-offload +
  pin/lock semantics so a cold child can be parked and a hot one kept.
- why: the single-server -np 2 layout pays the KV wall at one shared
  pool (F31): the second slot's growth is capped by the first slot's
  footprint. Two pinned single-card children isolate a wall hit to one
  child and double effective KV under the same total VRAM. The original
  twin-server data-parallel plan was dropped 10-22 (B7) as the
  PREEMPTION vehicle; it is revived 10-24 as the CAPACITY vehicle,
  additive to F31 (no-crash + auto-size), not a replacement.
- how: DRAFT ONLY - nothing approved, no code written. Grounding still
  owed before the draft is presented: (1) the per-backend free-memory
  surface fit.cpp can see under a device pin; (2) build placement
  (tools/CMakeLists.txt; unified-binary subcommand vs a new tool);
  (3) router location (new tool in this tree vs macube-side). Present
  to the user; implement only on explicit confirmation.
- exit: user confirms or drops; if confirmed, recon notes to private/
  then staged implementation.
- status: DRAFT, PENDING USER CONFIRMATION (10-25).
