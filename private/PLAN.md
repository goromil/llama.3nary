# PLAN - updated plan (start at IMMEDIATE NEXT STEPS)

Status markers: [x] done, [ ] open, [B] blocked on user.

## IMMEDIATE NEXT STEPS (fresh session starts here, 10-25)

Context: the 10-22 tensor-mode pivot is COMPLETE (B12 DONE 10-23,
7cb134632). The 10-24/10-25 work was the KV wall, shipped in two
stages (both user-approved 10-24: "start both" - no-crash first, then
auto-size; grow step 16K tokens, no further sizing discussion). The
SYCL build mystery was root-caused + fixed in between (F30). B14
(twin-server + router) is drafted, pending the user.

1. [x] KV wall stage 1: no-crash at the KV wall (6a30f9157, pushed).
   n_batch == 1 KV exhaustion ends ONLY the failing slot (clean
   EXCEED_CONTEXT_SIZE to that client); the others keep generating.
   decode() bool -> int (file-local decode_result enum). F31.
2. [x] KV wall stage 2: fit-time auto-size -fitm (db5cf4d80, pushed).
   Opt-in -fitm / --fit-ctx-max grows a pinned -c into free device
   memory in 16K-token steps up to the cap, before the margin check;
   no-op when absent. Verified on-device: 32768 -> 65536 grow with
   margin pass on both cards; over-budget abort; host-only skip. F31.
3. [x] SYCL build finding (F30): the SYCL=OFF OOM-bypass builds were
   CPU-only runtimes, no visible error. Fixed: one-time -DGGML_SYCL=ON
   reconfigure + ggml-sycl TU rebuild at -j 1 + downstream (all
   green); build-3nary.bat permanently SYCL=ON.
4. [x] private/ refresh (F30 + F31 + "for consideration"; BACKLOG
   B12 DONE / B13 unblock / B7a note / B14 new; STATE full refresh;
   PLAN here). Commit+push private/ ONLY.
5. [ ] PRESENT: the B14 twin-server + router draft (registry config
   shape, -dev SYCL0/SYCL1 pins, aggregated /v1/models, budget via
   fit.cpp projection, idle-offload + pin/lock) for user confirmation.
   Grounding owed first: per-backend free-memory surface under a pin,
   build placement (tools/CMakeLists.txt; unified-binary subcommand vs
   new tool), router location.
6. [ ] REPORT to the user: qwen35.cpp 9B MTP fix is GONE from the
   tree (working file matches HEAD); the SYCL finding + fix; stage-2
   grow verification (SHAs 6a30f9157, db5cf4d80).
7. [B] Await the user's extraArgs bisection outcome (-ngl 99 / -np 2 /
   -c 262144; prediction table in F31) - fold into FINDINGS.
8. [B] Await a free card (re-derive service VRAM first, PID 212452)
   -> run the Q5_K_S tensor bench (N17; GDN correctness check first).
9. [ ] B7a (next code item in this fork): startup loud-fail on
   unfittable ctx/KV + closed --dev gate. Target:
   common/fit.cpp:182-183 SPLIT_MODE_TENSOR fit throw + the -sm/-dev
   parse area. The 10-25 runtime KV work is NOT B7a.
10. [ ] B13 recon (read-only, clone hosts live service): diff fork vs
    clone meta backend, ternary enum mapping, missing/divergent work
    map (F25), seq_rm question (F28). THEN port commit-by-commit,
    scope user-confirmed.

## PARKED (do not start without user re-scoping, 10-22)

- B7 preemption C0/C1/C2: CLOSED 10-22 (twin-server data-parallel
  design DROPPED; C0/C1 DROPPED per user). Plan intact in F11 for the
  record only.
- B7a (startup-side loud-fail on unfittable ctx/KV + closed --dev
  gate): APPROVED 10-22, NOT parked - implement after B12 lands (same
  fork; see BACKLOG B7a).
- B5 free-VRAM suite / B6 Part-3 trigger / B8 PQ2_0 sweep / B9 final
  report / B10 small follow-ups: status as in BACKLOG.md (parked /
  open-low). The final report (B9) should include the tensor-mode work
  when it lands.

## DONE (reference)

- [x] 10-19/10-20: KV ring overcommit guard + diagnostic (a3f36885c,
  355b4a2b3); MTP 3-part fix (0589b1e1a); PTQ1_0 SYCL byte-parallel
  kernel B11 (bd993a9d7 + 033a6f0ad + 5b80a3bb1, tg128 6.90 -> 26.72);
  fit-params draft-aware fix (48e9082ca); private/ knowledge library
  (02af668bf, e844b224a, 11c3d328a, 7faf42ec4, 835d728b3).
- [x] 10-22: tensor-mode root cause pinned (F20), meta model mapped
  (F21), H-B dead (F22), H-A designed (F23), PQ2_0/PTQ1_0 codec facts
  (F24), port-target clone located (F25). F15-F18 reconstructed/retained.
- [x] 10-22 (evening): user decisions recorded - H-A APPROVED (3nary
  first), B7a APPROVED, B7 preemption CLOSED; clone git state verified
  (no local commits, 3 user mods to preserve). Docs: BACKLOG (B7a),
  STATE, PLAN refreshed.
- [x] 10-23: B12 H-A fold implemented + verified + committed
  (7cb134632) + unit test (tests/test-hadamard-fold.cpp); ssm_out
  gather bug found + fixed (F27); PQ2_0 + PTQ1_0 tensor-verified;
  F29 (H-C kernel fusion) documented as fallback.
- [x] 10-25: KV wall two stages - no-crash at the wall (6a30f9157) +
  fit-time auto-size -fitm (db5cf4d80), both pushed, grow verified
  on-device (F31); SYCL=OFF = CPU-only runtime root-caused + fixed,
  build recipe permanently SYCL=ON (F30); B14 (twin-server + router)
  drafted, pending user.
