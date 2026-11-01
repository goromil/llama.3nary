# PLAN - updated plan (start at IMMEDIATE NEXT STEPS)

Status markers: [x] done, [ ] open, [B] blocked on user.

## IMMEDIATE NEXT STEPS (fresh session starts here, 11-01)

Context: B12 tensor-mode pivot (10-23, 7cb134632) and the KV wall two
stages (10-25, 6a30f9157 + db5cf4d80) are SHIPPED; B15 adaptive ctx +
B17 handover (10-27/28) SHIPPED. B19 (fit settle, F40) + B20 (two-pass
grow) + the tensor-mode draft fit are COMMITTED (c9c4e20f1) and built,
NOT deployed. B14 (twin-server + router) is REJECTED 10-28 by the user
(final, dead end) - the record is BACKLOG B14 + F39, no re-presentation.
11-01 probes closed the batch-6144 memory question (F40 addendum 1);
10-31 crash/hang forensics in F40 addendum 2.

1. [B] DEPLOY + verify (needs an 8080 stop window): rebuild build\bin
   (the 11-01 tensor draft fit is in llama-common only, the exes need
   re-linking), deploy to macube, then the deterministic live growth-
   WARN catch (B20) + a sanity run of llama-fit-params tensor + draft
   fit (real loads, ~2 min+, co-residency-sensitive).
2. [B] T1 cold test (stop window): PTQ+MTP COLD load decides the
   release-lag theory (F40 addendum 2; prediction: cold loads clean).
   Kit ready: E:\tmp\cold-test.ps1 + T1.bat/T1b.bat (atomic
   stop->test->start); result to E:\tmp\T1-result.txt.
3. [ ] B7a (next code item in this fork): startup loud-fail on
   unfittable ctx/KV + closed --dev gate. Target: the SPLIT_MODE_TENSOR
   fit path + the -sm/-dev parse area (re-locate lines, the 10-25
   numbers are stale). Note: B19 changed what "unfittable" means -
   over-budget now settles down, so B7a's fail surface is the settled
   floor + the pinned-ctx deficit report.
4. [ ] B13 recon (read-only, clone hosts live service): diff fork vs
   clone meta backend, ternary enum mapping, missing/divergent work
   map (F25), seq_rm question (F28). THEN port commit-by-commit, scope
   user-confirmed.
5. [ ] N17 Q5_K_S tensor bench (GDN correctness check first): still
   user-run; no free card while the service holds both - run co-
   resident (11-01 probe style, variance caveat) or in a stop window.
6. [B] bad_alloc repro: needs the user's real workload + the next
   occurrence with preserved log context (logs truncate, the catch
   site is lost). Proposals on the table: detached log-tail watcher
   (E:\tmp only) / top-level try-catch + stack trace. Await user call.
7. [B] TG-status line discrepancy: the user's display shows a TG line
   that -lv 2 should suppress (SLT_INF, server-context.cpp:594);
   blocked on a sample line from the user.

## 10-25 items - resolved / retired (reference)

- 1-4 [x] KV wall stage 1 (6a30f9157) + stage 2 -fitm (db5cf4d80) +
  SYCL=OFF root cause (F30) + private/ refresh: done, pushed.
- 5 [x] B14 PRESENT (twin-server + router draft): REJECTED 10-28 by
  the user (final: "I will never start. It was dead end."). Tensor
  split and layer split both work in multi-chat; the successor is the
  parked B18 (steady-state decode pipeline). Record: BACKLOG B14 +
  F39 + STATE. No re-presentation.
- 6 [x] REPORT to the user (9B MTP fix gone from tree, SYCL fix,
  stage-2 grow): folded into the continuous 10-26..11-01 docs sync;
  the tree fact re-verified 11-01 (src/models/qwen35.cpp still matches
  HEAD, the 9B MTP fix still absent).
- 7 [x] extraArgs bisection (-ngl 99 / -np 2 / -c 262144): outcome
  never reported; superseded - F39 settled the -np 2 question (drop
  it, user-confirmed 10-27/28) and the 10-31 forensics took over the
  crash/hang work. Nothing left to fold.
- 8: Q5_K_S tensor bench: carried to open item 5 above.

## PARKED (do not start without user re-scoping, 10-22)

- B7 preemption C0/C1/C2: CLOSED 10-22 (twin-server data-parallel
  design DROPPED; C0/C1 DROPPED per user). Plan intact in F11 for the
  record only.
- B7a (startup-side loud-fail on unfittable ctx/KV + closed --dev
  gate): APPROVED 10-22, NOT parked - open item 3 above.
- B5 free-VRAM suite / B6 Part-3 trigger / B8 PQ2_0 sweep / B9 final
  report / B10 small follow-ups: status as in BACKLOG.md (parked /
  open-low). The final report (B9) should include the tensor-mode work
  when it lands.
- B18 (alternating pipeline / steady-state decode): DRAFT parked,
  user-requested 10-28 (DESIGN-alternating-pipeline.md), no code.

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
  drafted - REJECTED 10-28 (final).
- [x] 11-01: B19 settle + B20 two-pass grow + tensor-mode draft fit
  committed (c9c4e20f1) + private sync with F40 addenda 1+2 and the
  B20 design doc (0ce10b479); -b/-ub probe matrix ran, batch-6144
  memory question answered (no code change needed); deploy pending
  the stop window (open item 1).
