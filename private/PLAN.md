# PLAN - updated plan (start at IMMEDIATE NEXT STEPS)

Status markers: [x] done, [ ] open, [B] blocked on user.

## IMMEDIATE NEXT STEPS (fresh session starts here)

DONE 10-19: 1. private/ pushed (02af668bf). 2. rebuild after \n edit
green. 3. MTP fix pushed (0589b1e1a). Fresh session continues at step 4.

4. [ ] TELL USER: Part-3 fallback gap - Qwen3.8 AND gemma-4 presets have
   has_draft=true, so MTP ctx failure stays fatal in service runs. Offer to
   extend the fallback to the has_draft case.
5. [ ] Assert window (F1): read common.cpp :1230-1460 (common_init_from_
   params / common_init_ entry, "fitting params" site, the two "failed to
   create context" sites :1398 + :1449; check if caller sets
   cparams.no_alloc before fit) AND server-context.cpp :1030-1090 (spec
   params setup between "loading model" and common_init_from_params).
   Update FINDINGS.md F1/F3 with the result.
6. [ ] Assert exculpation test: rerun draft+spec 98304 repro args (NOTES.md
   N7) + `--fit off`, port 18999, detached launch (N4), poll stderr.
   If assert persists => fit fully exculpated, window is in spec setup /
   common_init_ entry. Also do one repro at MAX verbosity (-lv 6 or
   2147483647) to capture the last lines before the assert.
7. [B] NEEDS USER: stop 61850 (PID 568492) to free VRAM. Then:
   - rerun draft+spec 98304 on FREE VRAM (assert persists => deterministic
     host-side defect; gone => host commit exhaustion under pressure)
   - standalone MTP E2E: 98304 draft+spec (verify "loading draft model" +
     MTP acceptance stats) and 262144 short gen (razor-thin fit)
   - copy the 6 files build\bin -> C:\Users\admin\.local\3nary-sycl\
     (llama.dll, llama-server-impl.dll, llama-server.exe,
     llama-cli-impl.dll, llama-cli.exe, llama.exe)
   - user starts service + runs 2-client repro; then grep
     C:\Users\gorom\source\macube\logs\llama.log for "kv cache stream" at
     next FAILED_PREPARE, expect `used == size` (KV-guard verification).
8. [ ] Part-3 fallback test: pick a trigger (model WITHOUT nextn tensors /
   forced OOM / skip with note).

## PHASE 2 - Option C preemption (staged, after assert + MTP are done)

9. [ ] Locate the slot state machine + abort/response-stream plumbing
   (server-context.cpp + server-*.h; server-slot.cpp does NOT exist).
10. [ ] C0: error only the failing slot on alloc failure. Build, commit,
    push.
11. [ ] C1: victim = largest active slot; prompt_clear + rebuild + retry
    failing slot; explicit "preempted" abort for the victim. Build, commit,
    push.
12. [ ] C2 (optional): silent resume.

## PHASE 3 - remaining items

13. [ ] PQ2_0 sweep: standalone load WITHOUT -c (fit auto) - read computed
    ctx + load time; then run pq2-ctx-sweep.ps1 (service down, setvars in
    script). Part A baselines are in the script comments.
14. [ ] 58186 follow-up (F7): only if user asks - re-spawn ternary service
    with NEW binary, capture its own stderr.
15. [ ] Blocker B (ngl=0 comparison basis) - awaiting user choice:
    (a) accept SYCL-vs-SYCL-host + host unit test, (b) separate
    GGML_SYCL=OFF build dir, (c) skip.
16. [ ] Stray artifacts test-pq2_0-sycl-dot.obj /
    tests/test-pq2_0-sycl-dot.exe - pending user delete decision.
17. [ ] SYCL docs (user-facing build/usage notes for the fork).
18. [ ] FINAL REPORT with all open items: E2E numbers incl. ternary "pong"
    run; KV root cause + fix + guard/diagnostic commits; pre-existing
    test-dspark-forward LNK2019; stray artifacts; blocker B; ggml-sycl
    NULL-copy crash; MTP razor-thin-fit finding; fused Gated Delta Net
    warning; has_draft fallback gap; ggml.c:1661 assert status + window;
    llama-gguf 0xC0000005 crash; 58458 clean-OOM tracking entry (spawn args
    + classification); 58186 unconfirmed exit.

## DONE (reference)

- [x] PTQ1_0 SYCL decode port + prefill: committed+pushed, E2E PASS.
- [x] KV bug root cause + guard a3f36885c + diagnostic 355b4a2b3:
  committed+pushed.
- [x] SYCL kernel bd993a9d7 + PTQ1_0 prefill 033a6f0ad: committed+pushed.
- [x] MTP 3-part fix: all edits applied, build green (rebuild after \n
  edit pending).
- [x] Assert repro matrix (3 runs): MTP edits exonerated twice.
- [x] Ternary standalone E2E: PASS, exit-code-1 mystery closed.
- [x] 58458 gemma-4 OOM chain: tracked + classified (clean device KV OOM
  under pressure, status 1, NOT the assert).
- [x] Fit loop resolved: explicit -c => n_ctx no-op, but 56-iteration ngl
  loop still runs (28.32 s, metadata loads only).
- [x] Verbosity mapping resolved: -lv 3 = INFO; SRV_TRC invisible there;
  assert window corrected to :994 -> "fitting params" (F1).
- [x] pq2-ctx-sweep.ps1 fixed.
