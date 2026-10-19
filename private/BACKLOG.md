# BACKLOG - open items (ordered)

Each item: what / why / how / exit criteria / status.

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
  draft model cannot load). Awaiting user OK to close; likely NO code change.

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
- status: BLOCKED.

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
- how: PLAN.md phase 2; locate slot state machine first (no
  server-slot.cpp in fork; it is in server-context.cpp + server-*.h).
  Upstream TODO server-context.cpp:3589-3590 = design reference.
- exit: C0 committed+pushed; C1 committed+pushed; C2 optional.
- status: OPEN (after B4/B5).

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
