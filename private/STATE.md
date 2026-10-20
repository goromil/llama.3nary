# STATE - snapshot (refresh at each milestone)

Last refresh: 2026-10-20. B11 DONE (commit 5b80a3bb1, pushed): PTQ1_0
SYCL byte-parallel decode + fused multi-column dot; standalone tg128
6.90 -> 26.72 t/s; F14 RESOLVED. Remaining threads: B7 (F11 C0->C1),
F12 hygiene.
No llama service listening as of 10-20 03:40 (61850 stopped by user
10-19; 52510 - UD-Q5_K_S - down too).

## Environment

- Machine: Windows, AMD Ryzen 9 5950X 16C, 64 GB RAM.
- GPU: TWO physical Intel Arc Pro B70 32 GB GPUs (SYCL0 + SYCL1) + 64 GB RAM.
  61850 spreads across both (~21 GB + ~31 GB in use at last check).
  GGML_SYCL_DEVICE is a NO-OP in this fork (user's shell var used with -mg;
  fork never reads it). Verified pin options (F9): `-dev SYCL0` (name-based)
  or `--split-mode none -mg 0` (-mg is a GPU-only index, CPU excluded).
  Default split mode = layer => both GPUs used (61850 runs that way on purpose).
- Build tree: `build/` (Ninja, Release, shared libs, tests ON).
- Install tree (macube services run from here): `C:\Users\admin\.local\3nary-sycl\`.
- Workspace: working dir `C:\Users\admin\source\ai\llama.3nary`,
  workspace root `C:\Users\gorom\source\ai\llama.3nary` (same tree).
- Macube: `C:\Users\gorom\source\macube\` (DO NOT EDIT its config files).

## Services (macube)

- 61850 (Qwen3.8-27B-UD-Q5_K_S + mtp draft, model 568492, router 421716) was
  STOPPED by the user 10-19 (port down, PIDs gone). It was healthy right before:
  MTP tg 15.5-15.6 t/s on task 42323 (~2.2x the non-MTP 6.90 baseline).
  User runs it when wanted; do not pressure for it to stay up.
- After the stop, install dir was deployed with the NEW binaries (see Binaries).
- 52510 was Ternary-Bonsai-2-27B-PTQ1_0 (n_ctx_slot 131072, kv_unified,
  4 slots); the user switched it BACK to UD-Q5_K_S 10-19. Ternary numbers
  from its log are historical (N10): pp ~300 t/s, tg 6.59 -> 6.14 -> 7.07
  -> 9.24 tracking KV size (F14). Do not touch; user manages it.

DEAD instances (tracked, see FINDINGS.md F11/F12):
- 58458 gemma-4-26B-A4B-it-UD-Q5_K_XL (MTP draft) - clean device KV OOM 960 MiB under 61850 pressure, status 1.
- 58186 Ternary-Bonsai-2-27B-PTQ1_0 - status -1073740791 (0xC0000409) ~52 s after spawn, cause not in logs.
- 18999/18998 - our assert-repro ports, long gone.

## Git

- branch `main`, pushed HEAD: `5b80a3bb1` (PTQ1_0 SYCL byte-parallel
  kernel, B11), `e844b224a` (private library reopen: F14, F11 C1/C2
  plan, F12 actions, F5 status), `11c3d328a` (project closure docs),
  `7faf42ec4` (private/ refresh: 2x B70 GPUs, device index semantics,
  maxver rerun), `0589b1e1a` (MTP 3-part fix), `02af668bf` (private/
  knowledge library), `355b4a2b3` (KV diagnostic), `a3f36885c` (KV guard),
  `033a6f0ad` (PTQ1_0 prefill), `bd993a9d7` (PTQ1_0 SYCL kernel).
- Working tree: clean except user's file, do not touch:
  `tools/ui/src/lib/stores/models/index.svelte.ts`.
  Stray test artifacts (test-pq2_0-sycl-dot.obj/.exe) DELETED 10-19 on
  user command.
- B11 full build 10-20 03:20: ninja log shows 300/300 tasks done, zero
  errors. The batch wrapper reported exit 1 - ARTIFACT (same class as
  the %ERRORLEVEL% misreport); verify binary mtimes, not the wrapper
  code.

## Binaries

- `build\bin`: B11 build (10-20 03:20): byte-parallel PTQ1_0 kernel +
  all 10-19 fixes (MTP 3-part, KV guard + diagnostic, PTQ1_0 prefill).
- install dir (`C:\Users\admin\.local\3nary-sycl\`): 10-19 build
  (SHA256-verified 89/89 at deploy), BEHIND build\bin by the B11 commit
  (5b80a3bb1) - has the MTP fix + KV guard + prefill, NOT the B11
  kernel. Copy build\bin when the user restarts a service.
- Service restarts via macube will now run the MTP fix + KV guard + KV
  diagnostic (+ B11 kernel once redeployed).

## Models

| Model | Path | Notes |
|---|---|---|
| Qwen3.8-27B-UD-Q5_K_S (target) | `E:\DataVolume\Models\Qwen3.8-27B-UD-Q5_K_S\Qwen3.8-27B-UD-Q5_K_S.gguf` | 17801 MiB, HAS blk.64 nextn tensors (MTP fallback cannot trigger naturally). qwen35 arch. |
| Qwen3.8 MTP draft | same dir `\mtp-Qwen3.8-27B-Q4_0.gguf` | 1306.10 MiB. |
| gemma-4-26B-A4B-it-UD-Q5_K_XL | `E:\DataVolume\Models\gemma-4-26B-A4B-it-UD-Q5_K_XL\` | target + `mtp-gemma-4-26B-A4B-it.gguf` draft. arch likely gemma4-assistant. |
| Ternary PTQ1_0 (PREFERRED test model) | `E:\DataVolume\TernaryModels\Ternary-Bonsai-2-27B-PTQ1_0\` | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` 5671 MB + `...-mmproj-BF16.gguf` 888 MB. No MTP draft. |

## Logs (ours, in $env:TEMP\opencode\)

| File | Content |
|---|---|
| mtp-server-98304.log | 8-line assert repro (build\bin, draft+spec, -lv 3). Ends at "loading model", then ggml.c:1661 assert. |
| install-server-98304.log | same assert, install-dir binary + --numa distribute. |
| mtp-server-verbose.log | 9.7 MB, NO-draft run at max verbosity (2147483647). :21 loading model, :22 local path, :23 fitting params, :2442 "no change", :135601 fit done 28.32 s, :137874 SYCL-MEMTRACE 1536 MiB FAILED (final real KV OOM). |
| mtp-server-98304-maxver-err.log | 122504 lines, draft+spec max-verbosity rerun (10-19, warm cache). NO assert. Fit ~37 iters in ~1.6 s, then HUNG 118 s at sched_reserve in fit's final measurement ctx (tail layers 61-65 on SYCL1). Killed at 120 s. See F1. |
| ternary-d-err.log | 27 lines, full ternary standalone E2E (PASS). |
| ptq10-b11-bench.log | B11 after-bench (10-20): PTQ1_0 SYCL ngl 99 pp512 293.75 / tg128 26.72 (old kernel: 296.39 / 6.90). |
| ptq10-e2e2.log | B11 E2E rerun (10-20): comp-sycl exit 0, comp-cpu exit 0, TEXT: IDENTICAL (192 tok deterministic, answers correct). |

Macube logs:
- `C:\Users\gorom\source\macube\logs\llama-error.log` (~4931 lines at snapshot):
  router srv lines + spawn args + "exited with status" ONLY.
  Model-server stderr is NOT here.
  - 61850 spawn args: :4697-4740+
  - 58458 spawn args: :4789-4873+
  - 58186 spawn args: :4857-4914, exit line :4915
- `C:\Users\gorom\source\macube\logs\llama.log`: srv-target lines only.
  52510 ternary run starts line 61815 (PTQ1_0, n_slots=4, n_ctx_slot
  131072, kv_unified, pp ~300 t/s).

## Next (10-20, after B11)

1. B7: Option C C0 then C1 per the F11 plan (C0 = error only the failing
   slot on the decode-fail path server-context.cpp:3590-3639, TODO :3595;
   C1 = preempt the largest-KV generating slot).
2. F12: divergence hygiene per the F12 action list.
3. If the user restarts a service: deploy build\bin (B11) to the install
   dir first - it is behind by 5b80a3bb1.
