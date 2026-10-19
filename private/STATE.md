# STATE - snapshot (refresh at each milestone)

Last refresh: 2026-10-19. CLOSED by user decision that day, then REOPENED
the same day (F5 status, F12 actions, F11 C1/C2 plan, ternary slowness).
New thread: F14 ternary kernel gap + B11 kernel rewrite plan.
Service 52510 (ternary) switched back to UD-Q5_K_S by the user.

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

- branch `main`, pushed HEAD: `11c3d328a` (project closure docs),
  `7faf42ec4` (private/ refresh: 2x B70 GPUs, device index semantics,
  maxver rerun), `0589b1e1a` (MTP 3-part fix), `02af668bf` (private/
  knowledge library), `355b4a2b3` (KV diagnostic), `a3f36885c` (KV guard),
  `033a6f0ad` (PTQ1_0 prefill), `bd993a9d7` (PTQ1_0 SYCL kernel).
- Working tree: clean except user's file, do not touch:
  `tools/ui/src/lib/stores/models/index.svelte.ts`.
  Stray test artifacts (test-pq2_0-sycl-dot.obj/.exe) DELETED 10-19 on
  user command.
- MTP fix: rebuilt after the \n edit (build exit 0, 10-19), committed+pushed.

## Binaries

- `build\bin`: final MTP build (includes \n edit), 89 files.
- install dir (`C:\Users\admin\.local\3nary-sycl\`): NEW binaries DEPLOYED
  10-19 after user stopped 61850 (89/89 files, SHA256-verified:
  llama.dll, llama-common.dll, llama-server-impl.dll, llama-server.exe,
  llama-cli-impl.dll, llama-cli.exe, llama.exe all match build\bin; full
  tree also matches). No PDBs anywhere.
- Service restarts via macube will now run the MTP fix + KV guard + KV
  diagnostic for the first time.

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

## Next (10-19, after reopen)

1. B11: PTQ1_0 SYCL vec_dot rewrite per the F14 fix sketch (needs user
   go-ahead; verify with the standalone pp/tg repro before/after).
2. B7: Option C C0 then C1 per the F11 plan (state machine located).
3. F12: divergence hygiene per the F12 action list.
