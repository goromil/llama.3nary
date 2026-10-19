# STATE - snapshot (refresh at each milestone)

Last refresh: 2026-10-19. PROJECT CLOSED by user decision: "everything is
cleared and fixed, all objectives have got met". See BACKLOG.md banner for
the disposition of every item.

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

## Services (macube) - ALIVE

| Port | Instance | PID | Notes |
|---|---|---|---|
| 61850 | Qwen3.8-27B-UD-Q5_K_S (target + mtp draft) | model 568492, router 421716 | ctx 131072, OLD install binary, both GPUs (by design). MTP healthy: long gen task 42323 n_gen 3400+, tg 15.5-15.6 t/s (~2.2x the non-MTP 6.90 baseline) at router clock 143+ min. DO NOT KILL. All VRAM-headroom tests need the user to stop this. |

DEAD instances (tracked, see FINDINGS.md F11/F12):
- 58458 gemma-4-26B-A4B-it-UD-Q5_K_XL (MTP draft) - clean device KV OOM 960 MiB under 61850 pressure, status 1.
- 58186 Ternary-Bonsai-2-27B-PTQ1_0 - status -1073740791 (0xC0000409) ~52 s after spawn, cause not in logs.
- 18999/18998 - our assert-repro ports, long gone.

## Git

- branch `main`, pushed HEAD: `7faf42ec4` (private/ refresh: 2x B70 GPUs,
  device index semantics, maxver rerun), `0589b1e1a` (MTP 3-part fix),
  `02af668bf` (private/ knowledge library), `355b4a2b3` (KV diagnostic),
  `a3f36885c` (KV guard), `033a6f0ad` (PTQ1_0 prefill), `bd993a9d7`
  (PTQ1_0 SYCL kernel).
- Working tree: clean except
  - user's file, do not touch: `tools/ui/src/lib/stores/models/index.svelte.ts`
  - untracked strays pending user delete decision:
    `test-pq2_0-sycl-dot.obj`, `tests/test-pq2_0-sycl-dot.exe`
- MTP fix: rebuilt after the \n edit (build exit 0, 10-19), committed+pushed.

## Binaries

- `build\bin`: 9:43 AM MTP build + pending \n edit rebuild.
  `ggml.dll`, `ggml-base.dll`, `ggml-sycl.dll`, `ggml-cpu.dll`, `llama.dll`
  hash-identical build\bin vs install; ONLY `llama-common.dll`,
  `llama-server-impl.dll`, `llama-cli-impl.dll` differ. No PDBs anywhere.
- install dir: OLD 2:17 AM binaries (what 61850/58458/58186 ran).
  6 files to copy after 61850 is stopped by user:
  llama.dll, llama-server-impl.dll, llama-server.exe,
  llama-cli-impl.dll, llama-cli.exe, llama.exe.

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
