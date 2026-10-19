# STATE - snapshot (refresh at each milestone)

Last refresh: 2026-10-19, macube router clock ~103 min at last check.

## Environment

- Machine: Windows, AMD Ryzen 9 5950X 16C, 64 GB RAM.
- GPU: Intel Arc Pro B70 32 GB. Usable VRAM ~29.3-29.5 GiB.
- SYCL device_info shows TWO entries: SYCL0 (31906 MiB) and SYCL1 (31906 MiB,
  684 MiB free at last check) plus CPU. SYCL1 is almost certainly the same
  physical GPU enumerated twice (or a partition) - never select it.
  Always: `$env:GGML_SYCL_DEVICE = '0'`.
- Build tree: `build/` (Ninja, Release, shared libs, tests ON).
- Install tree (macube services run from here): `C:\Users\admin\.local\3nary-sycl\`.
- Workspace: working dir `C:\Users\admin\source\ai\llama.3nary`,
  workspace root `C:\Users\gorom\source\ai\llama.3nary` (same tree).
- Macube: `C:\Users\gorom\source\macube\` (DO NOT EDIT its config files).

## Services (macube) - ALIVE

| Port | Instance | PID | Notes |
|---|---|---|---|
| 61850 | Qwen3.8-27B-UD-Q5_K_S (target + mtp draft) | model 568492, router 421716 | ctx 131072, OLD install binary. MTP healthy (task 12627, tg 19.70 t/s). DO NOT KILL. All VRAM-headroom tests need the user to stop this. |

DEAD instances (tracked, see FINDINGS.md F11/F12):
- 58458 gemma-4-26B-A4B-it-UD-Q5_K_XL (MTP draft) - clean device KV OOM 960 MiB under 61850 pressure, status 1.
- 58186 Ternary-Bonsai-2-27B-PTQ1_0 - status -1073740791 (0xC0000409) ~52 s after spawn, cause not in logs.
- 18999/18998 - our assert-repro ports, long gone.

## Git

- branch `main`, last pushed HEAD: `355b4a2b3` (KV guard + diagnostic).
- Working tree (at snapshot time):
  - EDITED, uncommitted: `common/speculative.cpp` + `tools/server/server-context.cpp`
    = MTP 3-part fix (all applied, built OK at 9:43 AM, then ONE `\n` edit at
    server-context.cpp:1100 AFTER the build => rebuild pending).
  - user's file, do not touch: `tools/ui/src/lib/stores/models/index.svelte.ts`
  - untracked strays pending user delete decision:
    `test-pq2_0-sycl-dot.obj`, `tests/test-pq2_0-sycl-dot.exe`
  - NEW: `private/` knowledge library (this folder) - to be committed+pushed.

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
| ternary-d-err.log | 27 lines, full ternary standalone E2E (PASS). |

Macube logs:
- `C:\Users\gorom\source\macube\logs\llama-error.log` (~4931 lines at snapshot):
  router srv lines + spawn args + "exited with status" ONLY.
  Model-server stderr is NOT here.
  - 61850 spawn args: :4697-4740+
  - 58458 spawn args: :4789-4873+
  - 58186 spawn args: :4857-4914, exit line :4915
- `C:\Users\gorom\source\macube\logs\llama.log`: srv-target lines only.
