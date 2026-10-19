# private/ - knowledge library (llama.3nary fork)

Durable knowledge base for the llama.3nary fork work. Written so any future
session (human or agent) can restart implementation with zero loss of
acquired knowledge. This folder is meant to live in the private GitHub repo
only. It is NOT user-facing product docs.

## Files

| File | What it is |
|---|---|
| STATE.md | Snapshot: environment, services, ports, PIDs, binaries, git, log locations. Refresh at each milestone. |
| FINDINGS.md | Research findings with evidence and root causes. Each finding has confidence + open questions. |
| PLAN.md | Phased plan with exact commands. Start at "IMMEDIATE NEXT STEPS". |
| BACKLOG.md | Open items, each with: what / why / how to execute / exit criteria / status. |
| NOTES.md | Constraints, recipes (build/setvars/launch), mappings, log semantics, pitfalls, perf baselines. |

## How to restart work

1. Read NOTES.md first (environment + constraints + recipes).
2. Read STATE.md (what is alive, what is built, what is committed).
3. Read the "IMMEDIATE NEXT STEPS" block in PLAN.md and execute it.
4. Deep-dive findings in FINDINGS.md only when a task needs them.
5. Work BACKLOG.md items in the listed order unless the user redirects.
6. After each milestone: update STATE.md (timestamp), check off BACKLOG.md,
   append anything new to FINDINGS.md. Keep this folder in git.

## Ground rules (from user, never violate)

- User stops/starts macube services. Never kill a PID the user owns.
  Killing PIDs of processes WE spawned is fine.
- No changes to macube service files (services.json, presets). Reading OK.
- `llama-api-keys.txt` untouched.
- All SYCL binaries: oneAPI setvars first (recipe in NOTES.md), pin
  `$env:GGML_SYCL_DEVICE = '0'`, never use SYCL1 (see FINDINGS.md F9).
- Small changes, ASCII only in code/commits. Trailer:
  `Assisted-by: Claude, Qwen3.8-27B`.
- For testing prefer the ternary model (small, fits scarce free VRAM):
  `E:\DataVolume\TernaryModels\Ternary-Bonsai-2-27B-PTQ1_0\Ternary-Bonsai-2-27B-PTQ1_0.gguf`
  + `-mm E:\DataVolume\TernaryModels\Ternary-Bonsai-2-27B-PTQ1_0\Ternary-Bonsai-2-27B-mmproj-BF16.gguf`
- Do not add tests/* files, do not submit PRs, do not reply to reviewers.
  This is a private fork; we commit directly.
