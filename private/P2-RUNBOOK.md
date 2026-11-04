# P2 RUNBOOK - DeepSeek-V4.1-Flash-1017 ternary A/B (11-03)

Status: staged - load test done on this box (11-04, F43),
the A/B itself still pending. Nothing here touches the live
service until step 1. Step 0 runs against the running 8080
service and is read-only (7 short prompts, sequential, ~5-10
min at the baseline t/s). The stop window is step 1 only,
one user action.

## Locked decisions (11-02)

- A: the engram table is ternarized with the backbone
  (PTQ1_0 / PQ2_0), no imatrix, plain PTQ.
- B: both variants are validated.
- C: 9-shard split kept; shard 1 = KV/meta, 2-7 = 20 experts
  each, 8 = table, 9 = misc.
- D: dspark dflash hunk (P0, done).
- Text-only A/B: no mmproj, no DSpark drafter (files not
  local), no spec.

## Files

- PTQ1_0: E:\DataVolume\Models\DeepSeek-V4.1-Flash-1017-PTQ1_0-
  00001-of-00009.gguf (+ -00002..00009, 164.2 GB, 1.75 bpw,
  verified 11-02, F42).
- PQ2_0:  E:\DataVolume\Models\DeepSeek-V4.1-Flash-1017-PQ2_0-
  00001-of-00009.gguf (+ -00002..00009, 199.3 GB, 2.125 bpw,
  verified 11-02, F42).
- Baseline: live macube service, UD-Q4_K_XL 501.68 GB,
  router pid 511116 @ 0.0.0.0:8080 (recorded 774dbb6e3;
  re-derive via Get-NetTCPConnection -LocalPort 8080 before
  touching anything; child was 141080 @ 127.0.0.1:49802).
- Binary: build\bin\llama-server.exe, rebuilt on the other
  machine from 75138825b (or later). A pre-F43 build
  (774dbb6e3 + docs only) has the full port but crashes on
  any DSV41 load (deterministic NULL-buffer assert from the
  missing final graph expand, F43).
- A/B capture: %TEMP%\opencode\ab-dsv41.ps1, copied to
  E:\tmp\ab-dsv41.ps1 at run time; outputs to
  E:\tmp\ab-dsv41\{baseline,ptq1,pq2}\.
- Mini (load test, F43): E:\DataVolume\Models\DeepSeek-
  V4.1-Flash-1017-mini-N1.gguf, 8,815,791,424 B; kit for
  the other machine (tool source + build.bat + ab-dsv41.ps1
  + runner + inv.txt + README) staged at E:\tmp\dsv41mini.

## Launch lines (flags verified against common/arg.cpp +
## common.h + llama-context.cpp)

Defaults in force (no flag needed): KV F16/F16 (DSV41
requires type_k == type_v, llama-context.cpp:3944),
flash-attn AUTO (NOT force-enabled under layer split,
:3933), --jinja on (common.h use_jinja), --fit on with a
1 GiB/device margin (common.h fit_params), --load-mode auto
= mmap (arg.cpp:2693), -sm layer, -np 1, -t auto (pinned
16 here). -c 32768 is pinned explicitly. -ot syntax is
<pattern>=<buffer type> (arg.cpp:2743).

PTQ1_0 (port 58900, localhost only, no API key):

    build\bin\llama-server.exe ^
      -m "E:\DataVolume\Models\DeepSeek-V4.1-Flash-1017-PTQ1_0-00001-of-00009.gguf" ^
      -p 58900 -c 32768 -np 1 -ngl 22 -t 16 -tb 16 ^
      -ot per_layer_token_embd.weight=CPU

PQ2_0 (port 58901):

    build\bin\llama-server.exe ^
      -m "E:\DataVolume\Models\DeepSeek-V4.1-Flash-1017-PQ2_0-00001-of-00009.gguf" ^
      -p 58901 -c 32768 -np 1 -ngl 18 -t 16 -tb 16 ^
      -ot per_layer_token_embd.weight=CPU

Launch detached (background shells die on opencode
restarts; Start-Process survives - proven on the PQ2_0
quantize, pid 742304):

    $out = "$env:TEMP\opencode"
    Start-Process -WindowStyle Hidden -PassThru `
      -FilePath "C:\Users\admin\source\ai\llama.3nary\build\bin\llama-server.exe" `
      -ArgumentList '-m','E:\DataVolume\Models\DeepSeek-V4.1-Flash-1017-PTQ1_0-00001-of-00009.gguf','-p','58900','-c','32768','-np','1','-ngl','22','-t','16','-tb','16','-ot','per_layer_token_embd.weight=CPU' `
      -WorkingDirectory "C:\Users\admin\source\ai\llama.3nary" `
      -RedirectStandardOutput "$out\ab-ptq1.out" `
      -RedirectStandardError  "$out\ab-ptq1.err"
    # poll ab-ptq1.err for "server is listening", watch
    # Get-Process for OOM; no completion notification, so
    # poll the log, do not wait for the harness

KV at -c 32768 (llama-kv-cache-dsv4.{h,cpp} + the shard-1
KV dump, 92 keys): raw ISWA window K 128 x 40 x 512 F16
~5 MB + CSA 16384 rows x 4 source layers x 640 F16 ~84 MB
+ F32 state ~6 MB => ~100 MB class, split across both
cards. KV is NOT the fit constraint; weights + table are.

## Paper fit (weights only; KV/buffers excluded)

PTQ1_0, 152.8 GiB total:
  table    40.0 GiB  disk (E:) via mmap, -ot ...=CPU
  backbone 112.8 GiB  40 blocks @ ~2.82 GiB avg
    GPU -ngl 22   62.0 GiB   ~31 GiB/card, ~2 GiB headroom
    CPU RAM        50.8 GiB  + ~5 GB OS + KV = ~57 GB
                   < 63.9 GB physical, ~6 GB page-cache
                   headroom for the table's hot rows
  => fully resident in VRAM+RAM+disk; no backbone pageout.

PQ2_0, 185.3 GiB total:
  table    48.3 GiB  disk via mmap
  backbone 136.6 GiB  40 blocks @ ~3.42 GiB avg
    GPU -ngl 18   61.5 GiB
    CPU RAM        75.1 GiB  vs 63.9 GB physical - ~5 GB OS
                   => ~16 GiB of backbone paged out
  => NOT fully resident; the paged expert rows cost SSD
     page-ins per token (~35 MB/token at ~2 GB/s => +17 ms
     class). PQ2_0 is expected to land below the plain
     traffic ratio (4.90/4.12 = 1.19x) vs PTQ1_0.

The engram table per-token access is KB-scale (a few
gathered rows per token, ple layers 1 + 14, 32 rows x 256
each), so disk-backed table is a capacity problem, not a
bandwidth problem (DESIGN section 3; Unsloth ships the
same offload).

Fit arbiter: --fit is on by default; a pinned -ngl that
cannot fit raises a clean fit_exception at startup (B19).
Ladder on startup failure: drop -ngl by 2 (22 -> 20 -> 18 /
18 -> 16), read the printed estimate first, never raise it
blindly.

## Load test (this box, done 11-04, F43)

The 9-shard A/B runs on the other machine; this box
validates the port itself on the mini GGUF:
llama-cli with the A/B flags (--jinja --numa distribute
-cmoe -fa on -fit on -sm layer --adaptive-ctx none -c 8192
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --flash-attn on --reasoning-preserve
--reasoning-budget 4000 --image-max-tokens 4000 --image-
min-tokens 1024) + -n 32 + one prompt.
Runner: %TEMP%\opencode\run-load-test.ps1 (detached
Start-Process, empty-file stdin, recorded pid; F43 has the
two CLI gotchas it works around).
Result: PASS - prompt 44.36 t/s (40 tokens), generation
148.43 t/s (32 tokens), finish_reason "length", no jinja
error, no asserts. Output incoherent by construction
(truncated weights); pipeline verification, not quality.
Three source fixes came out of it: exp_probs_b_vl
registration, the missing final ggml_build_forward_expand
(the deterministic NULL-buffer crash that had looked like
an OOM), and the jinja dict() builtin.

## A/B protocol

Fixed prompt set (closes DESIGN section 7, item 2):
  p1 knowledge, engram-sensitive (facts + table recall)
  p2 knowledge, Chinese (required by the user)
  p3 arithmetic, multi-step with a final remainder
  p4 code, function + assert tests
  p5 logic trap (all but 9)
  p6 rate reasoning (100 machines / 100 widgets)
  p7 structured (3 bullets + 1 line)
Params for every request: temperature 0.2, top_p 1.0,
seed 42, max_tokens 512, sequential, one stream, model id
read from the server's own /v1/models.
Scoring: side-by-side read of the three captured response
sets (quality) + measured t/s per prompt (perf). No
auto-metric; the f12/f13 classes are the prior (PTQ1_0
~93-95% of the Q8_K_XL reference class, PQ2_0 ~97-99%).

t/s priors (f6 traffic, f7/f8 anchors):
  baseline UD-Q4_K_XL: 13.7 GB/token, DDR4 ~50-70 GB/s
    effective on a 16C 5950X => ~4-6 t/s class
  PTQ1_0: 4.12 GB/token, f8 x0.5-0.7 MoE discount
    => ~18-25 t/s standalone
  PQ2_0: 4.90 GB/token, same discount, plus the ~16 GiB
    backbone pageout => ~12-20 t/s minus the page penalty

## Steps

0. Baseline (service UP, safe, no stop needed):
   - re-derive the 8080 listener pid; it must be 511116 or
     a documented successor - if it is anything else, stop
     and re-derive the service state before proceeding.
   - copy the script to E:\tmp\ and run:
     pwsh E:\tmp\ab-dsv41.ps1 -Url http://127.0.0.1:8080 `
       -Tag baseline `
       -ApiKeyFile C:\Users\admin\source\macube\config\local\llama-api-keys.txt
   - the key is read from that file at runtime, never
     printed, never written to the outputs, never committed.
   - verify 7/7 responses captured, non-empty, coherent.
1. STOP (user action, the only stop window): the user
   stops the macube service by their own procedure; this
   side never kills user PIDs. Confirm the 8080 listener
   is gone and both GPUs are idle before launching.
2. PTQ1_0: launch detached (line above), wait for
   "server is listening", read the printed VRAM/RAM
   budget, then:
     pwsh E:\tmp\ab-dsv41.ps1 -Url http://127.0.0.1:58900 -Tag ptq1
   Teardown: Stop-Process on the server pid (ours, safe),
   confirm VRAM freed before step 3.
3. PQ2_0 (optional, same window): same, port 58901,
   -ngl 18, -Tag pq2.
4. RESUME (user action): the user restarts the macube
   service; confirm 8080 answers and the child spawned.
5. Compare: per prompt, read baseline/ptq1/pq2 responses
   side by side; tabulate t/s; write the verdict to
   PLAN.md P2 + append it to the FINDINGS F43 entry
   (load test); close DESIGN section 7 item 2; commit
   private/ + push.

## Risks and abort criteria

- fit_exception at startup: run the -ngl ladder; if
  -ngl 12 still fails, abort the run, go to step 4.
- OOM or crash mid-generation: capture the err log, stop
  the server, go to step 4.
- Quality collapse on p2 (Chinese) or p1 (facts): expected
  per f12 for PTQ1_0 - record it, do not abort.
- Never run the test server while 8080 is up: the two
  would fight for the same cards and the A/B would not be
  clean. The test server exists only inside the window.
- E:\DataVolume can transiently fail open() - a rerun of
  the same step is the fix, not a different path.
