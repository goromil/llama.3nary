# DSV4.1 ternary: size, fit, perf + quality expectations (11-02)

Status: written 11-02 on the user's interim docs request. All size
figures are computed from byte-exact shard measurements (f1-f5);
perf + quality figures are ESTIMATES to be verified in P2, with the
public references in section 6. Companion: the DSV4.1 TERNARY
section of PLAN.md (locked decisions A-D, P0/P1/P2). GB = 10^9,
GiB = 2^30.

## 1. Source and targets (measured 11-02)

f1. Source: E:\DataVolume\Models\DeepSeek-V4.1-Flash-1017-GGUF-
    UD-Q4_K_XL\ - 9 shards (fork GGUF variant) + mmproj-BF16.gguf
    (972,208,288 B, copied through unchanged) + engram\ (Unsloth
    Q5_K swap table + set_split.py). Shard total
    501,679,784,861 B = 501.68 GB = 467.1 GiB.

f2. Shard inventory (headers parsed byte-exact):
    | shard | content | bytes |
    |---|---|---|
    | 1 | 92 KVs, 0 tensors | 5,780,589 |
    | 2-7 | 20 MXFP4 tensors (type 39) each, [884736, 5120] = 4.5298B params, 4.25 bpw | 48,129,640,000 each, 288.78 GB total |
    | 8 | per_layer_token_embd.weight [256, 768022850], file type field 43 (MXFP8, 33 B/32) | 202,758,032,592 (202.76 GB data) |
    | 9 | 922 tensors: 351 Q8_0 (8,691,744,768 B) + 571 F32 (474,120,640 B) | 9,165,923,392 |

    The 120 expert tensors = 40 blocks x 3 (gate/up/down), each
    tensor = 384 experts x 2304 ff. The engram tensors exist only
    in blk.1/blk.14: engram_gate.weight [5120, 4] (F32) +
    engram_kv.weight [6144, 25600] (Q8_0). output.weight is
    [5120, 129280] Q8_0, tied shape with token_embd.weight.

f3. Params: 543.58B experts + 8.19B Q8_0 set + 196.61B table +
    0.12B F32 = 748.5B total. Cross-check vs Unsloth's claims for
    this release: their "Rest 297.9" + "MXFP8 202.8" GB vs our
    measured 297.94 (experts + Q8_0 + F32) + 202.76 GB: EXACT.
    So their family-size numbers can be used directly for
    comparison.

f4. Targets: PTQ1_0 (ggml type 143, 28 B per 128, 1.75 bpw) and
    PQ2_0 (ggml type 142, 34 B per 128, 2.125 bpw), both fork
    ternary codecs, plain PTQ, NO imatrix. The fork's MXFP8
    reader (new ggml type 144, P0 items 1-2) is the dequant path
    for the type-43 table input.

f5. Requant tensor map: 120 MXFP4 expert tensors + all 351 Q8_0
    weight tensors (incl engram_kv) -> PTQ1_0/PQ2_0. All 571 F32
    tensors (0.47 GB) stay F32. engram_gate.weight [5120, 4] is
    EXCLUDED from quantization (tiny gate, kept F32 by the
    llama-quant.cpp exclusion). All tensor dims are 128-multiples,
    so every tensor quantizes with no tail handling.

## 2. Expected output sizes (computed)

Per set: bytes = params x bpw/8, per 128-block packing.

| set | experts 543.58B | Q8_0 set 8.19B | F32 | table 196.61B | total | GiB | eff bpw |
|---|---|---|---|---|---|---|---|
| PTQ1_0 (1.75) | 118.9 | 1.79 | 0.47 | 43.0 | 164.2 GB | 152.8 | 1.76 |
| PQ2_0 (2.125) | 144.4 | 2.17 | 0.47 | 52.2 | 199.3 GB | 185.3 | 2.13 |

Derivation (PTQ1_0): experts 4,246,732,800 blocks x 28 B =
118.91 GB; Q8_0 set 8.186B params x 1.75/8 = 1.79 GB; table
1,536,045,700 blocks x 28 B = 43.01 GB; F32 0.47 GB stays.
PQ2_0: same block counts x 34 B. Output keeps the 9-shard split
(decision C): shard 1 KV, shards 2-7 = 120 expert tensors
(~19.8 GB/shard PTQ1_0, ~24.1 GB/shard PQ2_0), shard 8 table,
shard 9 misc (Q8_0-set + F32 tensors, ~2.3 GB PTQ1_0 / ~2.6 GB
PQ2_0).

Cross-check vs the Unsloth family (their claimed sizes, same
model; their "Rest" = non-engram weights, "table" = the
per_layer_token_embd pack at the per-quant precision):

| Unsloth variant | Rest | table | total | suite avg | retention |
|---|---|---|---|---|---|
| IQ1_S | 122.4 | IQ4_NL 110.6 | 233.0 | 83.53 | 95.3% |
| IQ1_M | (1-bit-M class) | IQ4_NL 110.6 | (233-247 class) | 85.55 | 97.6% |
| IQ2_XXS | 165.3 | Q5_K 135.2 | 300.5 | 86.63 | 98.8% |
| Q2_K_XL | 199.6 | Q5_K 135.2 | 334.8 | 87.49 | 99.8% |
| UD-Q8_K_XL (source) | 297.9 | MXFP8 202.8 | 500.7 | 87.68 | ref |

Positioning: our PTQ1_0 backbone (121.2 GB) sits AT their IQ1_S
Rest (122.4 GB), i.e. a 1-bit backbone; our PQ2_0 backbone
(147.0 GB) sits between their IQ2_XXS (165.3) and Q2_K_XL
(199.6) Rests. The size gap vs their same-score-class sets is
almost entirely the TABLE: ours 43.0/52.2 GB at 1.75/2.125 bpw
vs their IQ4_NL 110.6 / Q5_K 135.2 GB. Net: PTQ1_0 total 164.2 GB
vs their 1-bit-class total 233.0 GB (29% smaller), PQ2_0 total
199.3 GB vs their 2-bit-class total 300.5 GB (34% smaller) - at
the quality-cost discussed in section 5 (our table is 2.6x
coarser in bits than theirs).

## 3. Fit on this machine (2 x 32 GB B70, 63.9 GB RAM)

Machine totals: 64 GB VRAM + 63.9 GB RAM = 128 GB. The model
weights alone are 152.8 GiB (PTQ1_0) / 185.3 GiB (PQ2_0) -
before KV, draft, context buffers, OS.

A) All-on-GPU (152.8 GiB): NO, ~2.4x over the 64 GB VRAM.
B) Model on GPU + table on CPU: 112.8 GiB on GPU (backbone
   121.2 GB + mmproj + KV): still NO, ~1.8x over.
C) Any fully-resident split (weights anywhere in VRAM+RAM):
   NO - 152.8 GiB > 128 GB total. The set cannot be fully
   resident on this box in either variant.

What WILL run: a mixed GPU/CPU/disk split - GPU holds ~50-54 GB
of weights (both cards, layer or tensor split, headroom left
for KV + buffers + mmproj), the rest of the weights on CPU RAM
(~40-50 GB), and the 43-52 GB table mmap'd directly from disk
with only the gathered rows paged in - the weight set alone
needs 164.2 GB (PTQ1_0) > 128 GB VRAM+RAM, so 30+ GB of it
must be disk-backed in ANY working configuration. Unsloth's
own offload recipe is exactly this: --override-tensor
per_layer_token_embd.weight=CPU + mmap/lazy. Small ctx (a few
K, not the 262K-class service ctx), the live macube service
stopped (re-derive its PIDs via Get-NetTCPConnection -LocalPort
8080,58452 first; never kill user PIDs), and decode is
CPU-bandwidth-bound (DDR4-class ~50-70 GB/s effective on a 16C
5950X vs 608 GB/s per card) - expect single-digit t/s on the
CPU-resident expert rows, GPU-resident rows at the section-4
rates.

Why the table off-GPU is acceptable by design: per token, the
engram layer gathers only a few table rows (n-grams of up to 4
tokens x 256-dim projections, KB-scale), so the table is a
CAPACITY problem, not a bandwidth problem - the same reason
Unsloth ships it SSD/CPU-offloaded.

Unsloth's own hardware-requirement table for this model
(claimed, RAM+VRAM): 1-bit 128-145 GB, 2-bit 176-205 GB,
3-bit 256-290, 4-bit 305, 8-bit 315. This box (128 GB) sits at
the very bottom of their 1-bit floor - consistent with the C
verdict above.

## 4. Expected performance (estimates, P2 verifies)

f6. Per-token weight traffic (decode, 1 stream): 40 blocks x
    6/384 routed experts x 35.39M params/expert = 8.49B routed +
    8.19B non-expert (attention, shared expert, norms, embeds,
    LM head) = 16.68B params/token + 0.47 GB F32.
    | set | GB/token | vs source |
    |---|---|---|
    | PTQ1_0 | 4.12 | 3.3x less |
    | PQ2_0 | 4.90 | 2.8x less |
    | UD-Q4_K_XL (source) | 13.7 | - |

f7. B70 anchors (this box): spec 608 GB/s per card, 1216 GB/s
    combined (PugetSystems, 2 cards). Measured B11 reference:
    27B PTQ1_0 (5.54 GB) tg128 = 26.72 t/s = 148 GB/s effective
    (N10). Bandwidth roof at that effective rate: PTQ1_0
    148/4.12 = 36 t/s, PQ2_0 148/4.90 = 30 t/s.

f8. MoE overhead discount: the per-token expert traffic is 6 of
    384 experts per block (non-contiguous rows of the packed
    expert tensors, 35.39M-param GEMMs x 240 per token), so
    effective bandwidth drops well below the dense anchor.
    Discount x0.5-0.7: PTQ1_0 ~18-25 t/s, PQ2_0 ~12-20 t/s
    STANDALONE (service stopped, all weights resident in
    VRAM+RAM as far as the split allows). This is vs the live
    Q6_K 27B 12.5-18 t/s class - the MoE model at 1.75 bpw
    should beat the 27B dense 6-bit service on raw decode
    t/s, if the split keeps most expert rows on GPU.

f9. DSpark drafter (separate files, NOT local: Q8_0 7.98 GB /
    BF16 8.51 GB, matched checkpoint versions required):
    Unsloth claims 1.3-1.5x decode for V4.1; PrismML measured
    1.34x (Bonsai 27B, H100, 98 -> 131.8 t/s, accepted length
    tau ~3.7 at k=4). Applied to f8: PTQ1_0 ~24-38 t/s,
    PQ2_0 ~16-30 t/s. Lossless (verification preserves the
    target distribution).

f10. Scaling references (PrismML, fetched 11-02):
    - Bonsai 27B Q2_0 (1.71 bpw, same codec family): M5 Pro
      tg128 26.2 t/s streaming ~186 GB/s of weights - decode is
      memory-bandwidth-dominated, which is the regime where the
      ternary byte advantage pays.
    - The H100 caveat: at batch 1 on a high-bandwidth GPU,
      decode becomes kernel-launch/sync-limited and ternary
      converges with 4-bit (98 vs 104.8 t/s) despite ~1.9x fewer
      bytes per step. The B70 (608 GB/s, Xe2) is NOT in that
      regime at these sizes - it stays bandwidth-bound, so the
      ternary advantage holds here; it would NOT hold on an H100
      at batch 1.
    - Bonsai 2 27B (1.76 bpw = our PTQ1_0 class): up to 143 t/s
      on RTX 5090, 46.8 t/s on M5 Max, 0.714 mWh/token on RTX
      4090 (40% more efficient than an 8B model at FP).
    - Prompt processing (pp512) is compute-bound, not
      bandwidth-bound: expect pp comparable to the source
      Q4_K_XL service, not a 3x gain. (27B PTQ1_0 measured
      pp512 293.75 t/s vs tg 26.72 - the pp/tg ratio is the
      shape to expect, with the MoE routing overhead.)

## 5. Expected quality (estimates, P2 verifies)

f11. Unsloth v3.1 suite for this model (their claimed averages,
     ref = UD-Q8_K_XL 87.68):
     | variant | avg | retention | table precision |
     |---|---|---|---|
     | IQ1_S | 83.53 | 95.3% | IQ4_NL (4.5 bpw) |
     | IQ1_M | 85.55 | 97.6% | IQ4_NL |
     | IQ2_XXS | 86.63 | 98.8% | Q5_K (5.5 bpw) |
     | Q2_K_XL | 87.49 | 99.8% | Q5_K |
     Their 1-2 bit sets pair a low-bit backbone with a 4.5-5.5
     bpw table. Ours pairs a 1.75/2.125 bpw backbone with a
     1.75/2.125 bpw table (decision A).

f12. Expectation vs that table: PTQ1_0 (plain PTQ, no imatrix,
     table 2.6x coarser in bits than their IQ1_S table) -> at or
     BELOW the IQ1_S/IQ1_M class, call it ~93-95% of the
     Q8_K_XL reference. PQ2_0 -> near the IQ2_XXS class,
     ~97-99%. The table is the main unknown: it stores the
     per-n-gram memory embeddings (196.6B params), and no
     published result exists for ternarizing an engram table.
     Mitigating: per-token access is only a few rows, and the
     P2 comparison vs the live Q8_K_XL service on fixed prompts
     measures exactly this.

f13. PrismML ternary evidence (same codec family, 27B dense
     refs, fetched 11-02):
     - Bonsai 27B Q2_0, 1.71 bpw, plain PTQ: 94.6% of FP16
       (80.49 vs 85.07 thinking avg) - OUTSCORES the
       conventional IQ2_XXS 2.8 bpw build (85.5%, 72.73) by 7+
       points at two-thirds the size (5.9 vs 9.4 GB).
     - The conventional sub-4-bit collapse is SELECTIVE: IQ2_XXS
       falls to 57.5 on AIME26 and 56.4 on LiveCodeBench while
       keeping 88.93 on MMLU-Redux (casual testing misses it).
       Ternary holds the hard ones: AIME 87.5-90.8, LCB 82.8 vs
       FP16 93.3/87.8.
     - Bonsai 2 27B, 1.76 bpw (our PTQ1_0 bpw): 98.2% retention
       (83.9 vs 85.4) - the 1-bit-class operating point is now
       "practically lossless" per PrismML (was 95% gen-1).
     - 1-bit Bonsai 27B, 1.125 bpw: 89.5% (76.11).
     - Caveat: these are 27B DENSE models. DeepSeek V4.1 is a
       748B MoE (6/384 experts active) + a 196.6B memory table;
       there is no published ternary reference at this scale, so
       the f12 numbers are class expectations, not predictions.
       The P2 live A/B is the actual quality measurement.

## 6. References (accessed 2026-11-02)

- https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF -
  claimed family sizes (Rest + table per variant, exact match to
  our shard measurements), per-quant table precision, v3.1
  benchmark table, hardware-requirement table, engram CPU/SSD
  offload recipe (--override-tensor ...=CPU + mmap), DSpark
  1.3-1.5x claim.
- https://unsloth.ai/docs/models/deepseek-v4.1-flash - same
  family, docs view.
- https://prismml.com/news/bonsai-2-27b - Bonsai 2 27B: 1.76
  bpw, 5.9 GB, 98.2% retention (83.9 vs 85.4), 143 t/s RTX 5090,
  46.8 t/s M5 Max, 0.714 mWh/token RTX 4090, per-category table.
- https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf -
  Bonsai 27B Q2_0_g128 (1.71 bpw): 94.6% retention, per-benchmark
  table vs FP16, IQ2_XXS collapse details, tg128/pp512
  cross-platform table (M5 Max 44.0, M5 Pro 26.2 @ ~186 GB/s,
  M4 Pro 18.0, H100 98.0 vs 104.8 binary), peak-memory table
  (~1.3 GB overhead), DSpark 1.34x (98 -> 131.8 t/s, tau 3.7).
- B70 specs: 608 GB/s per card, 256-bit GDDR6, 32 GB (Intel
  datasheet; PugetSystems 2-card 1216 GB/s combined).
- Local: N10 perf anchors, NOTES.md N15 clone state, engram\
  README.md in the model dir.

## 7. Open items

- DSpark drafter files not local (need the matched checkpoint
  version before the f9 multiplier can be applied here).
- P2 fixed-prompt set for the live A/B (vs the UD-Q4_K_XL
  service on 8080) not yet chosen.
- The f8 discount factor (x0.5-0.7) is an assumption - P2
  measurement replaces it; if the SYCL MoE gather is worse than
  assumed, the t/s range drops proportionally.
