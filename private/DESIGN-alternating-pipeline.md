# DRAFT PROJECT - alternating 2-chat decode pipeline (10-28, evening)

Status: DRAFT, design only, NO code written. The user requested this
document ("2 chats, 1 token per half round"; the "phasing" of F39
[proposed]). F39's build decision stands (NOT building now - the
pre-registered gate is not met); this is the design record so the
idea is findable and reviewable, and so it can be picked up without
re-deriving anything. Backlog entry: B18.

Related: F39 (assessment + the 10-28 measurements + the user's
dispute of the 76% number), F38 (layer-mode per-layer P2P wait
structure), BACKLOG B14 (DP twin-server + router - REJECTED by the
user 10-28, "dead end"; the pipeline is the remaining 2x-aggregate
design).

## 1. What it is

Two concurrent chats decode in a staggered stream across the two
cards, on the EXISTING layer-mode split (meta backend, no new split
mode):

- Stage 1 = the low layers (the SYCL0 half), Stage 2 = the high
  layers (the SYCL1 half). The split is already at device/tensor
  boundaries in layer mode; the graph crossing the boundary is the
  residual stream (embd 5120 f32 = 20 KB per token) and the meta
  backend moves it asynchronously (ggml-backend-meta.cpp:2176
  subgraph split at PARTIAL tensor boundaries; :2418-2424 async
  enqueue; :2343 boundary copy - no host-side sync bubble).
- Steady state (chat X and chat Y):
  - t0:       X.s1 on SYCL0
  - t0+H/2:   Y.s1 on SYCL0  ||  X.s2 on SYCL1
  - t0+H:     X2.s1 on SYCL0 ||  Y.s2 on SYCL1
  - ... every token's stage 2 runs on one card while the next
    token's stage 1 runs on the other.
- One full token lands every H/2 (H = the current solo per-token
  step, ~71 ms at ~35K on this machine) = aggregate 2/H, each chat
  served at 1/H. "1 token per half round."

## 2. Why (the measurement that motivates it)

The 10-28 F39 measurement on the DEFAULT config (auto 4 slots +
unified KV, layer mode, two cards):

- Two concurrent chats DID batch into one 2-seq decode step
  (llama.log :178425-178437, child 51396).
- Per-stream tg_3s 10.7-10.8, SUM ~21.4 t/s vs the solo ~14 t/s at
  comparable ~35K ctx.
- The user disputed the resulting "76% of the 2/H roofline" verdict
  (tested repeatedly, expects ~2x from the weight-load-bound
  argument). The per-step numbers pin down WHY it is not 2x:

  - Solo step (34.8K ctx): 71 ms. 2-seq step (27.6K + 7K): 93 ms.
  - The 2-seq step is NOT 2x the solo step (else >= 142 ms): the
    weight set IS read once for both seqs. The weight sharing the
    user expects is real and already happening.
  - Decomposition: solo ~= W + A, 2-seq ~= W + 2A, with A ~= 22 ms
    (one seq's attention/KV/GDN-state share, A/H ~= 0.31). The
    measured 1.53x is AT the batching ceiling 2/(1 + A/H) ~= 1.52x:
    default batching is not misconfigured, it is already doing the
    right thing. The gap to 2/H is structural.

- The structural gap is STAGE IDLE. In layer mode a 2-seq batch is
  one graph: [both seqs' stage 1 on SYCL0] -> [both seqs' stage 2
  on SYCL1], the stage 2 subgraph depending on the residual stream
  crossing the boundary. SYCL1 sits idle for the whole of SYCL0's
  stage-1 work (W/2 + A per pair), then SYCL0 sits idle for
  SYCL1's stage 2. The pair takes (W/2 + A) x 2 = W + 2A = H + A,
  i.e. the batch runs the two cards in anti-phase as a PAIR.
- The stagger fixes exactly that: stage 2(token N) on SYCL1 overlaps
  stage 1(token N+1) on SYCL0. Both cards are continuously busy;
  the pair takes H (not H + A). Aggregate 2/H = 28.2 t/s on this
  baseline, +31% over the measured 21.4. The per-seq A is unchanged
  (each seq still reads its own KV - nothing removes that); what
  disappears is the idle bubble between the two stages.

- KV headroom (the other F39 win, unchanged): the pipeline shares
  both cards' KV across the SAME chat pair (each card owns the
  KV/GDN-state of its own layers for BOTH streams - the KV is
  already per-(layer, buft) and per-card local after B15,
  llama-kv-cache.cpp :231-232). DP is capped by one card's free
  VRAM per stream (~10 GB/card => ~20 GB total for two chats); the
  pipeline gets ~20 GB/card => ~40 GB total - ~2x the usable
  context for the two-chat case. The unified 262144-row ring holds
  two ~131K chats (the largest live conversation so far: 106815);
  two chats beyond ~131K each need this (or the 2x-cost per-slot
  KV that -np 2 gives).

## 3. Roofline (certain, carried from F39)

- A system token reads the full weight set W either way (serial
  layer split, two DP streams, or the pipeline): nothing beats 2/H
  without reducing W reads.
- Three-way comparison at this machine's H ~= 71 ms (~35K ctx):

  | design                | aggregate | per-stream | KV per pair        |
  |-----------------------|-----------|------------|--------------------|
  | solo, 1 chat          | 1/H (14)  | 14         | 1 stream           |
  | default 2-seq batch   | 2/(H+A)   | ~10.7      | shared ring        |
  |   (measured)          | (= 21.4)  | (measured) |                    |
  | DP (B14, REJECTED)    | 2/H (28.2)| 14         | ~20 GB (10/card)   |
  | pipeline (this draft) | 2/H (28.2)| 14         | ~40 GB (20/card)   |

- Pipeline aggregate 2/H TIES the DP aggregate 2/H (two independent
  cards, no cross-card dependency at all; both hold when the cards
  stay continuously busy) and the DP per-stream TG (1/H). Against
  DP the pipeline wins only on KV per pair (~40 vs ~20 GB) and on
  one model copy / one process / no router. Its edge vs the
  MEASURED default batching is +31%: the batch pays the stage
  bubble (W + 2A per pair) that the stagger removes.
- The 24% batch shortfall is the stage bubble A/(H+A) ~= 0.24 of
  the roofline, NOT per-seq waste: it vanishes under the stagger.
- The unified 262144-row ring holds two ~131K chats (the largest
  live conversation so far: 106815); two chats beyond ~131K each
  need the pipeline's shared KV (or the 2x-cost per-slot KV that
  -np 2 gives, which the user has now seen to be host-placed).

## 4. Design sketch (what would have to exist)

Nothing in the current server can express the stagger: the slot
loop is one blocking llama_decode() per token batch (the whole
graph, both cards, host waits). The pieces:

1. Stable stage boundary. The layer split's boundary must not move
   between tokens (rebalance = pipeline hazard). The boundary is
   already fixed per config; what is owed is a measurement of the
   two halves' per-token cost at the candidate boundary (the 65
   blocks are 48 GDN + 16 full-attn at 3,7,...,63 - the halves do
   NOT split evenly by compute; the boundary choice trades
   stage-imbalance against P2P frequency. One boundary per model,
   picked once, logged).
2. Two-stage submission without a host round-trip. Submit one
   token's full graph as today (stage 1 + stage 2, device-side
   dependency, async - this part already works), but from a slot
   loop that does NOT wait for the token to finish before
   submitting the next chat's token. The staggering emerges from
   the device queues: submit X, then Y, then X2, then Y2 - each
   graph is [s1 on SYCL0 -> s2 on SYCL1] with the device dependency
   inside the graph; SYCL0 queues X.s1, Y.s1, X2.s1... and SYCL1
   queues X.s2 after X.s1, Y.s2 after Y.s1... - which IS the
   stagger, with zero new device-side machinery. The host just
   needs to stop serializing the submissions.
   Open question: does the current graph build + enqueue already
   tolerate N unsubmitted tokens in flight (KV cell reservation,
   sampler state, slot bookkeeping)? That is the crux of the
   design; if the slot/ctx state assumes at most one in-flight
   decode per slot, a bounded in-flight window (2 tokens) with
   explicit KV reservation at submit time is the minimum.
3. The slot loop. Today: per token, build batch, llama_decode(),
   sample, emit, next. Pipeline mode: per token, build batch,
   submit (async), reserve KV for the in-flight window, and only
   sample/emit when the token's own stage-2 completion event
   fires. Two chats = two slots = two submission streams; the
   stagger is automatic (they share the two device queues).
4. Sampling + MTP. The sampler runs on the stage-2 output (host or
   device - to be checked; if host, it adds a host round-trip per
   token that the stagger must absorb: sample(token N) while
   submit(token N+1) proceeds - sampling is cheap vs H/2, so a
   one-token-lag is acceptable). MTP/speculative (the draft model +
   reasoning-budget path) assumes ONE blocking decode per token -
   pipeline mode starts with MTP OFF (spawn-flag gated); the
   interaction is an open question, not a blocker.
5. Errors + failure isolation. One chat errors (KV wall,
   EXCEED_CONTEXT_SIZE, bad_alloc): its in-flight token is
   discarded, the other chat's stagger continues unharmed (the
   device queues are independent per token). The F31 wall path
   (decode() returns the wall, try_clear_idle_slots, DECODE_RETRY)
   must be re-expressed for an async submit (the wall is discovered
   at submit/reservation time, not mid-decode).
6. Fit + ctx sizing. The no_alloc fit (common/fit.cpp:456-458,
   :528) sizes per-stream KV for n_streams; pipeline mode uses the
   UNIFIED ring (kv_unified = true, one ring, both streams), so
   the fit math is the existing one - no new fit surface. The
   adaptive-ctx grow/shrink (B15) works on the same ring
   unchanged.
7. Server surface. A spawn flag (draft name: --decode-pipeline,
   env LLAMA_ARG_DECODE_PIPELINE) gated to: layer mode, two SYCL
   devices, unified KV, >= 2 concurrent slots active, MTP off.
   Off = today's behavior, byte for byte.

## 5. What it buys / costs (summary table)

- Buys: aggregate 2/H (tied with DP; +31% over the measured
  single-server 2-seq batch of 21.4 t/s); per-stream 1/H; ~2x KV
  headroom for the two-chat pair (~40 vs ~20 GB DP); no second
  process, no router, no model duplication (one 17.5 GB model,
  not two).
- Costs: a second in-flight-decode execution model in the slot
  loop (the part upstream would not merge - no mainline dual-card
  SYCL pipeline exists; fork-specific by nature); stage-imbalance
  handling for the 48 GDN + 16 full-attn hybrid; MTP OFF while
  active; the F31 wall path re-expressed; error isolation for
  in-flight tokens.
- vs default batching (the fallback if this is never built): keep
  the auto 4-slot + unified KV layout; accept 2/(1+A/H) ~= 1.52x
  as the measured ceiling for 2-chat throughput on one model copy;
  the capacity path for > ~131K-per-chat pairs is the in-place
  ring growth (B15, shipped) + the ring itself (the 262144 unified
  rows hold two ~131K chats).

## 6. Open questions (to resolve before any implementation)

1. In-flight window: does the ctx/slot state tolerate 2 submitted
   not-yet-sampled tokens (KV reservation, pos, sampler queue)?
   Read the server-context decode path end to end and list every
   per-token state assumption. (This is the make-or-break.)
2. Where sampling runs (device vs host) and whether a one-token
   host round-trip per token is acceptable against H/2 ~= 35 ms.
3. The actual stage-1/stage-2 per-token cost split at the 65-block
   boundary (measure once: two 1-seq decodes with the boundary
   logged, or a single-token trace).
4. GDN recurrent-state ownership: layers 0-31 states on SYCL0,
   32-63 on SYCL1 - confirm no cross-card state sync exists today
   (it should not; GDN state is per-layer local).
5. MTP interaction (or a clean flag-exclusion).
6. Whether the stagger holds under the adaptive-ctx grow/shrink
   (a regrow mid-stream swaps KV tensors - in-flight tokens must
   drain or the resize must wait; B15 already pauses slots
   briefly, extend the same gate to the in-flight window).

## 7. Exit criteria (for when/if it is picked up)

- Design question 1 answered with a state list; a staged plan
  written (F33-style: stage 1 core, stage 2 server, stage 3
  verify).
- Measured: 2-chat staggered TG at comparable ctx vs the 21.4 t/s
  batching baseline - target: >= 26 t/s sum (90% of the 2/H
  roofline, i.e. the bubble is gone), per-stream >= 12 t/s.
- No regression: single-chat TG on the flag-off path byte-identical
  to today; flag-on single chat == today's single chat (the
  stagger only exists with >= 2 active slots).
- Upstream posture: stays fork-only; no PR.
