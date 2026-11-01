# B20 design: constrained joint solve over (ctx, weight placement)

Status: PROPOSED 10-29 - for review before implementation.
Scope: the infeasible + unpinned branch of the B19 adaptive settle
(common/fit.cpp, the branch that currently demotes to the floor and lets
step 3 redistribute). Dense models only (hp_nex == 0) in v1.

## 1. Problem

The B19 settle is sequential. When the requested ctx (anchor) does not
fit with the current weight placement and even the floor does not fit
there (infeasible), the unpinned path demotes the ctx to the floor and
lets step 3 redistribute the weights at the floor. The settled ctx is
then the floor, even when a mid ctx would fit with a different
placement.

The measured 56146 class shows the gap: co-resident with the live
service, the all-on-GPU free-memory split is over budget even at the
floor on at least one card, so B19 settles to 4096 and step 3 keeps
whatever number of layers fit at 4096. A joint pass could settle a
bigger ctx with a placement balanced for that ctx.

## 2. Source-verified facts (this session, fit.cpp + src)

f1. ngl = -1 (the unpinned default) means ALL layers on GPU
    (llama-model.cpp:2301). With an all-zero tensor split the layers
    are split across the devices proportional to free memory
    (llama-model.cpp:1704-1721). So dmds_full in the unpinned case is
    the all-on-GPU free-memory split.

f2. The KV cache of layer il sits on the device of layer il
    (llama-kv-cache.cpp:205, offload_kqv). KV tracks the layers. With
    ngl = 0 every layer (and the KV of every layer, and the output
    layer, llama-model.cpp:1761) is on the host: the per-GPU "low
    bound" of the fill is only the sched workspace, essentially 0
    (llama-context.cpp:3623-3628). Consequence: the low-bound
    feasibility test is nearly vacuous, and any C* derived from
    low-bound rates is meaningless. The mechanism sketched in the
    B19 close-out (measure the low bound at ngl=0, solve C* from
    low-bound rates) is rejected by these facts.

f3. Per-device footprint at ctx C with a contiguous slice P of layers:

        f_id(C, P) = W_id(P) + C * kv_id(P) + comp_id(C)

    W_id: weight bytes of the slice, independent of C.
    kv_id: per-token KV bytes of the slice, exactly linear in C
    (unified KV: n_ctx_seq = C, llama-context.cpp:379).
    comp_id: the sched workspace of the device backend
    (llama-context.cpp:3627), ctx dependent through the attention
    workspaces, usually a small fraction of the KV term.

    f_id is monotone non-decreasing in C and in the number of layers
    on the device.

f4. Step 3 fill invariant (fit.cpp:1087-1141): for any ctx C the fill
    returns a placement where each device's own footprint at C is <=
    its target (the low bound is only replaced when it fits; memory is
    monotone in the layer count). If a device cannot take its
    contiguous share, the FRONT layers fall to the host, because
    n_gpu_layers = sum of the per-device layer counts
    (set_ngl_tensor_split_tbo, fit.cpp:960-971, i_gpu_start at
    llama-model.cpp:1736). All layers stay on the devices iff
    sum(ngl_per_device) == hp_ngl + 1. There is no other failure
    mode, and no final check after step 3 (fit.cpp:1288-1289): a
    device whose workspace alone exceeds its target silently violates
    (pre-existing, practically unreachable, out of scope).

f5. Consequences for the joint problem:
    - The fill at C always returns a per-device-fitting placement, so
      the only meaningful joint constraint is "no front overflow to
      the host", i.e. sum(ngl_per_device) == hp_ngl + 1.
    - sum(ngl_per_device) is monotone non-increasing in C (higher C
      lowers every device's per-layer cost budget). The feasible set
      over C is a prefix of [floor, anchor].
    - The max feasible C is found by: fill at the floor, and if that
      placement keeps every layer on the devices, grow the ctx into
      the per-device slack of that placement, and re-fill once at the
      grown ctx.

## 3. Objective

Maximize the settled ctx C in [floor, anchor] subject to:
  - the fill at C keeps every layer on the devices (f5), and
  - every device meets its budget + margin at C (fill invariant, f4).

If no such C exists (the weights alone do not fit on the cards at the
floor), keep the B19 behavior: settle to the floor, step 3
redistributes, the front walks to the host.

Rationale for "no front overflow" as the constraint: a placement with
host-offloaded layers is a decode-speed cliff (CPU layers), and the
B19 philosophy is that the ctx is the variable and the placement keeps
everything on the cards. The host-overflow placement is the fallback
for "the weights do not fit", not the target of a settle.

## 4. Mechanism

The change is confined to the unpinned infeasible branch and step 3 of
common_params_fit_impl. No new subsystems, no new measurement
primitives; only the existing common_get_device_memory_data_impl
(no-alloc projection) and the existing step 3 fill are reused.

4.1 Demote branch (fit.cpp:840-852): as today, set
    cparams->n_ctx = n_ctx_base = extra->n_ctx = floor, warn, fall
    through to step 3. New: remember b20_anchor = anchor.

4.2 Step 3: run the existing fill (fit.cpp:1072-1141) in a two-pass
    loop. Pass 0 runs at the current cparams->n_ctx. For the B20 flow
    that is the floor, exactly what B19 does today, so pass 0 adds no
    measurements over B19.

    After pass 0, if (b20_anchor > 0 && hp_nex == 0 &&
    sum(ngl_per_device) == hp_ngl + 1 && b20_anchor > cparams->n_ctx):
      the floor placement keeps every layer on the devices, so there
      is slack to grow into.
      a. Measure the anchor once with the floor placement
         (common_get_device_memory_data_impl on a copy of mparams with
         the floor placement applied, cparams->n_ctx = b20_anchor).
         One no-alloc measurement, the same kind step 3 already does
         O(nd * log layers) of.
      b. Per device, with used_floor = pass-0 mem[id],
         used_anchor = measured, target = targets[id]:
         - used_anchor <= target: device is not a bottleneck.
         - else rate = (used_anchor - used_floor) / (anchor - floor);
           C*_id = anchor - ceil((used_anchor - target) / rate), the
           largest C with the linear footprint at the floor placement
           <= target. rate <= 0 is treated as infeasible (over budget
           without growing in ctx).
      c. C* = min over devices, aligned down to lcm(16384, n_streams)
         like the B19 settle (fit.cpp:827), clamped to [floor, anchor].
      d. If C* > floor: set cparams->n_ctx = n_ctx_base =
         extra->n_ctx = C*, warn, re-run the fill (pass 1) at C*.
         The fill at C* is guaranteed to keep every layer on the
         devices: the floor placement fits at C* by construction, so
         the fill (which maximizes each device's slice) cannot
         overflow to the host (f5). It may produce a different,
         better-balanced placement for C*; that is the point of the
         joint solve.
      e. If C* == floor: keep the floor placement from pass 0, no
         second fill.
    On any other path (no demote, MoE, overflow at the floor) the loop
    runs a single pass and the code is byte-identical in behavior to
    today.

    For MoE (hp_nex > 0) v1 skips the growth pass entirely: the
    dense-only fill of step 3 plus the step 4 conversion would need
    the ctx-dependent global_surplus_cpu_moe recomputed per pass, and
    the user's model is dense. Documented v1 limitation; the B19
    floor behavior remains for MoE.

4.3 What stays unchanged:
    - the pinned path (per-device deficit throw, fit.cpp:853-863)
    - the B19 settle to limit < anchor at the current placement
      (fit.cpp:825-839)
    - the tensor split mode (separate impl)
    - the n_ctx_auto (ctx = 0) path
    - step 4 (MoE) and the tail of step 3
    - the caller (common/common.cpp:1318-1336), the server ceiling
      (server-context.cpp:1119-1131), the fit_ctx_base consumption

4.4 Cost: over B19, the B20 path adds at most one no-alloc
    measurement (the anchor at the floor placement) and at most one
    extra fill (pass 1 at C*, O(nd * log layers) measurements). Both
    only when the demote branch fired AND the floor placement kept
    every layer on the devices. Startup time stays in the same
    order as B19.

## 5. Correctness argument

- Soundness (no over-commit): the fill only accepts a slice for a
  device when a direct no-alloc measurement of that device at the
  exact settled ctx is <= target (f4). The settled ctx is verified by
  the pass-1 fill itself, not by the linear estimate. The linear
  estimate only decides HOW MUCH to grow; if it overestimates, the
  pass-1 fill simply gives the bottleneck device fewer layers, and
  because the floor placement fits at C* by construction the total
  cannot drop below hp_ngl + 1, so no host overflow is introduced by
  the growth itself.
- Monotonicity: feasible C form a prefix of [floor, anchor] (f5), so
  "grow to the per-device linear limit of one good placement, then
  re-fill" is a sound one-shot strategy, not a search.
- Conservatism: comp_id (workspace) is not modeled explicitly; if it
  grows convexly in C the linear chord overestimates the usable
  slack and C* is slightly optimistic, which the pass-1 fill bounds
  as above. If it grows concavely, C* is conservative. The KV term,
  which dominates, is exactly linear (f3).

## 6. Verification plan (co-resident, no-alloc probes, same as B19)

All probes with the current build/bin/llama-fit-params.exe alongside
the live service; every number is co-residency dependent.

v1. NONE control: --ctx-size 163840 --n-gpu-layers 999 -fit
    adaptive none must still print "n_gpu_layers already set by user
    to 999, abort" and exit as before.
v2. Pinned infeasible control: --ctx-size 163840 --n-gpu-layers 999
    adaptive full, cards in the co-resident band, must still throw
    the per-device deficit (B19 behavior, unchanged branch).
v3. In-budget control: --ctx-size 163840 --n-gpu-layers auto (or 999
    on free cards) adaptive full, cards free, must settle nothing and
    exit 0 with the same --fit-print numbers as the B19 build.
v4. B20 target (the 56146 class): --ctx-size 163840 --n-gpu-layers
    auto adaptive full, cards in the band where the all-on-GPU split
    is over budget at the floor but the weights still fit on the
    cards. Expected: WARN demote to the floor, then WARN grew to
    C* > floor, --fit-print shows a redistributed placement with
    every layer on the devices (sum ngl == hp_ngl + 1) and per-device
    used <= free - margin at C*. The B19 build on the same probe
    settles to 4096: the delta is the B20 gain.
v5. Exact-floor control: --ctx-size 4096 --n-gpu-layers 999 adaptive
    full in the band: unchanged deficit throw.
v6. MoE guard: not testable on this model (dense); the guard is
    hp_nex == 0 and the B19 path is byte-identical otherwise (code
    review + v1-v5).

A deterministic catch of the growth path outside the live band needs
the stop window (ctx above the train cap forces the infeasible branch
on free cards).

## 7. Open questions for the user

q1. Two WARNs on the growth path (demote to the floor, then grew
    back to C*) versus one combined WARN. This design keeps the B19
    demote WARN and adds a second one.
q2. C* alignment: lcm(16384, n_streams) as in the B19 settle. Coarser
    (e.g. always 16384) or finer (256 as in the auto-ctx path,
    fit.cpp:766) is also defensible.
q3. v1 dense-only: accept the MoE limitation for now, or make the
    growth pass work for MoE in v1 (recompute global_surplus_cpu_moe
    per pass, growth after the step 4 conversion).
