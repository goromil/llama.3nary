# PLAN - updated plan (start at IMMEDIATE NEXT STEPS)

Status markers: [x] done, [ ] open, [B] blocked on user.

## IMMEDIATE NEXT STEPS (fresh session starts here, 10-22)

Context: pivot to tensor-mode ternary compatibility (user direction
10-22: "make ternary code tensor compatible and then ... port it back
to the min llama.cpp fork"). Investigation is COMPLETE (F19-F25).
USER DECISIONS 10-22 (evening): (1) H-A APPROVED - implement in this
fork first, then port; (2) B7 startup-side (B7a) APPROVED; (3) B7
runtime preemption C0/C1 DROPPED/CLOSED. Clone git state verified
(no local commits; 3 uncommitted user mods to preserve - F25/B13).

1. [x] PRESENT + GET OK: root-cause report (F20) + H-B death (F22) +
   H-A design (F23) vs D1 mirror fallback. User OK 10-22. B7
   C0/C1 dropped; B7a approved; B13 clone confirmed as port target.
2. [ ] IN PROGRESS: implement H-A (B12) in this fork:
   a. llama-model-loader.h: pre-transform callback member on
      llama_model_loader; llama-model-loader.cpp load loop (:1551-1672):
      branch forces the read_buf path for target tensors, calls the
      callback, skips the normal set; no-op when unset.
   b. Model side (llama-model.cpp, qwen35 arch): transform function
      (dequant PQ2_0 -> f32, per-1024-row FWHT butterfly W8 x W128,
      gather 128-row blocks grouped -> tiled t_b = 3*(b%16) + b/16,
      per-row sign flip, requant quantize_row_pq2_0_ref); wire it as
      the callback before ml.load_all_data (:1940); gate = tensor mode
      + hadamard_gdn_v_grouped + `.ssm_out.` names (builder runs AFTER
      load, so use hadamard_weight_blocks + the flag + split mode).
   c. llama-graph.h:23-32: add `bool folded = false;` to
      llama_hadamard_transform; builder sets it in tensor mode;
      build_lora_mm (llama-graph.cpp:1546-1603) skips the whole
      transform block when folded.
3. [ ] Rebuild (N3). Verify with N14: no assert; SPLIT_STATE for the
   ssm_out chain = {axis 0, 2048x3, 1024/1024}; output sanity.
4. [ ] Test PQ2_0 in tensor mode (N14), then PTQ1_0
   (`-m E:\DataVolume\TernaryModels\Ternary-Bonsai-2-27B-PTQ1_0\Ternary-
   Bonsai-2-27B-PTQ1_0.gguf -mm ...-mmproj-BF16.gguf -ngl all -sm
   tensor -c 160000 -dev SYCL0,SYCL1`). Expect slight degradation from
   the extra requant pass.
5. [ ] Refresh private/ (F19-F25 + results) and commit+push (private/
   ONLY, ASCII message + `Assisted-by: Claude, Qwen3.8-27B`).
6. [ ] B13 recon (after B12 lands): diff upstream vs 3nary meta
   backend (ggml-backend-meta.cpp), upstream PQ2_0/PTQ1_0 type enum
   values, map missing 3nary work in the clone (F25 `how` 1-3). Write
   recon notes to private/. THEN port commit-by-commit, scope
   user-confirmed.

## PARKED (do not start without user re-scoping, 10-22)

- B7 preemption C0/C1/C2: CLOSED 10-22 (twin-server data-parallel
  design DROPPED; C0/C1 DROPPED per user). Plan intact in F11 for the
  record only.
- B7a (startup-side loud-fail on unfittable ctx/KV + closed --dev
  gate): APPROVED 10-22, NOT parked - implement after B12 lands (same
  fork; see BACKLOG B7a).
- B5 free-VRAM suite / B6 Part-3 trigger / B8 PQ2_0 sweep / B9 final
  report / B10 small follow-ups: status as in BACKLOG.md (parked /
  open-low). The final report (B9) should include the tensor-mode work
  when it lands.

## DONE (reference)

- [x] 10-19/10-20: KV ring overcommit guard + diagnostic (a3f36885c,
  355b4a2b3); MTP 3-part fix (0589b1e1a); PTQ1_0 SYCL byte-parallel
  kernel B11 (bd993a9d7 + 033a6f0ad + 5b80a3bb1, tg128 6.90 -> 26.72);
  fit-params draft-aware fix (48e9082ca); private/ knowledge library
  (02af668bf, e844b224a, 11c3d328a, 7faf42ec4, 835d728b3).
- [x] 10-22: tensor-mode root cause pinned (F20), meta model mapped
  (F21), H-B dead (F22), H-A designed (F23), PQ2_0/PTQ1_0 codec facts
  (F24), port-target clone located (F25). F15-F18 reconstructed/retained.
- [x] 10-22 (evening): user decisions recorded - H-A APPROVED (3nary
  first), B7a APPROVED, B7 preemption CLOSED; clone git state verified
  (no local commits, 3 user mods to preserve). Docs: BACKLOG (B7a),
  STATE, PLAN refreshed.
