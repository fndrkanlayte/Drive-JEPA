# Experience retrieval — Phase A tooling

Data tooling for the offline "experience module" experiment on
`drive_jepa_perception_based` (NAVSIM v1). Exports the 32 candidate trajectories
+ latents per scene, labels them with the training PDMScorer, and runs two
diagnostic checks (headroom, oracle-kNN). **No model training here.**

## Environment

Same env as the rest of `navsim_v1` (`scripts/env.sh`):

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT=<...>/navsim_workspace/dataset/maps
export NAVSIM_EXP_ROOT=<...>/navsim_workspace/exp
export OPENSCENE_DATA_ROOT=<...>/navsim_workspace/dataset
export NAVSIM_DEVKIT_ROOT=<repo>/navsim_v1
export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
cd $NAVSIM_DEVKIT_ROOT   # important: the agent loads ./data/8192.npy relative to CWD
```

Conda env from the repo requirements + `scikit-learn` + `shapely>=2`.
Single GPU (batch inference); labeling is CPU multiprocessing.

Expected artifacts on disk:

- feature cache: `$NAVSIM_EXP_ROOT/train_drive_jepa_perception_based_cache/<log>/<token>/drive_jepa_{feature,target}.gz`
- navtrain metric cache: `$NAVSIM_EXP_ROOT/Drive-JEPA-cache/train_metric_cache/...`
- navtest metric cache: `$NAVSIM_EXP_ROOT/Drive-JEPA-cache/metric_cache/...`
- checkpoint: e.g. `$NAVSIM_EXP_ROOT/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl.ckpt`

Disk usage per scene: export ≈ 0.07 MB (or ≈ 0.2 MB with `--save_tokens`),
labels ≈ 0.02 MB. 3000 scenes ≈ 0.3–0.7 GB.

## Minimal first run (smoke test before the full export)

```bash
# 1. export 20 scenes with the consistency check
python scripts/experience/export_candidates.py \
    --checkpoint $CKPT \
    --feature_cache_dir $NAVSIM_EXP_ROOT/train_drive_jepa_perception_based_cache \
    --out_dir $NAVSIM_EXP_ROOT/experience/navtrain_export_smoke \
    --batch_size 8 --max_scenes 20 --check_consistency 20
# 2. label those 20 scenes
python scripts/experience/label_candidates.py \
    --export_dir .../navtrain_export_smoke \
    --metric_cache_dir $NAVSIM_EXP_ROOT/Drive-JEPA-cache/train_metric_cache \
    --cache_type train --out_dir .../navtrain_labels_smoke --workers 8
# 3. audit — sanity fraction should be high
python scripts/experience/audit_labels.py --labels_dir .../navtrain_labels_smoke
# then run the full pipeline below
```

## Commands

```bash
# A1 export candidates (GPU). navtrain from the feature cache:
python scripts/experience/export_candidates.py \
    --checkpoint $CKPT \
    --feature_cache_dir $NAVSIM_EXP_ROOT/train_drive_jepa_perception_based_cache \
    --out_dir $NAVSIM_EXP_ROOT/experience/navtrain_export \
    --batch_size 8 --check_consistency 20
# navtest from raw logs/sensors (all 4 cameras needed):
python scripts/experience/export_candidates.py \
    --checkpoint $CKPT \
    --navsim_log_path $OPENSCENE_DATA_ROOT/navsim_logs/test \
    --sensor_blobs_path $OPENSCENE_DATA_ROOT/sensor_blobs/test \
    --out_dir $NAVSIM_EXP_ROOT/experience/navtest_export \
    --batch_size 8 --check_consistency 20

# A2 label with PDMScorer (CPU, resumable, --workers ~ ncores)
python scripts/experience/label_candidates.py \
    --export_dir $NAVSIM_EXP_ROOT/experience/navtrain_export \
    --metric_cache_dir $NAVSIM_EXP_ROOT/Drive-JEPA-cache/train_metric_cache \
    --cache_type train --out_dir $NAVSIM_EXP_ROOT/experience/navtrain_labels --workers 16
python scripts/experience/label_candidates.py \
    --export_dir $NAVSIM_EXP_ROOT/experience/navtest_export \
    --metric_cache_dir $NAVSIM_EXP_ROOT/Drive-JEPA-cache/metric_cache \
    --cache_type official --out_dir $NAVSIM_EXP_ROOT/experience/navtest_labels --workers 16

# A3 audit (recompute check needs the run_pdm_score.py csv for navtest)
python scripts/experience/audit_labels.py --labels_dir .../navtest_labels \
    --official_csv <path-to-run_pdm_score.csv>
python scripts/experience/audit_labels.py --labels_dir .../navtrain_labels

# A4 headroom
python scripts/experience/headroom.py --labels_dir .../navtrain_labels --name navtrain

# A5 oracle kNN check — run once per feature set (timing is the
# leakage-safe default; 'full' additionally uses min_dist/overlap)
python scripts/experience/oracle_knn_check.py \
    --labels_dir .../navtrain_labels --export_dir .../navtrain_export \
    --out_dir $NAVSIM_EXP_ROOT/experience/oracle_timing --feature_set timing
python scripts/experience/oracle_knn_check.py \
    --labels_dir .../navtrain_labels --export_dir .../navtrain_export \
    --out_dir $NAVSIM_EXP_ROOT/experience/oracle_full --feature_set full
```

## Runtime knobs

- `--max_scenes N` (export, label): smoke tests. `--token_list file` for subsets.
- `--save_tokens` (export): also store per-pose tokens (32,8,256) fp16.
- `--check_consistency N` (export): re-forward N scenes, assert selected traj.
- `--workers` (label): process pool size. `cache_type official` recomputes the
  PDM reference trajectory's progress per scene (official cache lacks
  `pdm_progress`), which costs an extra simulation per scene.
- `--ratios/--seed` (oracle): memory/query_train/query_val split by log_name.

## First-run checklist

1. A1: run `--check_consistency 20` — asserts exported selected traj ==
   re-computed `trajectory` (argmax pdm_score) within 1e-5.
2. A1: confirm `pdm_score` in exports matches `sigmoid(pred_logit)[..., -1]`
   (double_score is off in the released config).
3. A2: label ~20 scenes first; check `audit_labels.py` sanity fraction
   (NC==0 by vehicle → main vehicle att_collision & min_dist≈0) is high.
4. A3 (navtest): mean|diff| vs official csv should be ≈0; nonzero diffs are
   documented there (train scorer vs official evaluator).
5. A5: the `oracle_*` predictors use the query's TRUE future — an upper bound,
   not deployable signal. `parametric_desc` vs `oracle_knn` tells you if
   retrieval adds anything beyond a parametric read of the same info.

## Q11 candidate-level diagnostic (navtrain query_val, 810 scenes; pre-registered)

`q11_diag.py` evaluates candidate-scoring methods on query_val only
(navtest is never touched). Pre-registered definitions:

Scene subsets (evaluated on scenes, masks fixed before looking at results):
- `all`: every query_val scene.
- `S_err`: the native-argmax candidate has NC<1 or TTC<1, AND the scene
  contains at least one candidate with NC==1 AND TTC==1.
- `S_flip`: the scene's candidates contain both unsafe (NC<1 or TTC<1)
  and fully-safe (NC==1 AND TTC==1) outcomes.
- `S_int_<itype>` (itype in SAME_DIR / CROSSING / ONCOMING): the
  native-argmax candidate's noatt main descriptor has conflict=1,
  |dt_enter|<2s, and that itype.
- `S_int_*_turn`: same, plus ego heading change >30 deg over the
  candidate horizon (|wrap(theta_last - theta_first)| of the argmax
  candidate's proposal).

Candidate density tertiles (rarity in descriptor space):
- Descriptor space = the timing descriptor feature vector
  (TIMING_FIELDS + ego_speed + itype one-hot, i.e. rows["desc"]),
  standardized by memory mean/std, NaN -> 0.
- density(candidate) = mean Euclidean distance to the k=20 nearest
  MEMORY rows (query_train/query_val excluded), excluding same-log
  memory rows.
- Tertile edges are the 33.3/66.7 percentiles of the same density
  computed for MEMORY rows themselves (each memory row vs its 20NN in
  memory, same-log excluded) — thresholds fixed from memory, not from
  the queries.
- A scene's tertile = the tertile of its native-argmax candidate's
  density. Candidate-pooled metrics use each candidate's own tertile.

Methods (score per candidate; higher = riskier):
- native_pdm: 1 - pdm_score.
- noexp / noexp_int / random / shuffle / retrieval / retrieval_int /
  pred_desc_retrieval_pp: query_val risk npz columns (mean over seeds).
- oracle_knn: mean label of the 16 nearest memory rows in true-desc
  space (same desc features as density; same-log excluded). Timing
  fields only, no GT fields.
- parametric_desc: logistic regression desc -> label, trained on
  memory rows only, applied to query_val.

Metrics per subset x tertile (95% CI by scene bootstrap):
- within-scene AUPRC for nc_unsafe and ttc_bad (per-scene AP, averaged
  over scenes with >=1 positive).
- mean rank (1..32) of the safest candidate under the method ordering
  (safest = argmax labelled final among NC==1 & TTC==1 candidates;
  ordering by descending -risk or pdm_score).
- top-1 unsafe rate: fraction of scenes where the argmin-risk (or
  argmax-pdm) pick has NC<1 or TTC<1.
- n scenes, n positive candidates per label.

## EWM-JEPA Step 2: latent + outcome caches (new direction)

Frozen-encoder latent cache for the Experience-Conditioned Latent World Model:

- `export_latents.py` — per-scene `<log>/<token>.npz` with `image_feature`
  (512,256) f16 (scene-level z_t), `bev_feature` (32,8,256) f16,
  `proposal_feature`, `proposals`, `pred_logit`, `pdm_score`, `ego_status`,
  `trajectory` (expert), `lidar2img`, `img_shape` (both needed to rebuild the
  backbone tuple for external scoring). Resumable, sharded by log.
- `experience/world_model.py` — `score_external_trajectories(model,
  image_feature, ego_status, trajs)`: scores arbitrary trajectories by
  re-running the shared `Bev_refiner` rounds with `pose=tau_ext` on a cached
  `image_feature`; `pack_image_feature` rebuilds the backbone tuple from npz.
- `test_score_external.py` — verifies argmax agreement >=99% when fed the
  model's own proposals, plus a reversed-poses control.
- `compute_anchor_subscores.py` — per-scene (n_anchor,6) PDM subscores over
  the 8192-anchor vocabulary (v1 port of calc_anchors_scores, no
  scores_index dependency). CPU/multiprocess/resumable; `--anchor_subset` for
  a fixed random subset.
- `build_future_map.py` — token -> ~+4s same-log token map from
  `ego_state.time_point.time_us` (for z_{t+H} targets); reports coverage.
