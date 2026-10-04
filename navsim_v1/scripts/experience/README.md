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
