#!/bin/bash
export NAVSIM_DEVKIT_ROOT=$(pwd)
# use navsim_v1 code even if navsim_v2 is pip-installed in the env
export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:$PYTHONPATH

CHECKPOINT="${NAVSIM_EXP_ROOT}/Drive-JEPA-cache/drive_jepa_perception_based_agent_resnet34.ckpt"
# Or you can use the ckpt trained by yourself.

TRAIN_TEST_SPLIT=navtest

python -u $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=drive_jepa_perception_based_resnet_agent \
agent.config.latent=False \
worker=single_machine_thread_pool \
worker.max_workers=4 \
worker.use_process_pool=true \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=eval_reproduce_drive_jepa_perception_based_resnet_agent
