#!/bin/bash
# The following script will run POPLIN-A with INIT methods on halfcheetah


# each env two run
for((i=1; i<=2; i++))
do
python ../scripts/mbexp.py -logdir ../log/POPLINA_INIT_walker2d \
    -env gym_walker2d \
    -o exp_cfg.exp_cfg.ntrain_iters 200 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINA-INIT \
    -o ctrl_cfg.cem_cfg.training_scheme BC-AI \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-A

python ../scripts/mbexp.py -logdir ../log/POPLINA_INIT_ant \
    -env gym_ant \
    -o exp_cfg.exp_cfg.ntrain_iters 200 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINA-INIT \
    -o ctrl_cfg.cem_cfg.training_scheme BC-AI \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-A

python ../scripts/mbexp.py -logdir ../log/POPLINA_INIT_hopper \
    -env gym_hopper \
    -o exp_cfg.exp_cfg.ntrain_iters 200 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINA-INIT \
    -o ctrl_cfg.cem_cfg.training_scheme BC-AI \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-A

python ../scripts/mbexp.py -logdir ../log/POPLINA_INIT_cheetah \
    -env gym_cheetah \
    -o exp_cfg.exp_cfg.ntrain_iters 200 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINA-INIT \
    -o ctrl_cfg.cem_cfg.training_scheme BC-AI \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-A
done