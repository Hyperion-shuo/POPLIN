#!/bin/bash
# The following script will run POPLIN-A with REPLAN methods on halfcheetah

python ../scripts/mbexp.py -logdir ../log/POPLINA_REPLAN_pusher \
    -env pusher \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINA-REPLAN \
    -o ctrl_cfg.cem_cfg.training_scheme BC-AI \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-A

python ../scripts/mbexp.py -logdir ../log/POPLINA_REPLAN_pusher \
    -env pusher \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINA-REPLAN \
    -o ctrl_cfg.cem_cfg.training_scheme BC-AI \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-A

python ../scripts/mbexp.py -logdir ../log/POPLINA_REPLAN_pusher \
    -env pusher \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINA-REPLAN \
    -o ctrl_cfg.cem_cfg.training_scheme BC-AI \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-A

python ../scripts/mbexp.py -logdir ../log/POPLINA_REPLAN_reacher \
    -env reacher \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINA-REPLAN \
    -o ctrl_cfg.cem_cfg.training_scheme BC-AI \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-A

python ../scripts/mbexp.py -logdir ../log/POPLINA_REPLAN_reacher \
    -env reacher \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINA-REPLAN \
    -o ctrl_cfg.cem_cfg.training_scheme BC-AI \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-A

python ../scripts/mbexp.py -logdir ../log/POPLINA_REPLAN_reacher \
    -env reacher \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINA-REPLAN \
    -o ctrl_cfg.cem_cfg.training_scheme BC-AI \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-A

python ../scripts/mbexp.py -logdir ../log/POPLINA_REPLAN_halfcheetah \
    -env halfcheetah \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINA-REPLAN \
    -o ctrl_cfg.cem_cfg.training_scheme BC-AI \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-A


python ../scripts/mbexp.py -logdir ../log/POPLINA_REPLAN_halfcheetah \
    -env halfcheetah \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINA-REPLAN \
    -o ctrl_cfg.cem_cfg.training_scheme BC-AI \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-A

python ../scripts/mbexp.py -logdir ../log/POPLINA_REPLAN_halfcheetah \
    -env halfcheetah \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINA-REPLAN \
    -o ctrl_cfg.cem_cfg.training_scheme BC-AI \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-A
