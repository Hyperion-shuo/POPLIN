#!/bin/bash


python ../scripts/mbexp.py -logdir ../log/debug/halfcheetah \
    -env gym_cheetah \
    -o exp_cfg.exp_cfg.ntrain_iters 200 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
    -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
    -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
    -o ctrl_cfg.opt_cfg.init_var 0.1 \
    -o ctrl_cfg.opt_cfg.plan_hor 30 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-P
