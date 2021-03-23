#!/bin/bash

for((i=0; i<=2; i++))
do
python ../scripts/mbexp1.py -logdir ../log/horizon_exp/POPLINP_AVG_HalfCheetah/h25 \
    -env halfcheetah \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
    -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
    -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
    -o ctrl_cfg.opt_cfg.init_var 0.1 \
    -o ctrl_cfg.opt_cfg.plan_hor 25 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-P
done