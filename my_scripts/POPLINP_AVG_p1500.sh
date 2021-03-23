#!/bin/bash

for((i=0; i<=1; i++))
do
python ../scripts/mbexp.py -logdir ../log/popsize_exp_h30/POPLINP_AVG_Cheetah/p1500 \
    -env gym_cheetah \
    -o exp_cfg.exp_cfg.ntrain_iters 200 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
    -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
    -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
    -o ctrl_cfg.opt_cfg.init_var 0.1 \
    -o ctrl_cfg.opt_cfg.plan_hor 30 \
    -o ctrl_cfg.opt_cfg.cfg.popsize 1500 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-P
done