#!/bin/bash

for((i=0; i<=2; i++))
do
python ../scripts/mbexp2.py -logdir ../log/horizon_exp/POPLINP_AVG_Cheetah/h50 \
    -env gym_cheetah \
    -o exp_cfg.exp_cfg.ntrain_iters 200 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
    -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
    -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
    -o ctrl_cfg.opt_cfg.init_var 0.1 \
    -o ctrl_cfg.opt_cfg.plan_hor 50  \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-P
done

