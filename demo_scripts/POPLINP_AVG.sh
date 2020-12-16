#!/bin/bash
# The following script will run POPLIN-P using the AVG training methods on halfcheetah

for((i=1; i<=2; i++))
do
python ../scripts/mbexp.py -logdir ../log/POPLINP_AVG_walker2d -env gym_walker2d \
    -o exp_cfg.exp_cfg.ntrain_iters 200 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
    -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
    -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
    -o ctrl_cfg.opt_cfg.init_var 0.1 \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-P

python ../scripts/mbexp.py -logdir ../log/POPLINP_AVG_ant -env gym_ant \
    -o exp_cfg.exp_cfg.ntrain_iters 200 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
    -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
    -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
    -o ctrl_cfg.opt_cfg.init_var 0.1 \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-P

python ../scripts/mbexp.py -logdir ../log/POPLINP_AVG_hopper -env gym_hopper \
    -o exp_cfg.exp_cfg.ntrain_iters 200 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
    -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
    -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
    -o ctrl_cfg.opt_cfg.init_var 0.1 \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-P

python ../scripts/mbexp.py -logdir ../log/POPLINP_AVG_cheetah -env gym_cheetah \
    -o exp_cfg.exp_cfg.ntrain_iters 200 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
    -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
    -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
    -o ctrl_cfg.opt_cfg.init_var 0.1 \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-P
done


