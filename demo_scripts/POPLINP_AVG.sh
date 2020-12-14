#!/bin/bash
# The following script will run POPLIN-P using the AVG training methods on halfcheetah

# python ../scripts/mbexp.py -logdir ../log/POPLINP_AVG_pusher -env pusher \
#     -o exp_cfg.exp_cfg.ntrain_iters 50 \
#     -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
#     -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
#     -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
#     -o ctrl_cfg.opt_cfg.init_var 0.1 \
#     -o ctrl_cfg.cem_cfg.test_policy 1 \
#     -ca model-type PE -ca prop-type E \
#     -ca opt-type POPLIN-P

# python ../scripts/mbexp.py -logdir ../log/POPLINP_AVG_pusher -env pusher \
#     -o exp_cfg.exp_cfg.ntrain_iters 50 \
#     -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
#     -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
#     -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
#     -o ctrl_cfg.opt_cfg.init_var 0.1 \
#     -o ctrl_cfg.cem_cfg.test_policy 1 \
#     -ca model-type PE -ca prop-type E \
#     -ca opt-type POPLIN-P

# python ../scripts/mbexp.py -logdir ../log/POPLINP_AVG_pusher -env pusher \
#     -o exp_cfg.exp_cfg.ntrain_iters 50 \
#     -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
#     -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
#     -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
#     -o ctrl_cfg.opt_cfg.init_var 0.1 \
#     -o ctrl_cfg.cem_cfg.test_policy 1 \
#     -ca model-type PE -ca prop-type E \
#     -ca opt-type POPLIN-P

# python ../scripts/mbexp.py -logdir ../log/POPLINP_AVG_reacher -env reacher \
#     -o exp_cfg.exp_cfg.ntrain_iters 50 \
#     -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
#     -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
#     -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
#     -o ctrl_cfg.opt_cfg.init_var 0.1 \
#     -o ctrl_cfg.cem_cfg.test_policy 1 \
#     -ca model-type PE -ca prop-type E \
#     -ca opt-type POPLIN-P

# python ../scripts/mbexp.py -logdir ../log/POPLINP_AVG_reacher -env reacher \
#     -o exp_cfg.exp_cfg.ntrain_iters 50 \
#     -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
#     -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
#     -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
#     -o ctrl_cfg.opt_cfg.init_var 0.1 \
#     -o ctrl_cfg.cem_cfg.test_policy 1 \
#     -ca model-type PE -ca prop-type E \
#     -ca opt-type POPLIN-P

# python ../scripts/mbexp.py -logdir ../log/POPLINP_AVG_reacher -env reacher \
#     -o exp_cfg.exp_cfg.ntrain_iters 50 \
#     -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
#     -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
#     -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
#     -o ctrl_cfg.opt_cfg.init_var 0.1 \
#     -o ctrl_cfg.cem_cfg.test_policy 1 \
#     -ca model-type PE -ca prop-type E \
#     -ca opt-type POPLIN-P

python ../scripts/mbexp.py -logdir ../log/POPLINP_AVG_halfcheetah -env halfcheetah \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
    -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
    -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
    -o ctrl_cfg.opt_cfg.init_var 0.1 \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-P

python ../scripts/mbexp.py -logdir ../log/POPLINP_AVG_halfcheetah -env halfcheetah \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
    -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
    -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
    -o ctrl_cfg.opt_cfg.init_var 0.1 \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -ca model-type PE -ca prop-type E \
    -ca opt-type POPLIN-P
