#!/bin/bash


# python ../spinup/algos/tf1/sac/my_sac.py --env gym_ant \
# --exp_name sac_mbrlant

# python ../spinup/algos/tf1/sac/my_sac.py --env gym_hopper \
# --exp_name sac_mbrlhopper

python ../spinup/algos/tf1/sac/my_sac.py --env gym_walker2d \
--exp_name sac_mbrlwalker

python ../spinup/algos/tf1/sac/my_sac.py --env gym_swimmer \
--exp_name sac_mbrlswimmer

