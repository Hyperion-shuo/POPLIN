#!/bin/bash


# python ../spinup/algos/tf1/td3/my_td3.py --env gym_hopper \
# --exp_name td3_mbrlhopper

python ../spinup/algos/tf1/td3/my_td3.py --env gym_walker2d \
--exp_name td3_mbrlwalker

python ../spinup/algos/tf1/td3/my_td3.py --env gym_swimmer \
--exp_name td3_mbrlswimmer

# python ../spinup/algos/tf1/td3/my_td3.py --env gym_ant \
# --exp_name td3_mbrlant