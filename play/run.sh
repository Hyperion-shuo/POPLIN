#!bin/bash
# for((i=1; i<=100; i++))
# do
#     sum=$(($sum + $i))
#     i=$(($i + 1))
# done
# echo "sum(1--100):" $sum



# mycount=0; while (( $mycount < 2 )); do  python hello_world.py;((mycount=$mycount+1)); done;
# mycount=0; while (( mycount < 10 )); do echo "mycount=$mycount"; ((mycount=$mycount+1)); done;



for((i=1; i<=10; i++))
do
     python hello_world.py
done
