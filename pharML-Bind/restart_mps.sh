#!/bin/bash
#source /cray/css/users/jbalma/bin/setup_env_cuda10_osprey_V100.sh
HOSTNAME=$(( hostname ))
echo "$HOSTNAME Killing MPS..."
#which nvidia-cuda-mps-control
echo "quit" | nvidia-cuda-mps-control
echo "$HOSTNAME Done KILL"
echo
sleep 5
echo "$HOSTNAME Restarting MPS..."
taskset -c 0 nvidia-cuda-mps-control -d
echo "$HOSTNAME Done RESTART"
