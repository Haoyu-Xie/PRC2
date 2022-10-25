# !/bin/bash

job_name=$1
train_gpu=$2
num_node=$3
epoch=$4
command=$5
total_process=$((train_gpu*num_node))

mkdir -p log



while true
do
    port=$(( $RANDOM % 300 + 23450 ))
    
    GLOG_vmodule=MemcachedClient=-1 \
    srun --partition=VA \
    --mpi=pmi2 -n$total_process \
    --gres=gpu:$train_gpu \
    --ntasks-per-node=$train_gpu \
    --job-name=$job_name \
    --kill-on-bad-exit=1 \
    --cpus-per-task=7 \
    $command --port $port 2>&1|tee -a log/$job_name.log

    if grep "Epoch: \[$epoch\]" log/$job_name.log
    then
        echo "done"
        break
    fi
done
