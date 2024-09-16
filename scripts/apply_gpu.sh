#!/bin/bash

# apply GPU from MSU HPC
salloc --gres=gpu:v100:4 --time=02:00:00 --account=cmse --nodes=1 --ntasks-per-node=4 --cpus-per-task=5 --mem=140GB