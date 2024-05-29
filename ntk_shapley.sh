#!/bin/sh

# Define variables at the front
DATASET="$1"  # {sst2, mr, mrpc, rte, mnli} are supported
CUDA_DEVICE="$2"
NUM_DP="$3"
POINT="$4"  # {True, False}; whether output instance score for each test (True)/instance score for who test sets
FILE_PATH="$5"
SEED=2023
TMC_SEED=2023
APPROXIMATE="inv"  # {inv, None, diagonal}; normally is inv which is the Block Inversion implementation
EARLY_STOPPING="True"  # {True, False}; normally is True which is TMC (or else is MC)
PYTHON_SCRIPT=vinfo/ntk.py
VAL_SAMPLE_NUM=1000
TMC_ITER=200
PROMPT=True
FILENAME="ntk_prompt"

# Run the Python script
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python $PYTHON_SCRIPT \
  --yaml_path="configs/dshap/$DATASET/$FILENAME.yaml" \
  --num_dp=$NUM_DP \
  --val_sample_num=$VAL_SAMPLE_NUM \
  --tmc_iter=$TMC_ITER \
  --dataset_name=$DATASET \
  --file_path=$FILE_PATH \
  --prompt=$PROMPT \
  --seed=$SEED \
  --tmc_seed=$TMC_SEED \
  --approximate=$APPROXIMATE \
  --per_point=$POINT \
  --early_stopping=$EARLY_STOPPING