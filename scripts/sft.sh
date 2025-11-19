# 分布式环境变量（兼容 torchrun / accelerate / Slurm 等）
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}  # 注意：有些系统用 RANK 表示 node rank

# 自动推导 num_machines（节点数）
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
NUM_MACHINES=${WORLD_SIZE}
TOTAL_GPUS=$((NUM_MACHINES * NUM_GPUS_PER_NODE))

# 设置 CUDA 相关环境变量
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 清理冲突变量
unset NCCL_BLOCKING_WAIT
unset NCCL_ASYNC_ERROR_HANDLING

MODEL="${MODEL:-sdar}" 
DATASET="${DATASET:-math}"
RUN_NAME=WANDB_${MODEL}_sft_${DATASET}
export WANDB_PROJECT=$RUN_NAME
export WANDB_DIR=${WANDB_PROJECT}
mkdir -p "$WANDB_DIR"
export WANDB_MODE=offline

# SFT 参数（可通过环境变量传入）
OPTIMIZATION_DATA=${OPTIMIZATION_DATA:-sft_s1_1k_qwen3max}
PROJECT_NAME=${PROJECT_NAME:-sft_${MODEL}_8k}
SAVE_STEPS=${SAVE_STEPS:-10}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-50}
TRAINING_METHOD=${TRAINING_METHOD:-semi-ar}
MAX_GEN_LENGTH=${MAX_GEN_LENGTH:-8192}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zhuying/SDAR_Trainer/rl_trado_full_memory_8k/ckpt/epoch-10}
LBS=${LBS:-1}

export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/inspire/hdd/global_user/liuxiaoran-240108120089/zhuying/cache/${SCRIPT_NAME}
export RESUME_FROM_STEP=${RESUME_FROM_STEP:-None}
export RESUME_BATCH_COUNT=${RESUME_BATCH_COUNT:-0}

echo "[NUM NODES: $NUM_MACHINES] SFT training start ..."
# # 启动训练
accelerate launch \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --num_machines ${NUM_MACHINES} \
  --config_file accelerate_configs/${WORLD_SIZE}_node_8_gpus_deepspeed_zero2.yaml \
  train/sft.py \
  config=configs/sft.yaml \
  dataset.optimization_data=${OPTIMIZATION_DATA} \
  experiment.project=${PROJECT_NAME} \
  experiment.num_nodes=${NUM_MACHINES} \
  experiment.resume_from_step=${RESUME_FROM_STEP} \
  experiment.resume_batch_count=${RESUME_BATCH_COUNT} \
  experiment.save_steps=${SAVE_STEPS} \
  training.gradient_checkpointing_enable=true \
  training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  training.num_train_epochs=${NUM_TRAIN_EPOCHS} \
  training.method=${TRAINING_METHOD} \
  training.max_gen_length=${MAX_GEN_LENGTH} \
  training.shrink=1 \
  training.large_batch_size=${LBS} \
  model.pretrained_model=${PRETRAINED_MODEL}

echo "SFT training finished ..."