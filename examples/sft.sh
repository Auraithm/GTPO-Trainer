# cd xxx/DiRL
export MODEL=sdar
export DATASET=math
export OPTIMIZATION_DATA=xxx
export PROJECT_NAME=sft_sdar_8k
export SAVE_STEPS=10
export GRADIENT_ACCUMULATION_STEPS=2
export NUM_TRAIN_EPOCHS=30
export TRAINING_METHOD=semi-ar
export MAX_GEN_LENGTH=8192
export PRETRAINED_MODEL=xxx/SDAR-8B-Chat

bash scripts/sft.sh
