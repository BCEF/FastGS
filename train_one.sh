#!/bin/bash

# ==================== 配置参数 ====================

# 数据路径和输出路径
SOURCE_PATH="/home/momo/Desktop/1112/keyframe/"
MODEL_PATH="/home/momo/Desktop/1112/keyframe/output_fg"
IMAGES="images"

# GPU 设置
GPU_ID=0

# FastGS 基础参数
ITERATIONS=30000
DENSIFICATION_INTERVAL=500
OPTIMIZER_TYPE="default"
TEST_ITERATIONS=30000
SAVE_ITERATIONS=30000

# FastGS 可调参数
HIGHFEATURE_LR=0.01
DENSE=0.01
GRAD_ABS_THRESH=0.0004
GRAD_THRESH=0.0002
LOSS_THRESH=0.1
MULT=1.0

# SUMO 时间正则化参数
USE_REG=false                    # 是否启用时间正则化
PREV_PLY_PATH=""                 # 前一帧 PLY 路径
LAMBDA_TEMP=0.1
LAMBDA_SMOOTH=0.05
ALPHA_TEMP=0.5
ALPHA_SMOOTH=0.5

# ==================== 构建命令 ====================

CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py"
CMD="${CMD} -s ${SOURCE_PATH}"
CMD="${CMD} -m ${MODEL_PATH}"
CMD="${CMD} -i ${IMAGES}"
CMD="${CMD} --eval"
CMD="${CMD} --iterations ${ITERATIONS}"
CMD="${CMD} --densification_interval ${DENSIFICATION_INTERVAL}"
CMD="${CMD} --optimizer_type ${OPTIMIZER_TYPE}"
CMD="${CMD} --test_iterations ${TEST_ITERATIONS}"
CMD="${CMD} --save_iterations ${SAVE_ITERATIONS}"
CMD="${CMD} --highfeature_lr ${HIGHFEATURE_LR}"
CMD="${CMD} --dense ${DENSE}"
CMD="${CMD} --grad_abs_thresh ${GRAD_ABS_THRESH}"
CMD="${CMD} --grad_thresh ${GRAD_THRESH}"
CMD="${CMD} --loss_thresh ${LOSS_THRESH}"
CMD="${CMD} --mult ${MULT}"

# SUMO 时间正则化
if [ "$USE_REG" = true ]; then
    CMD="${CMD} --use_reg"
    CMD="${CMD} --lambda_temp ${LAMBDA_TEMP}"
    CMD="${CMD} --lambda_smooth ${LAMBDA_SMOOTH}"
    CMD="${CMD} --alpha_temp ${ALPHA_TEMP}"
    CMD="${CMD} --alpha_smooth ${ALPHA_SMOOTH}"
    
    if [ -n "$PREV_PLY_PATH" ]; then
        CMD="${CMD} --init_ply_path ${PREV_PLY_PATH}"
    fi
fi

# ==================== 执行训练 ====================

echo "=========================================="
echo "FastGS Training"
echo "=========================================="
echo "Source: ${SOURCE_PATH}"
echo "Model:  ${MODEL_PATH}"
echo "GPU:    ${GPU_ID}"
if [ "$USE_REG" = true ]; then
    echo "Temporal Reg: Enabled"
    echo "  Prev PLY: ${PREV_PLY_PATH}"
fi
echo "=========================================="
echo ""

eval ${CMD}

echo ""
echo "Training complete."