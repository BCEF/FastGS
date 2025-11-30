#!/bin/bash

# ==================== 多帧序列训练配置 ====================

# 数据根目录和输出根目录
DATA_ROOT="/home/momo/Desktop/1112/test_data"
OUTPUT_ROOT="/home/momo/Desktop/1112/test_data/output_fg"

# 帧范围设置
START_FRAME=31
END_FRAME=60
FRAME_PREFIX="frame"      # 帧文件夹前缀，如 frame_000, frame_001
FRAME_DIGITS=6             # 帧编号位数，3 表示 000, 001, ...

# GPU 设置
GPU_ID=0

# 图像文件夹和分辨率
IMAGES="images"
RESOLUTION=1

# FastGS 基础参数
ITERATIONS=30000
OPTIMIZER_TYPE="default"
TEST_ITERATIONS=30000
SAVE_ITERATIONS=30000

# ============ 高质量核心参数 ============
DENSIFICATION_INTERVAL=100
GRAD_ABS_THRESH=0.0004
HIGHFEATURE_LR=0.02
DENSE=0.01
GRAD_THRESH=0.0002
LOSS_THRESH=0.1
MULT=1.0

# ============ SUMO 时间正则化参数 ============
# 是否对后续帧启用时间正则化
ENABLE_TEMPORAL_REG=true
LAMBDA_TEMP=0.1
LAMBDA_SMOOTH=0.05
ALPHA_TEMP=0.5
ALPHA_SMOOTH=0.5

# ============ SIBR Viewer 实时查看设置 ============
ENABLE_VIEWER=true         # 是否启用实时查看
VIEWER_IP="127.0.0.1"
VIEWER_PORT=6009

# ==================== 辅助函数 ====================

# 格式化帧编号
format_frame_id() {
    printf "%0${FRAME_DIGITS}d" $1
}

# 构建单帧训练命令
build_train_cmd() {
    local source_path=$1
    local model_path=$2
    local use_reg=$3
    local prev_ply=$4

    local CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py"
    CMD="${CMD} -s ${source_path}"
    CMD="${CMD} -m ${model_path}"
    CMD="${CMD} -i ${IMAGES}"

    if [ -n "$RESOLUTION" ]; then
        CMD="${CMD} -r ${RESOLUTION}"
    fi

    # ============ 启用实时查看 ============
    if [ "$ENABLE_VIEWER" = true ]; then
        CMD="${CMD} --websockets"
        CMD="${CMD} --ip ${VIEWER_IP}"
        CMD="${CMD} --port ${VIEWER_PORT}"
    fi

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

    # 时间正则化参数
    if [ "$use_reg" = true ]; then
        CMD="${CMD} --use_reg"
        CMD="${CMD} --lambda_temp ${LAMBDA_TEMP}"
        CMD="${CMD} --lambda_smooth ${LAMBDA_SMOOTH}"
        CMD="${CMD} --alpha_temp ${ALPHA_TEMP}"
        CMD="${CMD} --alpha_smooth ${ALPHA_SMOOTH}"

        if [ -n "$prev_ply" ] && [ -f "$prev_ply" ]; then
            CMD="${CMD} --init_ply_path ${prev_ply}"
        fi
    fi

    echo "$CMD"
}

# ==================== 主训练循环 ====================

echo "=========================================="
echo "FastGS Multi-Frame Sequence Training"
echo "=========================================="
echo "Data Root:    ${DATA_ROOT}"
echo "Output Root:  ${OUTPUT_ROOT}"
echo "Frame Range:  ${START_FRAME} - ${END_FRAME}"
echo "GPU:          ${GPU_ID}"
echo "Temporal Reg: ${ENABLE_TEMPORAL_REG}"
echo "=========================================="
echo ""

# 创建输出根目录
mkdir -p ${OUTPUT_ROOT}

# 记录开始时间
TOTAL_START_TIME=$(date +%s)

for i in $(seq ${START_FRAME} ${END_FRAME}); do
    FRAME_ID=$(format_frame_id $i)
    SOURCE="${DATA_ROOT}/${FRAME_PREFIX}${FRAME_ID}"
    MODEL="${OUTPUT_ROOT}/${FRAME_PREFIX}${FRAME_ID}"

    echo "=========================================="
    echo "Training Frame ${FRAME_ID} ($(($i - $START_FRAME + 1))/$(($END_FRAME - $START_FRAME + 1)))"
    echo "=========================================="
    echo "Source: ${SOURCE}"
    echo "Model:  ${MODEL}"

    # 检查源数据是否存在
    if [ ! -d "$SOURCE" ]; then
        echo "Warning: Source directory not found: ${SOURCE}"
        echo "Skipping frame ${FRAME_ID}..."
        continue
    fi

    # 记录帧开始时间
    FRAME_START_TIME=$(date +%s)

    # 判断是否使用时间正则化
    USE_REG=false
    PREV_PLY=""

    if [ $i -gt ${START_FRAME} ] && [ "$ENABLE_TEMPORAL_REG" = true ]; then
        PREV_ID=$(format_frame_id $((i-1)))
        PREV_PLY="${OUTPUT_ROOT}/${FRAME_PREFIX}${PREV_ID}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"

        if [ -f "$PREV_PLY" ]; then
            USE_REG=true
            echo "Using temporal regularization"
            echo "Prev PLY: ${PREV_PLY}"
        else
            echo "Warning: Previous PLY not found: ${PREV_PLY}"
            echo "Training without temporal regularization"
        fi
    else
        echo "First frame or temporal reg disabled, training without constraint"
    fi

    # 构建并执行训练命令
    CMD=$(build_train_cmd "$SOURCE" "$MODEL" "$USE_REG" "$PREV_PLY")
    
    echo ""
    echo "Command: ${CMD}"
    echo ""

    eval ${CMD}

    # 计算帧训练时间
    FRAME_END_TIME=$(date +%s)
    FRAME_DURATION=$((FRAME_END_TIME - FRAME_START_TIME))
    echo ""
    echo "Frame ${FRAME_ID} completed in ${FRAME_DURATION} seconds"
    echo ""

done

# 计算总时间
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo "=========================================="
echo "All Frames Training Complete"
echo "=========================================="
echo "Total frames: $(($END_FRAME - $START_FRAME + 1))"
echo "Total time:   ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "=========================================="