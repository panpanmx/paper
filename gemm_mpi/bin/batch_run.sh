#!/bin/bash

# ==============================================================================
# 弱扩展并行效率实验自动化执行脚本
# 主机函数: gemm_host_ASR
# 内核函数: gemm_dsp_AM_SM_d_opt / gemm_dsp_AM_SM_pipeline_opt
# 数据集:   512*512*512, 1024*1024*1024, 2048*2048*2048
# 节点数:   1 2 4 8 16 32 64 128 256 512
# 输出日志: ../log/final.log
# ==============================================================================

# 参数解析（支持命令行传参灵活运行单组或批量测试）
# 用法: ./batch_run.sh [kernel_name] [scale] [nodes...]
# 示例 1: ./batch_run.sh gemm_dsp_AM_SM_d_opt 512
# 示例 2: ./batch_run.sh gemm_dsp_AM_SM_pipeline_opt 1024 "1 2 4 8"
# 示例 3: ./batch_run.sh all all (按顺序执行全部 6 组正式实验)

TARGET_KERNEL=${1:-"gemm_dsp_AM_SM_d_opt"}
TARGET_SCALE=${2:-"512"}
CUSTOM_NODES=${3:-""}

PROJECT_DIR="/vol8/home/njust_ztc/hnu/jpp/gemm_mpi"
DATA_DIR="${PROJECT_DIR}/test_data"
LOG_FILE="${PROJECT_DIR}/log/final.log"
mkdir -p "${PROJECT_DIR}/log"

# 环境配置
export LD_LIBRARY_PATH="${PROJECT_DIR}/bin/libs:${LD_LIBRARY_PATH}"
export UCX_TLS=^cma

# 故障节点排除名单
EXCLUDE_NODES="cn[195,290,411,1677,4478,8202]"

# 默认节点列表
if [ -n "$CUSTOM_NODES" ]; then
    read -r -a DEFAULT_NODES <<< "$CUSTOM_NODES"
else
    DEFAULT_NODES=(1 2 4 8 16 32 64 128 256 512)
fi

run_single_group() {
    local kernel=$1
    local scale=$2
    shift 2
    local nodes=("$@")

    local matrixA_real="${DATA_DIR}/realH${scale}.bin"
    local matrixA_imag="${DATA_DIR}/imagH${scale}.bin"
    local matrixB_real="${DATA_DIR}/realH${scale}.bin"
    local matrixB_imag="${DATA_DIR}/imagH${scale}.bin"

    if [ ! -f "$matrixA_real" ] || [ ! -f "$matrixA_imag" ]; then
        echo "[ERROR] 数据集不存在: ${matrixA_real} 或 ${matrixA_imag}" | tee -a "${LOG_FILE}"
        return 1
    fi

    echo "" | tee -a "${LOG_FILE}"
    echo "================================================================================" | tee -a "${LOG_FILE}"
    echo "【实验名称】弱扩展并行效率实验" | tee -a "${LOG_FILE}"
    echo "【主机函数】gemm_host_ASR" | tee -a "${LOG_FILE}"
    echo "【内核函数】${kernel}" | tee -a "${LOG_FILE}"
    echo "【数据集大小】${scale}*${scale}*${scale}" | tee -a "${LOG_FILE}"
    echo "【测试节点列表】${nodes[*]} (每节点固定 4 进程，每进程 24 DSP 线程)" | tee -a "${LOG_FILE}"
    echo "【开始时间】$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
    echo "================================================================================" | tee -a "${LOG_FILE}"

    cd "${PROJECT_DIR}/bin" || exit 1

    for N in "${nodes[@]}"; do
        local total_ranks=$((N * 4))
        echo "--------------------------------------------------------------------------------" | tee -a "${LOG_FILE}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正在测试配置: -N ${N} (总进程数: ${total_ranks}, 线程数/进程: 24)" | tee -a "${LOG_FILE}"
        echo "--------------------------------------------------------------------------------" | tee -a "${LOG_FILE}"

        yhrun -p mt_work \
              -N "${N}" \
              --ntasks-per-node=4 \
              --cpus-per-task=1 \
              --mpi=pmi2 \
              --exclude="${EXCLUDE_NODES}" \
              ./gemm_host_ASR "${matrixA_real}" "${matrixA_imag}" "${matrixB_real}" "${matrixB_imag}" "${kernel}" 24 2>&1 | tee -a "${LOG_FILE}"

        sleep 1
    done

    echo "================================================================================" | tee -a "${LOG_FILE}"
    echo "【完成时间】$(date '+%Y-%m-%d %H:%M:%S') 实验组 [${kernel} | 规模 ${scale}*${scale}*${scale}] 完成" | tee -a "${LOG_FILE}"
    echo "================================================================================" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
}

# 编译前置检查
echo "=== 检查并准备可执行程序与设备固件 ==="
if [ ! -f "${PROJECT_DIR}/bin/gemm_host_ASR" ]; then
    echo "正在编译 gemm_host_ASR..."
    cd "${PROJECT_DIR}/host_code" && make gemm_host_ASR || exit 1
fi

if [ ! -f "${PROJECT_DIR}/bin/kernel_AS.dat" ] || [ ! -f "${PROJECT_DIR}/bin/kernel_pipeline.dat" ]; then
    echo "正在编译设备端固件..."
    cd "${PROJECT_DIR}/device_code" && make || exit 1
fi

# 根据参数分流执行
if [ "$TARGET_KERNEL" = "all" ] && [ "$TARGET_SCALE" = "all" ]; then
    echo ">>> 开始执行弱扩展并行效率全部 6 组实验 <<<"
    for scale in 512 1024 2048; do
        for kernel in gemm_dsp_AM_SM_d_opt gemm_dsp_AM_SM_pipeline_opt; do
            run_single_group "${kernel}" "${scale}" "${DEFAULT_NODES[@]}"
        done
    done
elif [ "$TARGET_KERNEL" = "all" ]; then
    for kernel in gemm_dsp_AM_SM_d_opt gemm_dsp_AM_SM_pipeline_opt; do
        run_single_group "${kernel}" "${TARGET_SCALE}" "${DEFAULT_NODES[@]}"
    done
elif [ "$TARGET_SCALE" = "all" ]; then
    for scale in 512 1024 2048; do
        run_single_group "${TARGET_KERNEL}" "${scale}" "${DEFAULT_NODES[@]}"
    done
else
    # 运行指定单组实验
    run_single_group "${TARGET_KERNEL}" "${TARGET_SCALE}" "${DEFAULT_NODES[@]}"
fi

echo "所有测试完成！正式数据已保存在: ${LOG_FILE}"
