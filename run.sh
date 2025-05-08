#!/bin/bash
conda env create -f env.yaml
# 启动 32B 模型（后台运行）
echo "启动 32B 模型..."
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup vllm serve \
    --model /home/aiscuser/zhengyu_blob_home/hugging_face_models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746 \
    --port 8001 \
    --tensor-parallel-size 7 > deepseek_32b.log 2>&1 &

# 等待大模型加载完成
sleep 30

# 小模型列表
declare -a MODEL_IDS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "microsoft/phi-4-mini-instruct"
)

# 依次测试小模型
for MODEL_ID in "${MODEL_IDS[@]}"; do
    echo "启动小模型: $MODEL_ID"

    # 启动小模型服务（后台运行）
    CUDA_VISIBLE_DEVICES=0 nohup vllm serve \
        --model "$MODEL_ID" \
        --port 8000 \
        --tensor-parallel-size 1 > small_model.log 2>&1 &

    # 等待模型加载
    sleep 30

    # 运行测试
    python test.py "$MODEL_ID"

    # 杀掉该小模型服务
    echo "关闭小模型: $MODEL_ID"
    pkill -f "vllm.*serve.*--model $MODEL_ID"

    # 稍作等待以释放端口
    sleep 10
done

echo "测试完成。"

# 可选：关闭 32B 模型
pkill -f "vllm.*serve.*DeepSeek-R1-Distill-Qwen-32B"

source activate vllm

conda activate vllm


python /home/aiscuser/zhengyu_blob_home/vllm_example/kkk.py