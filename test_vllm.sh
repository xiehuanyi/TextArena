CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 vllm serve \
    --model /home/aiscuser/zhengyu_blob_home/hugging_face_models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746 \
    --port 8001 \
    --tensor-parallel-size 7
