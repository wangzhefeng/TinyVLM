export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LOG_NAME="nanoVLM"

python -u ./nanoVLM/generate.py \
    --checkpoint ./nanoVLM/saved_results/nanoVLM-450M
