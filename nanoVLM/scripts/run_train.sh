export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LOG_NAME="nanoVLM"

torchrun \
    --standalone \
    --nnode=1 \
    --nproc_per_node=1 \
    ./nanoVLM/train.py
