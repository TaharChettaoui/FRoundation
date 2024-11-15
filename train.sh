export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
--node_rank=0 --master_addr="localhost" --master_port=12355 src/train.py --debug=False
