SERVER_IP="$1"
SERVER_PORT="$2"
WORLD_SIZE="$3"
RANK="$4"
OPTIMIZER="$5"
LR="$6"

echo "tcp://${SERVER_IP}:${SERVER_PORT}"
ulimit -n 1000000
cd /home/ubuntu/src/ersgd/dist-ef-sgdm/imagenet
python3.6 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr="0.0.0.0" \
    --master_port=$SERVER_PORT benchmark_main.py ~/data/imagenet -a resnet50 -b 128 --lr $LR \
    --test_evaluate --optimizer "$OPTIMIZER" \
    --epochs 80 --save-dir ./ --world-size $WORLD_SIZE --print-freq 600 --compress --signum --dist_backend gloo --weight-decay 1e-4 --momentum 0.9 --warm-up \
    --dist-url "tcp://${SERVER_IP}:${SERVER_PORT}" -j 4