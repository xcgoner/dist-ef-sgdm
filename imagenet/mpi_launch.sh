echo "tcp://${SERVER_IP}:${SERVER_PORT}"
sudo su -c "ulimit -n 1000000"
sudo python -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr="0.0.0.0" \
    --master_port=$SERVER_PORT benchmark_main.py ~/data/imagenet -a resnet50 -b 128 --lr 0.1 \
    --epochs 80 --save-dir ./ --world-size $OMPI_COMM_WORLD_SIZE --print-freq 600 --compress --signum --dist_backend gloo --weight-decay 1e-4 --momentum 0.9 --warm-up \
    --dist-url "tcp://${SERVER_IP}:${SERVER_PORT}" -j 4
