INPUT_FILE="$1"
SERVER_PORT="$2"
OPTIMIZER="$3"
LR="$4"

RANK=0

WORLD_SIZE=$(wc -l < "$INPUT_FILE")

while IFS= read -r line
do
    if [ "${RANK}" = "0" ]
    then
        SERVER_IP="${line}"
    fi
    echo "${RANK} in ${WORLD_SIZE}: ${line}, tcp://${SERVER_IP}:${SERVER_PORT}"
    ssh "$line" "bash /home/ubuntu/src/ersgd/dist-ef-sgdm/imagenet/ssh_cmd.sh ${SERVER_IP} ${SERVER_PORT} ${WORLD_SIZE} ${RANK} ${OPTIMIZER} ${LR}" &
    RANK=$(( RANK+1 ))
done < "$INPUT_FILE"

