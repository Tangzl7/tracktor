#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# != 4 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_distributed_train_ascend.sh [DEVICE_NUM] [CONFIG_PATH] [RANK_TABLE_FILE] [LR]"
echo "for example: bash scripts/run_distributed_train_gpu.sh 8 ./default_config.yaml /home/hccl_8p_01234567_192.168.88.13.json 0.008"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

ulimit -u unlimited
export HCCL_CONNECT_TIMEOUT=600
export DEVICE_NUM=$1
export RANK_SIZE=$1
export RANK_TABLE_FILE=$3
CONFIG_PATH="$2"
BASE_LR="$4"

OUTPUT_PATH="run_distribute_train"

rm -rf "$OUTPUT_PATH"
mkdir "$OUTPUT_PATH"
cp "$CONFIG_PATH" "$OUTPUT_PATH"

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    # shellcheck disable=SC2115
    rm -rf $OUTPUT_PATH/$OUTPUT_PATH$i
    mkdir $OUTPUT_PATH/$OUTPUT_PATH$i
    cp ./*.py $OUTPUT_PATH/$OUTPUT_PATH$i
    cp ./*.yaml $OUTPUT_PATH/$OUTPUT_PATH$i
    cp ./*.ckpt $OUTPUT_PATH/$OUTPUT_PATH$i
    cp ./scripts/*.sh $OUTPUT_PATH/$OUTPUT_PATH$i
    cp -r ./src $OUTPUT_PATH/$OUTPUT_PATH$i
    cp -r ./MOT17DET $OUTPUT_PATH/$OUTPUT_PATH$i
    cd $OUTPUT_PATH/$OUTPUT_PATH$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    touch log.txt
    python train.py  \
    --config_path="$CONFIG_PATH" \
    --save_checkpoint_path="./" \
    --run_distribute=True \
    --device_target="Ascend" \
    --device_num="$RANK_SIZE" \
    --base_lr="$BASE_LR" > log.txt 2>&1 &
    cd ../../
done