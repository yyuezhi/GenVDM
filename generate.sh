#!/usr/bin/env bash

# Usage: ./generate.sh <EAR2> <EXAMPLE_RUN>


IMGNAME=$1
EXPNAME=$2
CHECKPOINT=$3

# 1) Run the multiview generation
python inference.py --base config/"${EXPNAME}.yaml" \
                    --resume "${CHECKPOINT}.ckpt"  \
                    --prompt "${IMGNAME}"

# # 2) Run sparse view reconstruction
cd instant-nsr-pl

# Remove any file extension from IMGNAME
IMGNAME="${IMGNAME%.*}"
python launch.py --config configs/example_run.yaml \
                 --gpu 0 \
                 --train dataset.root_dir=../mvoutput/"${EXPNAME}"/ \
                         dataset.scene="${IMGNAME}"

# 3) Run VDM Generation
cd ../
python extract_VDM.py --base_name "${IMGNAME}" \
                      --exp_name "${EXPNAME}"

###Example: bash generate.sh ear2 example_run example