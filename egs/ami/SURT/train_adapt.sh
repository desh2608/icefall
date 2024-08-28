#!/usr/bin/env bash

set -eou pipefail

. parse_options.sh || exit 1

queue-freegpu.pl --gpu 4 --mem 16G --config conf/gpu_32gb.conf dprnn_zipformer_hat/exp/h8_adapt2/train2.log \
  python dprnn_zipformer_hat/train2_adapt.py \
    --master-port 14678 \
    --use-fp16 True \
    --exp-dir dprnn_zipformer_hat/exp/h8_adapt2 \
    --world-size 4 \
    --max-duration 500 \
    --max-duration-valid 200 \
    --max-cuts 200 \
    --num-buckets 50 \
    --num-epochs 40 \
    --start-epoch 21 \
    --enable-spec-aug True \
    --enable-musan False \
    --ctc-loss-scale 0.0 \
    --base-lr 0.00005 \
    --lr-epochs 2 \
    --chunk-width-randomization True \
    --model-init-ckpt dprnn_zipformer_hat/exp/h9/pretrained.pt \
    --use-aux-encoder True \
    --freeze-main-model True \
    --aux-left-context-frames -1 \
    --use-aux-joint-encoder-layer lstm \
    --max-speakers 4 \
    --aux-output-layer 0 \
    --aux-encoder-dim 192,256,256 \
    --use-speaker-prefixing True \
    --speaker-buffer-frames 128 \
    --fixed-prefix-speakers True