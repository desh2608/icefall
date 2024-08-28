#!/usr/bin/env bash

set -eou pipefail

. parse_options.sh || exit 1

# Decode RNNT model
queue-freegpu.pl --gpu 1 --mem 16G --config conf/gpu_decode.conf dprnn_zipformer_hat/exp/h2_adapt1/decode_greedy_full.log \
  python dprnn_zipformer_hat/decode_full.py \
    --epoch 20 --avg 5 --use-averaged-model True \
    --exp-dir dprnn_zipformer_hat/exp/h2_adapt1 \
    --max-duration 250 \
    --decoding-method greedy_search \
    --chunk-size 32 \
    --left-context-frames 128 \
    --aux-chunk-size 32 \
    --aux-left-context-frames -1 \
    --use-aux-encoder True \
    --use-aux-joint-encoder-layer lstm \
    --max-speakers 4 \
    --aux-output-layer 0 \
    --aux-encoder-dim 192,256,256 \
    --use-speaker-prefixing False \
    --speaker-buffer-frames 128