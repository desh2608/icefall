#!/usr/bin/env bash

set -eou pipefail

# Decode RNNT model
queue-freegpu.pl --gpu 1 --mem 10G --config conf/gpu.conf zipformer_ctc/exp/v1/decode_greedy.log \
  python zipformer_ctc/decode.py \
    --epoch 99 --avg 1 \
    --exp-dir zipformer_ctc/exp/v1 \
    --manifest-dir data/manifests \
    --max-duration 500 \
    --decoding-method greedy_search \
    --beam-size 4
