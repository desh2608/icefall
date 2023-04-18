#!/usr/bin/env bash

set -eou pipefail

# Decode RNNT model
queue-freegpu.pl --gpu 1 --mem 10G --config conf/gpu.conf pruned_transducer_stateless2/exp/v0/decode_modified.log \
  python pruned_transducer_stateless2/decode.py \
    --epoch 99 --avg 1 \
    --exp-dir pruned_transducer_stateless2/exp/v0 \
    --manifest-dir data/manifests \
    --max-duration 500 \
    --decoding-method modified_beam_search \
    --beam-size 4
