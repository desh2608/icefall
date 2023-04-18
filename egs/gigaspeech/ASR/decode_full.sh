# This script perform long-form decoding. It is based on https://github.com/k2-fsa/icefall/pull/980/.
#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

# This script is used to recogize long audios. The process is as follows:
# 1) Split long audios into chunks with overlaps.
# 2) Perform speech recognition on chunks, getting tokens and timestamps.
# 3) Merge the overlapped chunks into utterances acording to the timestamps.

# Each chunk (except the first and the last) is padded with extra left side and right side.
# The chunk length is: left_side + chunk_size + right_side.
chunk=30.0
extra=2.0

stage=1
stop_stage=4
decoding_method=modified_beam_search

cmd="queue-freegpu.pl --config conf/gpu.conf --gpu 1 --mem 4G"
. shared/parse_options.sh || exit 1


log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  # Chunk manifests are saved to data/manifests/cuts_{subset}_chunks.jsonl.gz
  log "Stage 1: Split long audio into chunks"
  python local/split_into_chunks.py \
    --manifest-dir data/manifests \
    --chunk $chunk \
    --extra $extra  # Extra duration (in seconds) at both sides
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Perform speech recognition on splitted chunks"
  queue-freegpu.pl --gpu 1 --mem 10G --config conf/gpu.conf pruned_transducer_stateless2/exp/v0/decode_chunked_beam.log \
    python pruned_transducer_stateless2/decode_chunked.py \
      --epoch 99 --avg 1 \
      --exp-dir pruned_transducer_stateless2/exp/v0 \
      --manifest-dir data/manifests \
      --max-duration 2400 \
      --decoding-method ${decoding_method} \
      --beam-size 4
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Merge splitted chunks into utterances and score."
  python local/merge_chunks.py \
    --res-dir pruned_transducer_stateless2/exp/v0/${decoding_method}-chunked \
    --manifest-dir data/manifests \
    --bpe-model data/lang_bpe_500/bpe.model \
    --extra $extra
fi