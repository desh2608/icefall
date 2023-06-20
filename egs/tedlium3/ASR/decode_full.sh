# This script perform long-form decoding. It is based on https://github.com/k2-fsa/icefall/pull/980/.
#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export KALDI_ROOT=/home/hltcoe/draj/kaldi

set -eou pipefail

# This script is used to recogize long audios. The process is as follows:
# 1) Split long audios into chunks with overlaps.
# 2) Perform speech recognition on chunks, getting tokens and timestamps.
# 3) Merge the overlapped chunks into utterances acording to the timestamps.

# Each chunk (except the first and the last) is padded with extra left side and right side.
# The chunk length is: left_side + chunk_size + right_side.
chunk=30
extra=4

stage=1
stop_stage=4
exp_dir=zipformer/exp/
decoding_method=greedy_search

# scoring options
overlap_spk=2
hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

decode_cmd="queue-freegpu.pl --config conf/gpu.conf --gpu 1 --mem 4G"
score_cmd="queue.pl --mem 12G"
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
  $decode_cmd $exp_dir/decode_chunked_${decoding_method}.log \
    python zipformer/decode_chunked.py \
      --epoch 99 --avg 1 \
      --exp-dir $exp_dir \
      --manifest-dir data/manifests \
      --max-duration 600 \
      --decoding-method ${decoding_method} \
      --chunk $chunk --extra $extra \
      --beam-size 4
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Merge splitted chunks into utterances and score."
  python local/merge_chunks.py \
    --res-dir ${exp_dir}/${decoding_method}-chunked \
    --manifest-dir data/manifests \
    --bpe-model data/lang_bpe_500/bpe.model \
    --chunk $chunk --extra $extra
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute WERs"
  for part in dev test; do
    scoring_dir=${exp_dir}/${decoding_method}-chunked/${part}_chunk${chunk}_extra${extra}_scoring
    mkdir -p $scoring_dir
    cat $scoring_dir/hyp.ctm | python local/join_suffix.py > $scoring_dir/hyp_score.ctm
    for pause in 0.0 0.2 0.5; do
      log "split: $part pause: $pause"
      scoring_dir_pause=${scoring_dir}/pause${pause}
      mkdir -p $scoring_dir_pause
      cat download/tedlium_ctm/${part}.ctm | python local/ctm_to_stm.py --max-pause $pause > $scoring_dir_pause/ref.stm
      sclite -r $scoring_dir_pause/ref.stm stm -h $scoring_dir/hyp_score.ctm ctm \
        -O $scoring_dir_pause -o all
      grep "Sum/Avg" $scoring_dir_pause/hyp_score.ctm.sys 
    done
  done
fi
