# This script perform long-form decoding. It is based on https://github.com/k2-fsa/icefall/pull/980/.
#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

# This script is used to recogize long audios. It is similar to decode_full.sh, but instead of
# chunking and decoding each chunk, we chunk on-the-fly and only compute encoder outputs
# for each chunk separately. The decoding is performed on the merged encoder outputs.

# Each chunk (except the first and the last) is padded with extra left side and right side.
# The chunk length is: left_side + chunk_size + right_side.
# After computing the encoder outputs, the extra left side and right side are removed.
chunk=30
extra=4

stage=1
stop_stage=2
exp_dir=zipformer/exp_new/a0a
epoch=50
avg=22
use_averaged_model=True
decoding_method=greedy_search
decode_batch_size=32

decode_cmd="queue-freegpu.pl --config conf/gpu.conf --gpu 1 --mem 4G"
score_cmd="queue.pl --mem 12G"
. shared/parse_options.sh || exit 1


log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Perform speech recognition on full recordings"
  $decode_cmd $exp_dir/decode_full_${decoding_method}.log \
    python zipformer/full_decode.py \
      --epoch $epoch --avg $avg --use-averaged-model $use_averaged_model \
      --exp-dir $exp_dir \
      --reference-ctm-dir download/tedlium_ctm \
      --manifest-dir data/manifests \
      --max-duration 600 \
      --decoding-method ${decoding_method} \
      --chunk $chunk --extra $extra --decode-batch-size $decode_batch_size \
      --beam-size 4
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute WERs using sclite"
  for part in dev test; do
    scoring_dir=${exp_dir}/${decoding_method}-full/${part}_chunk${chunk}_extra${extra}_scoring
    mkdir -p $scoring_dir
    cat download/tedlium3/legacy/$part/stm/*.stm | awk '{$2 = "0"; print}' | python local/join_suffix_stm.py > $scoring_dir/ref.stm
    cat ${exp_dir}/${decoding_method}-full/${part}_full.ctm > $scoring_dir/hyp.ctm
    sclite -r $scoring_dir/ref.stm stm -h ${scoring_dir}/hyp.ctm ctm \
      -O $scoring_dir -o all
    grep "Sum/Avg" $scoring_dir/hyp.ctm.sys 
  done
fi
