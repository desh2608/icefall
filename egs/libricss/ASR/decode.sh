#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

stage=0
stop_stage=100
rttm_affix=""
gss_affix=""    # can be used to distinguish between different GSS outputs

. ./path.sh
. shared/parse_options.sh

# This script uses pretrained LibriSpeech model to decode LibriCSS GSS-enhanced audio,
# and then compute the cpWER.
test_sets="dev test"

# Append _ to affixes if not empty
rttm_affix=${rttm_affix:+_$rttm_affix}
gss_affix=${gss_affix:+_$gss_affix}

EXP_DIR=pruned_transducer_stateless2/exp/libricss${rttm_affix}${gss_affix}

mkdir -p $EXP_DIR
cp pruned_transducer_stateless2/exp/epoch-25.pt $EXP_DIR/epoch-25.pt

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "Decoding LibriCSS data"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Decoding..."
  utils/queue-ackgpu.pl --mem 4G --gpu 1 --config conf/gpu.conf $EXP_DIR/decode.log \
    python ./pruned_transducer_stateless2/decode.py \
      --manifest-dir data/manifests \
      --rttm-affix "$rttm_affix" \
      --gss-affix "$gss_affix" \
      --epoch 25 \
      --exp-dir $EXP_DIR \
      --max-duration 500 \
      --decoding-method fast_beam_search \
      --beam-size 4 \
      --max-contexts 4 \
      --max-states 8
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  decode_dir=${EXP_DIR}/fast_beam_search
  for split in dev test; do
    ref_file=data/manifests/libricss-sdm_supervisions_all.jsonl.gz
    for part in ihm sdm gss; do
      log "Stage 1: Computing cpWER for ${part}_${split} set"
      hyp_file="${decode_dir}/${split}_${part}-beam_4_max_contexts_4_max_states_8-hyps.jsonl.gz"
      python local/compute_cpwer.py ${ref_file} ${hyp_file} \
        --stats-file ${decode_dir}/${split}_${part}_cpwer_stats.txt
    done
  done
fi
