#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

stage=0
rttm_tag="oracle"

. ./path.sh
. shared/parse_options.sh

# This script uses pretrained LibriSpeech model to decode LibriCSS GSS-enhanced audio,
# and then compute the cpWER.
test_sets="dev test"

mkdir -p pruned_transducer_stateless2/exp/libricss_${rttm_tag}
cp pruned_transducer_stateless2/exp/libricss_oracle/epoch-25.pt pruned_transducer_stateless2/exp/libricss_${rttm_tag}/epoch-25.pt

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "Decoding LibriCSS ${rttm_tag} enhanced data"

if [ $stage -le 0 ]; then
  log "Stage 0: Decoding..."
  utils/queue-ackgpu.pl --mem 4G --gpu 1 --config conf/gpu.conf pruned_transducer_stateless2/exp/libricss_${rttm_tag}/decode.log \
    python ./pruned_transducer_stateless2/decode.py \
      --manifest-dir data/manifests/libricss_${rttm_tag} \
      --epoch 25 \
      --exp-dir ./pruned_transducer_stateless2/exp/libricss_${rttm_tag} \
      --max-duration 500 \
      --decoding-method fast_beam_search \
      --beam-size 4 \
      --max-contexts 4 \
      --max-states 8
fi

if [ $stage -le 1 ]; then
  exp_dir="./pruned_transducer_stateless2/exp/libricss_${rttm_tag}/fast_beam_search/"
  for part in dev test; do
    log "Stage 1: Computing cpWER for ${part} set"
    recog_path="${exp_dir}/recogs-${part}-beam_4_max_contexts_4_max_states_8-epoch-25-beam-4-max-contexts-4-max-states-8.txt"
    cat $recog_path | python local/convert_output_to_supervision.py - ${exp_dir}/${part}_hyp.jsonl.gz 
    python local/compute_cpwer.py --ref data/manifests/libricss_${rttm_tag}/raw_cuts_${part}_orig.jsonl \
      --hyp ${exp_dir}/${part}_hyp.jsonl.gz --stats-file ${exp_dir}/${part}_cpwer_stats.txt
  done
fi
