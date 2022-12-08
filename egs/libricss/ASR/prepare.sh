#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -euo pipefail

stage=-1
stop_stage=100
nj_gss=8
rttm_affix=""
gss_affix=""    # can be used to distinguish between different GSS outputs

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/LibriCSS
#      Download using `lhotse download libricss`
#
# We use pre-trained LibriSpeech models to decode the LibriCSS audio. Please
# download from the following link:
# https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless2_20220625/tree/main
# - data/lang_bpe_500
# - pruned_transducer_stateless2/exp/epoch-25.pt (pretrained-epoch-24-avg-10.pt)
#
# This script requires the `gss` package to be installed (https://github.com/desh2608/gss)
#
# The different stages in this script can be used to prepare LibriCSS data in several formats:
# 1. IHM (equivalent to LibriSpeech test-clean)
# 2. SDM (single distant microphone)
# 3. GSS-enhanced MDM (requires GPU)
#
# For SDM and GSS, an RTTM file can be additionally provided to use as segments instead
# of oracle segments. In this case, the RTTM directory should be placed in $dl_dir and the
# name should be provided as --rttm-affix.

dl_dir=$PWD/download
cmd="queue-ackgpu.pl --gpu 1 --mem 4G --config conf/gpu.conf"

. ./path.sh
. shared/parse_options.sh || exit 1

mkdir -p data

if [ -z $rttm_affix ]; then
    sdm_supervisions_file=libricss-sdm_supervisions_all.jsonl.gz
    mdm_supervisions_file=libricss-mdm_supervisions_all.jsonl.gz
else
    rttm_dir=$dl_dir/$rttm_affix
    sdm_supervisions_file=libricss-sdm_supervisions_all_${rttm_affix}.jsonl.gz
    mdm_supervisions_file=libricss-mdm_supervisions_all_${rttm_affix}.jsonl.gz
fi

# Append _ to affixes if not empty
rttm_affix=${rttm_affix:+_$rttm_affix}
gss_affix=${gss_affix:+_$gss_affix}

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/LibriCSS,
  # you can create a symlink
  #
  #   ln -sfv /path/to/LibriCSS $dl_dir/LibriCSS
  #
  if [ ! -d $dl_dir/LibriSpeech/train-other-500 ]; then
    lhotse download libricss $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare LibriCSS manifest"
  # We assume that you have downloaded the LibriCSS corpus
  # to $dl_dir/LibriCSS
  for mic in ihm sdm mdm; do
    lhotse prepare libricss --type $mic $dl_dir/LibriCSS data/manifests/
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ] && [ ! -z $rttm_affix ]; then
    echo "Stage 2: Create supervisions from RTTM file"
    gss utils rttm-to-supervisions --channels 1 $rttm_dir data/manifests/$sdm_supervisions_file
    gss utils rttm-to-supervisions --channels 7 $rttm_dir data/manifests/$mdm_supervisions_file
fi

# Stage 3 to 6 are used to create the GSS-enhanced MDM data
gss_exp_dir=exp/gss${rttm_affix}
mkdir -p $gss_exp_dir
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Prepare cut set"
    # --force-eager must be set if recordings are not sorted by id
    lhotse cut simple --force-eager \
      -r data/manifests/libricss-mdm_recordings_all.jsonl.gz \
      -s data/manifests/$mdm_supervisions_file \
      $gss_exp_dir/cuts.jsonl.gz
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Trim cuts to supervisions (1 cut per supervision segment)"
    lhotse cut trim-to-supervisions --discard-overlapping \
        $gss_exp_dir/cuts.jsonl.gz $gss_exp_dir/cuts_per_segment.jsonl.gz
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Stage 5: Split segments into $nj_gss parts"
    gss utils split $nj_gss $gss_exp_dir/cuts_per_segment.jsonl.gz $gss_exp_dir/split${nj_gss}
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "Stage 6: Enhance segments using GSS"
    mkdir -p $gss_exp_dir/cuts${gss_affix}

    $cmd JOB=1:$nj_gss $gss_exp_dir/log/enhance.JOB.log \
        gss enhance cuts \
          $gss_exp_dir/cuts.jsonl.gz $gss_exp_dir/split$nj_gss/cuts_per_segment.JOB.jsonl.gz \
          $gss_exp_dir/enhanced${gss_affix} \
          --channels 0,1,2,3,4,5,6 \
          --bss-iterations 10 \
          --context-duration 15.0 \
          --min-segment-length 0.1 \
          --max-segment-length 15.0 \
          --max-batch-duration 20.0 \
          --num-buckets 3 \
          --enhanced-manifest $gss_exp_dir/cuts${gss_affix}/cuts.JOB.jsonl.gz \
          --force-overwrite

    # Merge the enhanced cuts
    lhotse combine $gss_exp_dir/cuts${gss_affix}/cuts.*.jsonl.gz data/manifests/libricss-gss_cuts_all${rttm_affix}${gss_affix}.jsonl.gz
fi

# Stages 7 to 9 are used to extract features for the different data types
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    log "Stage 7: Compute fbank features for LibriCSS IHM data"
    python local/compute_fbank_libricss.py --mic ihm --data-dir data/manifests \
      --output-dir data/fbank
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    log "Stage 8: Compute fbank features for LibriCSS SDM data"
    python local/compute_fbank_libricss.py --mic sdm --data-dir data/manifests \
      --output-dir data/fbank --rttm-affix "$rttm_affix"
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    log "Stage 9: Compute fbank features for LibriCSS MDM data enhanced with GSS"
    $cmd exp/feats.log python local/compute_fbank_libricss.py --mic gss --data-dir data/manifests \
      --output-dir data/fbank --rttm-affix "$rttm_affix" --gss-affix "$gss_affix"
fi
