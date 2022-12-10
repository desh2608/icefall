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
#  - $dl_dir/alimeeting
#      Download using `lhotse download ali-meeting`
#
# We use pre-trained AliMeeting models to decode the audio. Please
# download from the following link:
# https://huggingface.co/desh2608/icefall-asr-alimeeting-pruned-transducer-stateless7
# - data/lang_char
# - pruned_transducer_stateless7/exp/pretrained.pt
#
# This script requires the `gss` package to be installed (https://github.com/desh2608/gss)
#
# The different stages in this script can be used to prepare AMI data in several formats:
# 1. IHM (individual headset microphone)
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
  eval_sdm_supervisions_file=alimeeting-sdm_supervisions_eval.jsonl.gz
  test_sdm_supervisions_file=alimeeting-sdm_supervisions_test.jsonl.gz
  eval_mdm_supervisions_file=alimeeting-mdm_supervisions_eval.jsonl.gz
  test_mdm_supervisions_file=alimeeting-mdm_supervisions_test.jsonl.gz
else
  rttm_dir=$dl_dir/$rttm_affix
  eval_sdm_supervisions_file=alimeeting-sdm_supervisions_eval_${rttm_affix}.jsonl.gz
  test_sdm_supervisions_file=alimeeting-sdm_supervisions_test_${rttm_affix}.jsonl.gz
  eval_mdm_supervisions_file=alimeeting-mdm_supervisions_eval_${rttm_affix}.jsonl.gz
  test_mdm_supervisions_file=alimeeting-mdm_supervisions_test_${rttm_affix}.jsonl.gz
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

  # If you have pre-downloaded it to /path/to/AliMeeting,
  # you can create a symlink
  #
  #   ln -sfv /path/to/AliMeeting $dl_dir/alimeeting
  #
  if [ ! -d $dl_dir/alimeeting ]; then
    lhotse download ali-meeting $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare AliMeeting manifest"
  # We assume that you have downloaded the AliMeeting corpus
  # to $dl_dir/alimeeting
  for mic in ihm sdm mdm; do
    lhotse prepare ali-meeting --mic $mic --save-mono --normalize-text m2met \
      $dl_dir/alimeeting data/manifests/
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ] && [ ! -z $rttm_affix ]; then
  echo "Stage 2: Create supervisions from RTTM file"
  gss utils rttm-to-supervisions --channels 1 $rttm_dir/eval data/manifests/$eval_sdm_supervisions_file
  gss utils rttm-to-supervisions --channels 1 $rttm_dir/test data/manifests/$test_sdm_supervisions_file
  gss utils rttm-to-supervisions --channels 8 $rttm_dir/eval data/manifests/$eval_mdm_supervisions_file
  gss utils rttm-to-supervisions --channels 8 $rttm_dir/test data/manifests/$test_mdm_supervisions_file
fi

# Stage 3 to 6 are used to create the GSS-enhanced MDM data
gss_exp_dir=exp/gss${rttm_affix}
mkdir -p $gss_exp_dir/{eval,test}
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: Prepare cut set"
  # --force-eager must be set if recordings are not sorted by id
  lhotse cut simple --force-eager \
    -r data/manifests/alimeeting-mdm_recordings_eval.jsonl.gz \
    -s data/manifests/$eval_mdm_supervisions_file \
    $gss_exp_dir/eval/cuts.jsonl.gz

  lhotse cut simple --force-eager \
    -r data/manifests/alimeeting-mdm_recordings_test.jsonl.gz \
    -s data/manifests/$test_mdm_supervisions_file \
    $gss_exp_dir/test/cuts.jsonl.gz
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Stage 4: Trim cuts to supervisions (1 cut per supervision segment)"
  for part in eval test; do
    lhotse cut trim-to-supervisions --discard-overlapping \
        $gss_exp_dir/$part/cuts.jsonl.gz $gss_exp_dir/$part/cuts_per_segment.jsonl.gz
  done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo "Stage 5: Split segments into $nj_gss parts"
  for part in eval test; do
    gss utils split $nj_gss $gss_exp_dir/$part/cuts_per_segment.jsonl.gz $gss_exp_dir/$part/split${nj_gss}
  done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  echo "Stage 6: Enhance segments using GSS"
  for part in eval test; do
    mkdir -p $gss_exp_dir/$part/cuts${gss_affix}

    $cmd JOB=1:$nj_gss $gss_exp_dir/$part/log/enhance.JOB.log \
        gss enhance cuts \
          $gss_exp_dir/$part/cuts.jsonl.gz $gss_exp_dir/$part/split$nj_gss/cuts_per_segment.JOB.jsonl.gz \
          $gss_exp_dir/$part/enhanced${gss_affix} \
          --channels 0,1,2,3,4,5,6,7 \
          --use-garbage-class \
          --bss-iterations 10 \
          --context-duration 15.0 \
          --min-segment-length 0.05 \
          --max-segment-length 15.0 \
          --max-batch-duration 20.0 \
          --max-batch-cuts 3 \
          --num-buckets 3 \
          --enhanced-manifest $gss_exp_dir/$part/cuts${gss_affix}/cuts.JOB.jsonl.gz \
          --force-overwrite || exit 1

    # Merge the enhanced cuts
    lhotse combine $gss_exp_dir/$part/cuts${gss_affix}/cuts.*.jsonl.gz data/manifests/alimeeting-gss_cuts_${part}${rttm_affix}${gss_affix}.jsonl.gz
  done
fi

# Stages 7 to 9 are used to extract features for the different data types
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Compute fbank features for AMI IHM data"
  python local/compute_fbank_alimeeting.py --mic ihm --data-dir data/manifests \
    --output-dir data/fbank --dataset-parts eval test
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Compute fbank features for AMI SDM data"
  $cmd exp/feats${rttm_affix}.log python local/compute_fbank_alimeeting.py --mic sdm --data-dir data/manifests \
    --output-dir data/fbank --rttm-affix "$rttm_affix" --dataset-parts eval test
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Compute fbank features for AMI MDM data enhanced with GSS"
  $cmd exp/feats${rttm_affix}${gss_affix}.log python local/compute_fbank_alimeeting.py --mic gss --data-dir data/manifests \
    --output-dir data/fbank --rttm-affix "$rttm_affix" --gss-affix "$gss_affix" \
    --dataset-parts eval test
fi
