<div align="center">
<img src="https://raw.githubusercontent.com/k2-fsa/icefall/master/docs/source/_static/logo.png" width=168>
</div>

## Introduction to icefall

icefall contains ASR recipes for various datasets
using <https://github.com/k2-fsa/k2>.

You can use <https://github.com/k2-fsa/sherpa> to deploy models
trained with icefall.

You can try pre-trained models from within your browser without the need
to download or install anything by visiting <https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>
See <https://k2-fsa.github.io/icefall/huggingface/spaces.html> for more details.

## Speaker-attributed multi-talker ASR

This branch contains scripts to perform speaker-attributed ASR inference in the style
of the CHiME-6 Track 2, i.e., diarization -> enhancement -> ASR. The outputs are
evaluated using cpWER. We provide recipes for the following corpora:

* LibriCSS: `egs/libricss/ASR_multi`
* AMI: `egs/ami/ASR_multi`
* AliMeeting: `egs/alimeeting/ASR_multi`

### Usage

We assume that you have run diarization on the data before. End-to-end reproducible
recipes for clustering-based diarization are available on my [diarizer](https://github.com/desh2608/diarizer) repo. Once you have the RTTM files, you can run the scripts mentioned
above to perform speaker-attributed ASR. This is done in 3 steps:

1. GSS-based front-end enhancement using my [gss](https://github.com/desh2608/gss) package
2. ASR decoding using icefall pretrained models
3. cpWER computation for eval

The recipes contain a `prepare.sh` script (for step 1) and a `decode.sh` script (for steps
2 and 3). Please refer to these scripts for more details.
