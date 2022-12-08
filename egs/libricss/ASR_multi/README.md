# LibriCSS

This is a speaker-attributed ASR recipe for LibriCSS. Note that this
is an inference-only recipe, i.e., we assume that you have a pretrained
ASR model available. We use the pruned_transducer_stateless2 recipe from
LibriSpeech for the ASR model.

This recipe measures performance in terms of concatenated minimum-permutation
word error rate (cpWER). Please check the CHiME-6 challenge baseline paper
for details about the metric. You can use oracle segments or segments from a
diarization output for performing the inference. Please check the `prepare.sh` script
for details about how to use diarization output.

The recipe additionally provides the option to perform GSS-based front-end
enhancement of the segments, or to use only the single distant microphone
(SDM). Results for both settings are given below.

## Performance Record

| **Diarizer**   | **SDM Test** | **GSS Test** |
|----------------|:------------:|:------------:|
| Spectral       |     18.30    |     15.86    |
| Spectral + OVL |     17.09    |     12.12    |
