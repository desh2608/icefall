# Introduction

This recipe includes some different ASR models trained with TedLium3.

# Transducers

There are various folders containing the name `transducer` in this folder.
The following table lists the differences among them.

|                                  | Encoder   | Decoder            | Comment                     |
|----------------------------------|-----------|--------------------|-----------------------------|
| `transducer_stateless`           | Conformer | Embedding + Conv1d |                             |
| `pruned_transducer_stateless`    | Conformer | Embedding + Conv1d | Using k2 pruned RNN-T loss  |                      |

The decoder in `transducer_stateless` is modified from the paper
[Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.

## Long-form ASR and segmentation

The script `decode_full.sh` supports decoding long-form audio. The method is as follows:

* Split the full recording into chunks of 30 seconds each (with some left and right context).
* Decode each chunk separately.
* Combine the results from all chunks, based on time-stamps of the predicted BPE tokens.

We score these in 2 ways:

1. Concatenated WER (c-WER): Concatenate all ref and hyp texts, and score using kaldialign.
    This kind of scoring does not take into account segmentation errors.
2. sclite WER: We create reference STM files from the force-aligned CTM files (using different
    max-pause values), and score against the hyp CTM files using sclite. This kind of scoring
    also takes into account segmentation errors.
