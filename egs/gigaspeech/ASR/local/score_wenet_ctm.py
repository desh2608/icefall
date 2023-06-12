#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang, Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This script will score the decoding results of Wenet model provided by Jennifer.
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import sentencepiece as spm
from icefall.utils import store_transcripts, write_error_stats
from lhotse import CutSet, load_manifest
from lhotse.supervision import AlignmentItem
from lhotse.utils import fastcopy
from lhotse import SupervisionSegment

from gigaspeech_scoring import asr_text_post_processing


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scoring-dir",
        type=Path,
        default=None,
        help="Path to directory to write scoring output.",
    )

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Path to directory with full cuts.",
    )

    parser.add_argument(
        "--hyp-ctm",
        type=Path,
        default=None,
        help="Path to the decoding results of Wenet model.",
    )

    return parser.parse_args()


def words_post_processing(text: List[AlignmentItem]) -> List[AlignmentItem]:
    res = []
    for item in text:
        text = item.symbol
        text = asr_text_post_processing(text)
        if len(text) == 0:
            continue
        else:
            res.append(
                AlignmentItem(
                    start=item.start,
                    duration=item.duration,
                    symbol=text,
                )
            )
    return res


def read_in_ctm(ctm_file: Path, cuts_orig: CutSet) -> CutSet:
    """
    Create a Lhotse CutSet from an input CTM file.
    """
    # Read all lines from CTM file and store them in a dict indexed by recording ids
    words = defaultdict(list)
    with open(ctm_file, "r") as f:
        for line in f:
            reco_id, channel, start, duration, word, *_ = line.strip().split(" ")
            reco_id = reco_id.split("_")[0]  # POD1000000005_chunk0_18000
            start = float(start)
            duration = float(duration)
            words[reco_id].append(
                AlignmentItem(
                    start=start,
                    duration=duration,
                    symbol=word,
                )
            )

    # Create a new CutSet with the new supervisions
    cuts = []
    for cut in cuts_orig:
        reco_id = cut.recording.id
        # Get reference text and normalize it
        old_text = " ".join(s.text for s in cut.supervisions)
        old_text = asr_text_post_processing(old_text)
        # Get hyp text and normalize it
        hyp_alis = words[reco_id]
        hyp_alis = words_post_processing(hyp_alis)
        hyp_text = " ".join(ali.symbol for ali in hyp_alis)

        new_sup = SupervisionSegment(
            id=reco_id,
            recording_id=reco_id,
            start=0,
            duration=cut.duration,
            text=hyp_text,
            alignment={"word": hyp_alis},
            language=cut.supervisions[0].language,
            speaker=cut.supervisions[0].speaker,
            custom={"orig_text": old_text},
        )
        cuts.append(fastcopy(cut, supervisions=[new_sup]))

    return CutSet.from_cuts(cuts)


def save_results(
    res_dir: Path,
    test_set_name: str,
    results: List[Tuple[str, List[str], List[str]]],
):
    """Save the results to files.
    Args:
        res_dir:
            The directory to save the results.
        test_set_name:
            The name of the test set (e.g. greedy-DEV).
        results:
            A list of tuples (recording_id, hyp, ref).
    """
    recog_path = res_dir / f"recogs-{test_set_name}.txt"
    store_transcripts(filename=recog_path, texts=results)
    logging.info(f"The transcripts are stored in {recog_path}")

    # The following prints out WERs, per-word error statistics and aligned
    # ref/hyp pairs.
    errs_filename = res_dir / f"errs-{test_set_name}.txt"
    with open(errs_filename, "w") as f:
        wer = write_error_stats(f, f"{test_set_name}", results, enable_log=True)

    logging.info("Wrote detailed error stats to {}".format(errs_filename))

    errs_info = res_dir / f"wer-summary-{test_set_name}.txt"
    with open(errs_info, "w") as f:
        print("WER: {:.2f}".format(wer * 100), file=f)


def main():
    args = get_parser()

    scoring_dir = args.scoring_dir
    scoring_dir.mkdir(parents=True, exist_ok=True)

    orig_cuts = load_manifest(args.manifest_dir / f"cuts_TEST_full.jsonl.gz")
    stm_file = scoring_dir / f"ref.stm"
    ctm_file = scoring_dir / f"hyp.ctm"

    recog_cuts = read_in_ctm(args.hyp_ctm, orig_cuts)
    recog_cuts.to_file(scoring_dir / f"recog_cuts.jsonl.gz")
    logging.info(f"Cuts saved to {scoring_dir / f'recog_cuts.jsonl.gz'}")

    # Write CTM file
    out_sups = recog_cuts.decompose()[1]
    out_sups.write_alignment_to_ctm(ctm_file, type="word")

    # Write STM file. Gigaspeech does not have speaker labels, so we assign different
    # speaker labels to each supervision.
    with open(stm_file, "w") as f:
        for cut in orig_cuts:
            for idx, sup in enumerate(cut.supervisions):
                text = asr_text_post_processing(sup.text)
                print(
                    f"{sup.recording_id} {sup.channel} {sup.speaker} {sup.start:.2f} {sup.end:.2f} {text}",
                    file=f,
                )

    results = []
    for cut in recog_cuts:
        ref = cut.supervisions[0].custom["orig_text"]
        hyp = cut.supervisions[0].text
        # convert ref and hyp to list of words
        ref = ref.split()
        hyp = hyp.split()
        results.append((cut.id, ref, hyp))
    save_results(args.scoring_dir, "DEV", results)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
