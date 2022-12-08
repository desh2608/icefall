#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path

import numpy as np
from cytoolz.itertoolz import groupby
from kaldialign import edit_distance
from lhotse import SupervisionSet, load_manifest
from scipy.optimize import linear_sum_assignment

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Compute cpWER.")
    parser.add_argument(
        "ref",
        type=Path,
        help="Path to reference supervision segments.",
    )
    parser.add_argument(
        "hyp",
        type=Path,
        help="Path to decoded supervision segments.",
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        required=True,
        help="Path to output stats file.",
    )
    return parser.parse_args()


def concat_text(segments):
    """
    This function takes as input a SupervisionSet and returns a list of strings, where
    each string is a concatenation of utterances of one particular speaker, appended
    in order of start time.
    """
    speakers = set(s.speaker for s in segments)
    text = []
    for speaker in speakers:
        speaker_segs = list(filter(lambda s: s.speaker == speaker, segments))
        text.append(
            " ".join(
                s.text.strip() if s.text is not None else ""
                for s in sorted(speaker_segs, key=lambda s: s.start)
            )
        )
    return text


def compute_cpWER(ref_text, hyp_text):
    """
    ref_text and hyp_text are lists of strings.
    """
    M = len(ref_text)
    N = len(hyp_text)
    costs = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            cur_ref = ref_text[i].split()
            cur_hyp = hyp_text[j].split()
            result = edit_distance(cur_ref, cur_hyp)
            wer = result["total"] / len(cur_ref)
            costs[i, j] = wer
    row_ind, col_ind = linear_sum_assignment(costs)
    ref_text = [ref_text[i] for i in row_ind]
    hyp_text = [hyp_text[i] for i in col_ind]
    count = num_ins = num_del = num_sub = total = 0
    for ref, hyp in zip(ref_text, hyp_text):
        ref = ref.strip().split()
        hyp = hyp.strip().split()
        count += len(ref)
        result = edit_distance(ref, hyp)
        num_ins += result["ins"]
        num_del += result["del"]
        num_sub += result["sub"]
        total += result["total"]
    return {
        "ref_text": ref_text,
        "hyp_text": hyp_text,
        "count": count,
        "num_ins": num_ins,
        "num_del": num_del,
        "num_sub": num_sub,
        "ins": num_ins / count,
        "del": num_del / count,
        "sub": num_sub / count,
        "cpwer": total / count,
    }


def main(ref, hyp, stats_file):
    ref_segs = load_manifest(ref)
    hyp_segs = load_manifest(hyp)

    hyp_segs_by_recording = groupby(
        lambda s: s.recording_id, sorted(hyp_segs, key=lambda s: s.recording_id)
    )

    wer_dict = {}
    # Each cut in the reference corresponds to a recording, so we iterate over all the
    # cuts (i.e. recordings)
    for reco_id, reco_hyps in hyp_segs_by_recording.items():

        ref_text = concat_text(ref_segs.filter(lambda s: s.recording_id == reco_id))

        if len(ref_text) == 0:
            logging.warning(f"Empty reference for {reco_id}")

        # Get hypothesis text
        hyp_text = concat_text(reco_hyps)

        # Compute cpWER
        stats = compute_cpWER(ref_text, hyp_text)

        # Store results
        wer_dict[reco_id] = stats

    # Compute average cpWER
    total_num_words = sum(wer_dict[reco_id]["count"] for reco_id in wer_dict)
    total_ins = sum(wer_dict[reco_id]["num_ins"] for reco_id in wer_dict)
    total_del = sum(wer_dict[reco_id]["num_del"] for reco_id in wer_dict)
    total_sub = sum(wer_dict[reco_id]["num_sub"] for reco_id in wer_dict)
    avg_ins = total_ins / total_num_words
    avg_del = total_del / total_num_words
    avg_sub = total_sub / total_num_words
    avg_cpwer = (total_ins + total_del + total_sub) / total_num_words

    wer_dict["TOTAL"] = {
        "ref_text": [],
        "hyp_text": [],
        "count": total_num_words,
        "num_ins": total_ins,
        "num_del": total_del,
        "num_sub": total_sub,
        "ins": avg_ins,
        "del": avg_del,
        "sub": avg_sub,
        "cpwer": avg_cpwer,
    }

    # Write results to file
    with stats_file.open("w") as f:
        json.dump(wer_dict, f, indent=2)

    # Print averages
    print(f"Average insertion rate: {avg_ins:.2%}")
    print(f"Average deletion rate: {avg_del:.2%}")
    print(f"Average substitution rate: {avg_sub:.2%}")
    print(f"Average cpWER: {avg_cpwer:.2%}")


if __name__ == "__main__":
    args = get_args()
    main(args.ref, args.hyp, args.stats_file)
