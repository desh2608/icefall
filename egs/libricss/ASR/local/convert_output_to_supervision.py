#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Data preparation for LibriCSS dataset.
import sys
from pathlib import Path

from lhotse import SupervisionSet, SupervisionSegment

import logging


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create Lhotse SupervisionSet from decoded output."
    )
    parser.add_argument(
        "text_file", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    parser.add_argument("out_file", type=Path, help="Path to output file.")
    return parser.parse_args()


def main(text_file, out_file):
    segments = []
    for line in text_file:
        try:
            uttid, text = line.strip().split("\t", 1)
        except ValueError:
            # Empty text line.
            continue
        reco, spkid, start_end, _, _ = uttid.split("-")
        start = float(start_end.split("_")[0]) / 100
        end = float(start_end.split("_")[1]) / 100
        duration = end - start
        segments.append(
            SupervisionSegment(
                id=uttid,
                recording_id=reco,
                start=start,
                duration=duration,
                text=text,
                channel=0,
                speaker=spkid,
            )
        )
    supervision = SupervisionSet.from_segments(segments)
    supervision.to_file(out_file)


if __name__ == "__main__":
    args = get_args()
    main(args.text_file, args.out_file)
