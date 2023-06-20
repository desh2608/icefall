#!/usr/bin/env python3
# Copyright    2023  Johns Hopkins University        (authors: Desh Raj)
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
This script converts a CTM file into an STM file where segments are defined based on
the max pause argument.

CTM file is read from stdin and STM file is written to stdout.
"""

import argparse
import logging
import sys
from collections import defaultdict


EPSILON = 1e-6


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-pause",
        type=float,
        default=None,
        help="""Maximum pause between words to put in same segment.""",
    )

    return parser.parse_args()


def main():
    args = get_args()
    logging.info(vars(args))

    words = defaultdict(list)
    for line in sys.stdin:
        reco, _, start, dur, word, *_ = line.strip().split()
        start = float(start)
        dur = float(dur)
        words[reco].append((start, dur, word))

    # combine words into segments
    for reco, word_list in words.items():
        cur_start = word_list[0][0]
        cur_end = cur_start + word_list[0][1]
        cur_text = word_list[0][2]
        for start, dur, word in word_list[1:]:
            if args.max_pause is None or start - cur_end <= args.max_pause + EPSILON:
                cur_end = start + dur
                cur_text += " " + word
            else:
                print(f"{reco} 1 {reco} {cur_start:.2f} {cur_end:.2f} {cur_text}")
                cur_start = start
                cur_end = start + dur
                cur_text = word

        # write the last segment
        print(f"{reco} 1 {reco} {cur_start:.2f} {cur_end:.2f} {cur_text}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
