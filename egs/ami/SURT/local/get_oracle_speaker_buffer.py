#!/usr/bin/env python3
# Copyright    2022  Johns Hopkins University        (authors: Desh Raj)
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
This script creates speaker prefix buffer for AMI dev and test sets.
"""
import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import torch
from lhotse import load_manifest_lazy


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "manifest_dir",
        type=Path,
        help="""Path to the manifest directory. This directory should contain
        the AMI dev and test IHM manifests.""",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="""Path to the output directory. This directory will contain
        the speaker buffer manifests.""",
    )

    parser.add_argument(
        "--speaker-buffer-frames",
        type=int,
        help="""The size of the speaker buffer in frames, for each speaker.
        """,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="""Random seed""",
        default=0,
    )

    return parser.parse_args()


def get_oracle_speaker_buffer(
    manifest_dir: Path,
    output_dir: Path,
    speaker_buffer_frames: int = 32,
    affix="",
):
    """
    This function creates speaker buffer for AMI sets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for data in ["ami"]:
        buffer = {}
        speaker_order = {}
        for split in ["test"]:
            mixed_cuts = load_manifest_lazy(
                manifest_dir / f"cuts_{data}-ihm-mix_{split}.jsonl.gz"
            )
            source_cuts = load_manifest_lazy(
                manifest_dir / f"{data}-ihm_cuts_{split}.jsonl.gz"
            )
            source_cuts = source_cuts.trim_to_alignments(
                type="word",
                max_pause=0.0,
                max_segment_duration=1.1 * (speaker_buffer_frames / 100),
            )
            # Only keep cuts which have more frames than the speaker buffer size, and
            # less than twice the speaker buffer size.
            source_cuts = source_cuts.filter(
                lambda c: c.num_frames > speaker_buffer_frames
            )
            # Group cuts by speaker and session
            cuts_by_speaker = defaultdict(lambda: defaultdict(list))
            for cut in source_cuts:
                session_id = cut.recording_id
                speaker = cut.supervisions[0].speaker
                cuts_by_speaker[session_id][speaker].append(cut)

            for cut in mixed_cuts:
                speaker_cuts = cuts_by_speaker[cut.recording_id]
                selected_cuts = []
                for spk in sorted(speaker_cuts):
                    selected_cut = random.choice(speaker_cuts[spk])
                    selected_cuts.append(selected_cut)
                spk_feats = []
                for cut in selected_cuts:
                    # Load the features for this cut
                    feats = torch.from_numpy(cut.load_features())
                    _, F = feats.shape
                    # Select a random start frame
                    start_frame = random.randint(
                        0, cut.num_frames - speaker_buffer_frames - 1
                    )
                    feat = feats[start_frame : start_frame + speaker_buffer_frames]
                    # Add the features to the buffer
                    spk_feats.append(feat)
                # Concatenate the features for this session
                spk_feats = torch.cat(spk_feats, dim=0)
                # Add the features to the buffer
                buffer[cut.recording_id] = spk_feats
                speaker_order[cut.recording_id] = [
                    cut.supervisions[0].speaker for cut in selected_cuts
                ]

        torch.save(
            buffer,
            output_dir / f"{data}-ihm_buffer_{speaker_buffer_frames}{affix}.pt",
        )
        with open(
            output_dir / f"{data}-ihm_buffer_order_{speaker_buffer_frames}{affix}.json",
            "w",
        ) as f:
            json.dump(speaker_order, f)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    random.seed(args.seed)

    get_oracle_speaker_buffer(
        args.manifest_dir,
        args.output_dir,
        args.speaker_buffer_frames,
        affix=f"_seed{args.seed}",
    )
