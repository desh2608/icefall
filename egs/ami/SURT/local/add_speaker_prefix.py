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
This file adds speaker prefixes as temporal arrays to the mixture manifests.
It looks for manifests in the directory data/manifests.
"""
import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from lhotse import CutSet, LilcomChunkyWriter, load_manifest, load_manifest_lazy
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--speaker-buffer-frames",
        type=int,
        help="""The size of the speaker buffer in frames, for each speaker.
        """,
        default=128,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        help="""Random seed for reproducibility.""",
        default=0,
    )

    return parser.parse_args()


def add_speaker_prefix(speaker_buffer_frames):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    def _process(
        mixed_cuts: CutSet, source_cuts: CutSet, out_cuts: Path, out_feats: Path
    ):
        source_cuts = source_cuts.filter(
            lambda c: c.num_frames >= speaker_buffer_frames
        )
        # Group source cuts by speaker
        source_cuts_by_speaker = defaultdict(list)
        for cut in source_cuts:
            source_cuts_by_speaker[cut.supervisions[0].speaker].append(cut)
        # we will keep the source feats in memory to avoid repeated disk access
        source_feats = {}

        with tqdm() as pbar, CutSet.open_writer(
            out_cuts
        ) as cut_writer, LilcomChunkyWriter(out_feats) as prefix_feat_writer:
            for cut in mixed_cuts:
                # Get all speakers in the cut
                speakers = set([sup.speaker for sup in cut.supervisions])
                # Get a random prefix for each speaker
                prefix_feats = {}
                for speaker in speakers:
                    if (
                        speaker not in source_feats
                        or len(source_feats[speaker]) < speaker_buffer_frames
                    ):
                        source_cut = random.choice(source_cuts_by_speaker[speaker])
                        source_feats[speaker] = source_cut.load_features()
                    spk_feats = source_feats[speaker]
                    prefix_feats[speaker] = spk_feats[:speaker_buffer_frames]
                    # Update the source feats
                    spk_feats = spk_feats[speaker_buffer_frames:]
                    source_feats[speaker] = spk_feats
                cut.speakers = list(prefix_feats.keys())  # speakers in order
                cut.speaker_prefix = prefix_feat_writer.store_array(
                    cut.id, np.concatenate(list(prefix_feats.values()), axis=0)
                )
                cut_writer.write(cut)
                pbar.update(1)

    # Train sessions
    # mixed_name = "train_comb_v1_sources"
    # mixed_cuts = load_manifest_lazy(src_dir / f"cuts_{mixed_name}.jsonl.gz")
    # source_cuts = []
    # for data in ["ami", "icsi"]:
    #     for mic in ["ihm", "sdm"]:
    #         cuts = load_manifest_lazy(src_dir / f"{data}-{mic}_cuts_train.jsonl.gz")
    #         cuts = cuts.modify_ids(lambda x: f"{x}_{mic}")
    #         source_cuts.extend(cuts)
    # source_cuts = CutSet.from_cuts(source_cuts)
    # logging.info("Adding speaker prefix to train sessions")
    # _process(
    #     mixed_cuts,
    #     source_cuts,
    #     src_dir / f"cuts_{mixed_name}_speaker.jsonl.gz",
    #     output_dir / f"feats_{mixed_name}_speaker",
    # )

    # Dev sessions
    # mixed_name = "ami-ihm-mix_dev_groups"
    # mixed_cuts = load_manifest_lazy(src_dir / f"cuts_{mixed_name}.jsonl.gz")
    # source_cuts = load_manifest_lazy(src_dir / f"ami-ihm_cuts_dev.jsonl.gz")
    # logging.info("Adding speaker prefix to dev sessions")
    # _process(
    #     mixed_cuts,
    #     source_cuts,
    #     src_dir / f"cuts_{mixed_name}_speaker.jsonl.gz",
    #     output_dir / f"feats_{mixed_name}_speaker",
    # )

    # Test sessions
    # for mic in ["ihm-mix", "sdm", "mdm8-bf"]:
    #     mixed_name = f"ami-{mic}_test"
    #     mixed_cuts = load_manifest_lazy(src_dir / f"cuts_{mixed_name}.jsonl.gz")
    #     mixed_cuts = mixed_cuts.trim_to_supervision_groups(max_pause=0.0)
    #     if mic == "sdm":
    #         source_cuts = load_manifest_lazy(src_dir / "ami-sdm_cuts_test.jsonl.gz")
    #     else:
    #         source_cuts = load_manifest_lazy(src_dir / "ami-ihm_cuts_test.jsonl.gz")
    #     logging.info(f"Adding speaker prefix to {mic} test sessions")
    #     _process(
    #         mixed_cuts,
    #         source_cuts,
    #         src_dir / f"cuts_{mixed_name}_grouped_speaker.jsonl.gz",
    #         output_dir / f"feats_{mixed_name}_grouped_speaker",
    #     )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    random.seed(args.random_seed)
    add_speaker_prefix(args.speaker_buffer_frames)
