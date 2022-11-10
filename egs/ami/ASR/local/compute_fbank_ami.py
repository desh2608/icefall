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
This file computes fbank features of the AMI dataset.
For the training data, we pool together IHM, reverberated IHM, and GSS-enhanced
audios. For the test data, we separately prepare IHM, SDM, and GSS-enhanced
parts (which are the 3 evaluation settings).
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""
import logging
import math
from pathlib import Path

import torch
import torch.multiprocessing
from lhotse import CutSet, LilcomChunkyWriter
from lhotse.features.kaldifeat import (
    KaldifeatFbank,
    KaldifeatFbankConfig,
    KaldifeatFrameOptions,
    KaldifeatMelOptions,
)
from lhotse.recipes.utils import read_manifests_if_cached

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


def compute_fbank_ami():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    sampling_rate = 16000
    num_mel_bins = 80

    extractor = KaldifeatFbank(
        KaldifeatFbankConfig(
            frame_opts=KaldifeatFrameOptions(sampling_rate=sampling_rate),
            mel_opts=KaldifeatMelOptions(num_bins=num_mel_bins),
            device="cuda",
        )
    )

    logging.info("Reading manifests")
    manifests_ihm = read_manifests_if_cached(
        dataset_parts=["train", "dev", "test"],
        output_dir=src_dir,
        prefix="ami-ihm",
        suffix="jsonl.gz",
    )
    manifests_sdm = read_manifests_if_cached(
        dataset_parts=["train", "dev", "test"],
        output_dir=src_dir,
        prefix="ami-sdm",
        suffix="jsonl.gz",
    )
    # For GSS we already have cuts so we read them directly.
    manifests_gss = read_manifests_if_cached(
        dataset_parts=["train", "dev", "test"],
        output_dir=src_dir,
        prefix="ami-gss",
        suffix="jsonl.gz",
        types=["cuts"],
    )

    logging.info(
        "Preparing training cuts: IHM + reverberated IHM + SDM + GSS (optional)"
    )
    cuts_ihm = CutSet.from_manifests(
        **manifests_ihm["train"]
    ).trim_to_supervisions(keep_overlapping=False, keep_all_channels=False)
    cuts_ihm_rvb = cuts_ihm.reverb_rir()
    cuts_sdm = CutSet.from_manifests(
        **manifests_sdm["train"]
    ).trim_to_supervisions(keep_overlapping=False)
    cuts_train = cuts_ihm + cuts_ihm_rvb + cuts_sdm
    if "cuts" in manifests_gss:
        cuts_gss = manifests_gss["train"]["cuts"]
        cuts_train = cuts_train + cuts_gss

    cuts_train = (
        cuts_train
        + cuts_train.perturb_speed(0.9)
        + cuts_train.perturb_speed(1.1)
    )
    # Print out statistics of the training set
    cuts_train.describe()

    # We split the train set and compute features for each
    # (this is to avoid memory leaks)
    num_split = 10
    logging.info(f"Splitting training data into {num_split} parts")
    chunk_size = math.ceil(len([c.id for c in cuts_train]) / num_split)
    cut_sets = cuts_train.split_lazy(
        output_dir=src_dir / f"cuts_train_raw_split{num_split}",
        chunk_size=chunk_size,
    )
    for idx in range(0, num_split):
        logging.info(f"Processing train split {idx+1}")
        _ = cut_sets[idx].compute_and_store_features_batch(
            extractor=extractor,
            storage_path=output_dir / f"feats_train_{idx+1}",
            manifest_path=src_dir / f"cuts_train_{idx+1}.jsonl.gz",
            batch_duration=5000,
            num_workers=8,
            storage_type=LilcomChunkyWriter,
        )

    logging.info("Preparing test cuts: IHM, SDM, GSS (optional)")
    for split in ["test"]:
        logging.info(f"Processing {split} IHM")
        cuts_ihm = (
            CutSet.from_manifests(**manifests_ihm[split])
            .trim_to_supervisions(
                keep_overlapping=False, keep_all_channels=False
            )
            .compute_and_store_features_batch(
                extractor=extractor,
                storage_path=output_dir / f"feats_{split}_ihm",
                manifest_path=src_dir / f"cuts_{split}_ihm.jsonl.gz",
                batch_duration=5000,
                num_workers=8,
                storage_type=LilcomChunkyWriter,
            )
        )
        logging.info(f"Processing {split} SDM")
        cuts_sdm = (
            CutSet.from_manifests(**manifests_sdm[split])
            .trim_to_supervisions(keep_overlapping=False)
            .compute_and_store_features_batch(
                extractor=extractor,
                storage_path=output_dir / f"feats_{split}_sdm",
                manifest_path=src_dir / f"cuts_{split}_sdm.jsonl.gz",
                batch_duration=500,
                num_workers=4,
                storage_type=LilcomChunkyWriter,
            )
        )
        if split in manifests_gss:
            logging.info(f"Processing {split} GSS")
            cuts_gss = (
                manifests_gss[split]["cuts"]
                .filter(lambda c: c.duration > 0)
                .compute_and_store_features_batch(
                    extractor=extractor,
                    storage_path=output_dir / f"feats_{split}_gss",
                    manifest_path=src_dir / f"cuts_{split}_gss.jsonl.gz",
                    batch_duration=500,
                    num_workers=4,
                    storage_type=LilcomChunkyWriter,
                )
            )


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank_ami()
