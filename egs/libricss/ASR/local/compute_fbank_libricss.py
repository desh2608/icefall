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
This file computes fbank features of the LibriCSS dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import torch
from lhotse import load_manifest_lazy, LilcomChunkyWriter
from lhotse.features.kaldifeat import (
    KaldifeatFbank,
    KaldifeatFbankConfig,
    KaldifeatMelOptions,
    KaldifeatFrameOptions,
)
from lhotse.manipulation import combine

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default="data/manifests")
    parser.add_argument("--output-dir", type=Path, default="data/fbank")
    parser.add_argument("--type", type=str, default="enh", choices=["enh", "orig"])
    return parser.parse_args()


def compute_fbank_libricss(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sampling_rate = 16000
    num_mel_bins = 80

    extractor = KaldifeatFbank(
        KaldifeatFbankConfig(
            frame_opts=KaldifeatFrameOptions(sampling_rate=sampling_rate),
            mel_opts=KaldifeatMelOptions(num_bins=num_mel_bins),
            device="cuda",
        )
    )

    for partition in ["dev", "test"]:
        logging.info(f"Processing {partition}")
        cut_set = load_manifest_lazy(
            data_dir / f"raw_cuts_{partition}_{args.type}.jsonl"
        )
        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=output_dir / f"feats_{partition}_{args.type}",
            manifest_path=data_dir / f"cuts_{partition}_{args.type}.jsonl.gz",
            batch_duration=500,
            num_workers=4,
            storage_type=LilcomChunkyWriter,
            overwrite=True,
        )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    compute_fbank_libricss(args)
