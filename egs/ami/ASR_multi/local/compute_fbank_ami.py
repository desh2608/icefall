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
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import torch
from lhotse import (
    CutSet,
    LilcomChunkyWriter,
    RecordingSet,
    SupervisionSet,
    load_manifest,
)
from lhotse.features.kaldifeat import (
    KaldifeatFbank,
    KaldifeatFbankConfig,
    KaldifeatFrameOptions,
    KaldifeatMelOptions,
)
from lhotse.manipulation import combine
from lhotse.utils import Pathlike
from tqdm import tqdm

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
    parser.add_argument("--mic", type=str, default="ihm", choices=["ihm", "sdm", "gss"])
    parser.add_argument("--dataset-parts", type=str, nargs="+", default=["dev", "test"])
    parser.add_argument(
        "--rttm-affix",
        type=str,
        default="",
        help="Affix to use for supervisions created from RTTM files",
    )
    parser.add_argument(
        "--gss-affix",
        type=str,
        default="",
        help="Affix to use for GSS-enhanced cut manifests",
    )
    return parser.parse_args()


def read_manifests(
    data_dir: Pathlike,
    dataset_parts: Optional[Sequence[str]] = None,
    prefix: str = "",
    supervision_suffix: Optional[str] = "",
) -> Optional[Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:
    """
    This is modified from lhotse.recipe.utils.read_manifests_if_cached.
    """
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"
    manifests = defaultdict(dict)
    dataset_parts = [] if dataset_parts is None else dataset_parts
    for part in dataset_parts:
        for manifest in ["recordings", "supervisions", "cuts"]:
            if manifest == "recordings":
                path = data_dir / f"{prefix}{manifest}_{part}.jsonl.gz"
            else:
                path = (
                    data_dir / f"{prefix}{manifest}_{part}{supervision_suffix}.jsonl.gz"
                )
            if not path.is_file():
                continue
            manifests[part][manifest] = load_manifest(path)
    return dict(manifests)


def compute_fbank_ami(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mic != "gss":
        args.gss_affix = ""

    sampling_rate = 16000
    num_mel_bins = 80

    extractor = KaldifeatFbank(
        KaldifeatFbankConfig(
            frame_opts=KaldifeatFrameOptions(sampling_rate=sampling_rate),
            mel_opts=KaldifeatMelOptions(num_bins=num_mel_bins),
            device="cuda",
        )
    )

    manifests = read_manifests(
        data_dir=data_dir,
        dataset_parts=args.dataset_parts,
        prefix=f"ami-{args.mic}",
        supervision_suffix=f"{args.rttm_affix}{args.gss_affix}",
    )

    for partition in args.dataset_parts:
        logging.info(f"Processing {partition}")
        if args.mic == "gss":
            cut_set = manifests[partition]["cuts"]
        else:
            cut_set = CutSet.from_manifests(
                **manifests[partition]
            ).trim_to_supervisions(keep_overlapping=False, keep_all_channels=False)

        storage_path = (
            output_dir
            / f"feats_{partition}_{args.mic}{args.rttm_affix}{args.gss_affix}"
        )
        manifest_path = (
            data_dir
            / f"cuts_{partition}_{args.mic}{args.rttm_affix}{args.gss_affix}.jsonl.gz"
        )
        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=storage_path,
            manifest_path=manifest_path,
            batch_duration=1000,
            num_workers=2,
            storage_type=LilcomChunkyWriter,
            overwrite=True,
        )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    compute_fbank_ami(args)
