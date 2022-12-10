# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
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


import argparse
import logging
from functools import lru_cache
from pathlib import Path

from lhotse import CutSet, Fbank, FbankConfig, load_manifest
from lhotse.dataset import BucketingSampler, PrecomputedFeatures, UnsupervisedDataset
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class AlimeetingAsrDataModule:
    """
    DataModule for k2 ASR experiments.
    This is only for evaluation, so we only provide test dataloader.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description=(
                "These options are used for the preparation of "
                "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
                "effective batch sizes, sampling strategies, applied data "
                "augmentations, etc."
            ),
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/manifests"),
            help="Path to directory with dev/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=200.0,
            help=(
                "Maximum pooled recordings duration (seconds) in a "
                "single batch. You can reduce it if it causes CUDA OOM."
            ),
        )
        group.add_argument(
            "--max-cuts",
            type=int,
            default=100,
            help=(
                "Maximum number of cuts in a single batch. You can "
                "reduce it if it causes CUDA OOM."
            ),
        ),
        group.add_argument(
            "--num-buckets",
            type=int,
            default=20,
            help=(
                "The number of buckets for the BucketingSampler"
                "(you might want to increase it for larger datasets)."
            ),
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help=(
                "When enabled, use on-the-fly cut mixing and feature "
                "extraction. Will drop existing precomputed feature manifests "
                "if available."
            ),
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help=(
                "The number of training dataloader workers that " "collect the batches."
            ),
        )

        group.add_argument(
            "--rttm-affix",
            type=str,
            default="",
            help="The affix of RTTM file name, e.g. `_spectral`.",
        )
        group.add_argument(
            "--gss-affix",
            type=str,
            default="",
            help="GSS affix to distinguish different enhanced manifests.",
        )

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = UnsupervisedDataset()
        sampler = BucketingSampler(
            cuts,
            num_buckets=self.args.num_buckets,
            shuffle=False,
            max_duration=self.args.max_duration,
            max_cuts=self.args.max_cuts,
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def eval_sdm_cuts(self) -> CutSet:
        logging.info("About to get SDM eval cuts")
        return load_manifest(
            self.args.manifest_dir / f"cuts_eval_sdm{self.args.rttm_affix}.jsonl.gz"
        )

    @lru_cache()
    def eval_gss_cuts(self) -> CutSet:
        logging.info("About to get GSS-enhanced eval cuts")
        return load_manifest(
            self.args.manifest_dir
            / f"cuts_eval_gss{self.args.rttm_affix}{self.args.gss_affix}.jsonl.gz"
        )

    @lru_cache()
    def test_sdm_cuts(self) -> CutSet:
        logging.info("About to get SDM test cuts")
        return load_manifest(
            self.args.manifest_dir / f"cuts_test_sdm{self.args.rttm_affix}.jsonl.gz"
        )

    @lru_cache()
    def test_gss_cuts(self) -> CutSet:
        logging.info("About to get GSS-enhanced test cuts")
        return load_manifest(
            self.args.manifest_dir
            / f"cuts_test_gss{self.args.rttm_affix}{self.args.gss_affix}.jsonl.gz"
        )
