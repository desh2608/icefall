# Copyright      2021  Piotr Żelasko
# Copyright      2021  Xiaomi Corporation (Author: Mingshuang Luo)
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
import random
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    SupervisionSegment,
    load_manifest,
    load_manifest_lazy,
)
from lhotse.cut import Cut
from lhotse.dataset import (
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    SingleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import (
    OnTheFlyFeatures,
    BatchIO,
    PrecomputedFeatures,
)
from lhotse.utils import (
    TimeSpan,
    Seconds,
    compute_num_samples,
    ifnone,
    overlaps,
    overspans,
)
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class SpeechRecognitionDataset(K2SpeechRecognitionDataset):
    def __init__(
        self,
        return_cuts: bool = False,
        input_strategy: BatchIO = PrecomputedFeatures(),
    ):
        super().__init__(return_cuts=return_cuts, input_strategy=input_strategy)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[Cut]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_frames and max_cuts.
        """
        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        # Get a tensor with batched feature matrices, shape (B, T, F)
        # Collation performs auto-padding, if necessary.
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            # An input strategy with fault tolerant audio reading mode.
            # "cuts" may be a subset of the original "cuts" variable,
            # that only has cuts for which we succesfully read the audio.
            inputs, _, cuts = input_tpl
        else:
            inputs, _ = input_tpl

        # Get a dict of tensors that encode the positional information about supervisions
        # in the batch of feature matrices. The tensors are named "sequence_idx",
        # "start_frame/sample" and "num_frames/samples".
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        batch = {"inputs": inputs, "supervisions": supervision_intervals}
        if self.return_cuts:
            batch["supervisions"]["cut"] = [cut for cut in cuts]

        return batch


# The following is from: https://github.com/lhotse-speech/lhotse/discussions/1092#discussioncomment-6184497
def sample_alignment_segment(
    cut: Cut,
    min_duration: Seconds,
    max_duration: Seconds,
    supervisions_index: Optional[Any] = None,
    seed: Optional[int] = None,
) -> Cut:
    """
    Given a cut that has possibly multiple supervisions with alignments,
    create a sub-cut with a single supervision that may combine text from several supervisions.
    We use the word-level alignment to determine output cut's transcript.
    The output cut's duration is sampled uniformly between ``min_duration`` and ``max_duration``.

    Example usage::

        >>> cuts = CutSet.from_file(...)  # long cuts with multiple supervisions
        >>> segment_cuts = cuts.repeat().map(sample_alignment_segment)  # infinite cut set of segments

    """
    if cut.duration < min_duration:
        return cut

    rng = random if seed is None else random.Random(seed)

    def _quantize(dur: Seconds) -> Seconds:
        # Avoid potential numerical issues later on
        num_samples = compute_num_samples(dur, cut.sampling_rate)
        return num_samples / cut.sampling_rate

    start = _quantize(rng.random() * (cut.duration - min_duration))
    duration = rng.uniform(min_duration, max_duration)

    # ensure that cut end time is not exceeded
    duration = min(duration, cut.duration - start)

    alignment_items = []
    trimmed = cut.truncate(
        offset=start,
        duration=duration,
        keep_excessive_supervisions=True,
        _supervisions_index=supervisions_index,
    )
    for s in trimmed.supervisions:
        # collect all alignment items that fall within the segment
        for ai in ifnone(s.alignment, {}).get("word", []):
            if overlaps(TimeSpan(0, duration), ai):
                alignment_items.append(ai)

    supervision = SupervisionSegment(
        id=trimmed.id,
        recording_id=trimmed.recording_id,
        start=0,
        duration=duration,
        text=" ".join(ai.symbol for ai in alignment_items),
    )
    trimmed.supervisions = [supervision]
    return trimmed


class TedLiumAsrDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. TEDLium3 dev
    and test).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/fbank"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--uniform-duration",
            type=float,
            default=25.0,
            help="Uniform segment duration used for dynamic training.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="When enabled, use on-the-fly cut mixing and feature "
            "extraction. Will drop existing precomputed feature manifests "
            "if available.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="When enabled, use SpecAugment for training dataset.",
        )
        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )
        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=True,
            help="When enabled, select noise from MUSAN and mix it"
            "with training dataset.",
        )
        group.add_argument(
            "--lf-affix",
            type=str,
            default=None,
            help="Affix to add to the manifest name for long-form training",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
        prefetch_factor: Optional[int] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")

            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=10,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                    max_frames_mask_fraction=0.15,
                    p=0.9,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to get Musan cuts")
        transforms = []
        if self.args.enable_musan:
            logging.info("Enable MUSAN")
            cuts_musan = load_manifest(self.args.manifest_dir / "musan_cuts.jsonl.gz")
            transforms.append(
                CutMix(cuts=cuts_musan, prob=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

        if self.args.concatenate_cuts:
            logging.info(
                f"Using cut concatenation with duration factor "
                f"{self.args.duration_factor} and gap {self.args.gap}."
            )
            # Cut concatenation should be the first transform in the list,
            # so that if we e.g. mix noise in, it will fill the gaps between
            # different utterances.
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        logging.info("About to create train dataset")
        if self.args.on_the_fly_feats:
            # NOTE: the PerturbSpeed transform should be added only if we
            # remove it from data prep stage.
            # Add on-the-fly speed perturbation; since originally it would
            # have increased epoch size by 3, we will apply prob 2/3 and use
            # 3x more epochs.
            # Speed perturbation probably should come first before
            # concatenation, but in principle the transforms order doesn't have
            # to be strict (e.g. could be randomized)
            # transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2/3)] + transforms   # noqa
            # Drop feats to be on the safe side.
            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
            )
        else:
            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
            )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                drop_last=True,
                quadratic_duration=30.0,
            )
        else:
            logging.info("Using SingleCutSampler.")
            train_sampler = SingleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        logging.info("About to create train dataloader")
        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:

        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                return_cuts=self.args.return_cuts,
            )

        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )

        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(self, cuts_test: CutSet, chunked: bool = False) -> DataLoader:

        logging.debug("About to create test dataset")
        if chunked:
            test = SpeechRecognitionDataset(
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
                if self.args.on_the_fly_feats
                else PrecomputedFeatures(),
                return_cuts=self.args.return_cuts,
            )
        else:
            test = K2SpeechRecognitionDataset(
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
                if self.args.on_the_fly_feats
                else PrecomputedFeatures(),
                return_cuts=self.args.return_cuts,
            )

        sampler = DynamicBucketingSampler(
            cuts_test,
            max_duration=self.args.max_duration,
            shuffle=False,
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
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        lf_affix = f"_{self.args.lf_affix}" if self.args.lf_affix else ""
        return load_manifest_lazy(
            self.args.manifest_dir / f"tedlium_cuts_train{lf_affix}.jsonl.gz"
        )

    @lru_cache()
    def train_cuts_dynamic(
        self, min_duration: Seconds = 1.0, max_duration: Seconds = 20.0
    ) -> CutSet:
        logging.info("About to get dynamically segmented train cuts (infinite)")
        cuts = load_manifest_lazy(self.args.manifest_dir / "cuts_train_full.jsonl.gz")
        supervisions_index = cuts.index_supervisions()
        fn = partial(
            sample_alignment_segment,
            min_duration=min_duration,
            max_duration=max_duration,
            supervisions_index=supervisions_index,
        )
        return cuts.repeat(preserve_id=True).map(fn)  # infinite cut set of segments

    @lru_cache()
    def train_cuts_uniform(self) -> CutSet:
        logging.info("About to get uniformly segmented train cuts")
        cuts = (
            load_manifest_lazy(self.args.manifest_dir / "cuts_train_full.jsonl.gz")
            .merge_supervisions(merge_policy="keep_first")
            .trim_to_alignments(
                type="word",
                max_pause=2.0,
                max_segment_duration=self.args.uniform_duration,
            )
            .shuffle()
            .filter(lambda c: c.duration > 0.5)
        )
        return cuts

    @lru_cache()
    def dev_cuts(self, affix: str = "") -> CutSet:
        logging.info("About to get dev cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / f"tedlium_cuts_dev{affix}.jsonl.gz"
        )

    @lru_cache()
    def test_cuts(self, affix: str = "") -> CutSet:
        logging.info("About to get test cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / f"tedlium_cuts_test{affix}.jsonl.gz"
        )
