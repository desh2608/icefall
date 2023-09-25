#!/usr/bin/env python3
# Copyright    2021  Johns Hopkins University (Piotr Żelasko)
# Copyright    2021  Xiaomi Corp.             (Fangjun Kuang)
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

import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
from lhotse import CutSet, FeatureSet, KaldifeatFbank, KaldifeatFbankConfig
from lhotse.recipes.utils import read_manifests_if_cached

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")
torchaudio.set_audio_backend("soundfile")


def compute_fbank_gigaspeech_dev_test():
    out_dir = Path("data/fbank")
    src_dir = Path("data/manifests")
    # number of workers in dataloader
    num_workers = 20

    # number of seconds in a batch
    batch_duration = 2000

    subsets = ["M"]

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))

    logging.info(f"device: {device}")

    # prefix = "gigaspeech"
    # suffix = "jsonl.gz"
    # manifests = read_manifests_if_cached(
    #     dataset_parts=subsets,
    #     output_dir=src_dir,
    #     prefix=prefix,
    #     suffix=suffix,
    # )
    # assert manifests is not None

    # for partition in subsets:
    # cut_set = CutSet.from_manifests(**manifests[partition])
    cut_set = CutSet.from_file(src_dir / f"gigaspeech_cuts_200h_new.jsonl.gz")
    logging.info("Computing features")
    cut_set = cut_set.compute_and_store_features_batch(
        extractor=extractor,
        storage_path=f"{out_dir}/feats_200h",
        num_workers=num_workers,
        batch_duration=batch_duration,
        overwrite=True,
    )

    cut_set.to_file(src_dir / f"gigaspeech_cuts_200h.jsonl.gz")

    # with FeatureSet.open_writer(
    #     src_dir / f"gigaspeech_feats_{partition}.jsonl.gz"
    # ) as f:
    #     for cut in tqdm(cut_set):
    #         f.write(cut.features)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank_gigaspeech_dev_test()


if __name__ == "__main__":
    main()
