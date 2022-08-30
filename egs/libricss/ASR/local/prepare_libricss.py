#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Data preparation for LibriCSS dataset.

from pathlib import Path

from lhotse.recipes import prepare_libricss
from lhotse import CutSet, Recording, RecordingSet, SupervisionSet, SupervisionSegment
from lhotse.kaldi import export_to_kaldi

import logging


SESSIONS = {
    "dev": ["session0"],
    "test": [
        "session1",
        "session2",
        "session3",
        "session4",
        "session5",
        "session6",
        "session7",
        "session8",
        "session9",
    ],
}


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="LibriCSS enhanced dataset preparation."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Path to LibriCSS data directory."
    )
    parser.add_argument(
        "--enhanced-dir",
        type=Path,
        required=True,
        help="Path to enhanced data directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Path to output directory."
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Minimum duration to retain segment.",
    )
    return parser.parse_args()


def main(data_dir, enhanced_dir, output_dir):
    manifests = prepare_libricss(data_dir)

    recordings_enh = []
    supervisions_enh = []
    for audio in enhanced_dir.rglob("*.flac"):
        parts = audio.stem.split("-")
        spkid = parts[1]
        recording = Recording.from_file(audio)
        duration = recording.duration
        if duration <= args.min_duration:
            logging.warning(f"Skipping audio {audio.stem} with duration {duration}.")
            continue
        recordings_enh.append(recording)
        supervisions_enh.append(
            SupervisionSegment(
                id=audio.stem,
                recording_id=audio.stem,
                start=0,
                duration=duration,
                channel=0,
                speaker=spkid,
                text="",
            )
        )
    recordings_enh = RecordingSet.from_recordings(recordings_enh)
    supervisions_enh = SupervisionSet.from_segments(supervisions_enh)

    output_dir.mkdir(parents=True, exist_ok=True)

    for part in ["dev", "test"]:
        logging.info(f"Processing {part}...")
        logging.info(
            "Preparing original cuts which will be used for evaluation as reference"
        )
        recordings = manifests["recordings"].filter(
            lambda r: any(session in r.id for session in SESSIONS[part])
        )
        supervisions = manifests["supervisions"].filter(
            lambda s: any(session in s.recording_id for session in SESSIONS[part])
        )

        cuts_orig = CutSet.from_manifests(
            recordings=recordings, supervisions=supervisions
        ).filter(lambda c: c.channel == 0)
        cuts_orig.to_file(output_dir / f"raw_cuts_{part}_orig.jsonl")

        logging.info("Preparing enhanced cuts which will be used for decoding")
        recordings = recordings_enh.filter(
            lambda r: any(session in r.id for session in SESSIONS[part])
        )
        supervisions = supervisions_enh.filter(
            lambda s: any(session in s.recording_id for session in SESSIONS[part])
        )

        cuts_enh = CutSet.from_manifests(
            recordings=recordings, supervisions=supervisions
        )
        cuts_enh.to_file(output_dir / f"raw_cuts_{part}_enh.jsonl")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    main(args.data_dir, args.enhanced_dir, args.output_dir)
