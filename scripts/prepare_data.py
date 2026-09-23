"""CLI script to prepare and mix CubiCasa5k and Synthetic floor plan datasets."""

import argparse
import sys
from pathlib import Path

from floorplan_reader.dataset.cubicasa_parser import CubiCasaParser
from floorplan_reader.dataset.synthetic_adapter import (
    SyntheticFloorPlanAdapter,
    generate_mock_synthetic_dataset,
)
from floorplan_reader.dataset.dataset_mixer import DatasetMixer


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare mixed floor plan dataset for Qwen-VL LoRA fine-tuning.")
    parser.add_argument(
        "--synthetic_dir",
        type=str,
        default=None,
        help="Path to directory containing synthetic floor plans and json annotations.",
    )
    parser.add_argument(
        "--generate_mock_synthetic",
        type=int,
        default=0,
        help="Generate N mock synthetic floor plan samples for quick pipeline testing.",
    )
    parser.add_argument(
        "--cubicasa_dir",
        type=str,
        default=None,
        help="Path to local CubiCasa5k dataset directory (or omit to load from HuggingFace).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Directory to save train.jsonl, val.jsonl, and converted images.",
    )
    parser.add_argument(
        "--cubicasa_ratio",
        type=float,
        default=0.7,
        help="Target proportion of CubiCasa samples (default 0.7 = 70% CubiCasa, 30% Synthetic).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split proportion (default 0.1 = 10%).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum total samples to produce (useful for fast Colab test runs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    synthetic_samples = []
    synth_adapter = SyntheticFloorPlanAdapter()

    # 1. Handle Synthetic Data
    if args.generate_mock_synthetic > 0:
        print(f"Generating {args.generate_mock_synthetic} mock synthetic floor plans...")
        mock_dir = out_dir / "mock_synthetic"
        generate_mock_synthetic_dataset(mock_dir, num_samples=args.generate_mock_synthetic, format="svg")
        synthetic_samples = synth_adapter.ingest_paired_directory(mock_dir, output_images_dir=images_dir)
        print(f"Successfully processed {len(synthetic_samples)} synthetic samples.")
    elif args.synthetic_dir:
        synth_p = Path(args.synthetic_dir)
        print(f"Ingesting synthetic dataset from {synth_p}...")
        synthetic_samples = synth_adapter.ingest_paired_directory(synth_p, output_images_dir=images_dir)
        print(f"Successfully processed {len(synthetic_samples)} synthetic samples from {synth_p}.")
    else:
        print("Note: No synthetic dataset provided (--synthetic_dir) and --generate_mock_synthetic not set.")

    # 2. Handle CubiCasa Data
    cubicasa_samples = []
    cubi_parser = CubiCasaParser()

    if args.cubicasa_dir:
        print(f"Loading CubiCasa from local directory {args.cubicasa_dir}...")
        # Local CubiCasa directory traversal
        # Implement directory loader
        cubi_dir = Path(args.cubicasa_dir)
        # Search for svg/png pairs
        for svg_f in cubi_dir.rglob("model.svg"):
            sample_name = svg_f.parent.name
            img_candidate = svg_f.parent / "color.png"
            if not img_candidate.exists():
                img_candidate = svg_f.parent / "original.png"
            if img_candidate.exists():
                try:
                    entry = cubi_parser.parse_svg_model(svg_f, img_candidate, sample_id=sample_name)
                    cubicasa_samples.append(entry)
                except Exception as e:
                    pass
        print(f"Loaded {len(cubicasa_samples)} local CubiCasa samples.")
    else:
        print("Tip: In Colab or local training, you can stream or load CubiCasa from HuggingFace Hub ('phungpx/cubicasa5k-coco').")

    if not synthetic_samples and not cubicasa_samples:
        print("Error: No samples available to mix. Please specify --synthetic_dir, --generate_mock_synthetic, or --cubicasa_dir.")
        sys.exit(1)

    # 3. Mix and Export
    print(f"Mixing datasets (CubiCasa ratio: {args.cubicasa_ratio:.1%}, Synthetic ratio: {1.0 - args.cubicasa_ratio:.1%})...")
    mixer = DatasetMixer(
        cubicasa_ratio=args.cubicasa_ratio,
        val_split_ratio=args.val_split,
    )

    summary = mixer.build_and_export(
        cubicasa_samples=cubicasa_samples,
        synthetic_samples=synthetic_samples,
        output_dir=out_dir,
        target_total_samples=args.max_samples,
    )

    print("\nDataset Preparation Complete!")
    print(f" - Train Samples: {summary['train_samples']}")
    print(f" - Val Samples:   {summary['val_samples']}")
    print(f" - Train File:    {summary['train_file']}")
    print(f" - Val File:      {summary['val_file']}")


if __name__ == "__main__":
    main()
