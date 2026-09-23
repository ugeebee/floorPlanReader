"""Joint dataset mixer for CubiCasa5k and Synthetic floor plans.

Implements Strategy 1 (Joint Blend Mixture) to prevent catastrophic forgetting
by creating a balanced, shuffled dataset of real-world and synthetic blueprints.
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union


class DatasetMixer:
    """Blends and partitions CubiCasa5k and Synthetic floor plan datasets."""

    def __init__(
        self,
        cubicasa_ratio: float = 0.7,
        val_split_ratio: float = 0.1,
        seed: int = 42,
    ):
        """Initialize DatasetMixer.

        Args:
            cubicasa_ratio: Target proportion of CubiCasa samples (e.g. 0.7 = 70% CubiCasa, 30% Synthetic).
            val_split_ratio: Proportion of total samples allocated to validation split.
            seed: Random seed for reproducible shuffling.
        """
        self.cubicasa_ratio = cubicasa_ratio
        self.synthetic_ratio = 1.0 - cubicasa_ratio
        self.val_split_ratio = val_split_ratio
        self.seed = seed

    def mix_and_split(
        self,
        cubicasa_samples: List[Dict[str, Any]],
        synthetic_samples: List[Dict[str, Any]],
        target_total_samples: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Mix and partition samples into train and validation sets.

        Args:
            cubicasa_samples: List of Qwen-VL conversation dicts from CubiCasa.
            synthetic_samples: List of Qwen-VL conversation dicts from synthetic generator.
            target_total_samples: Optional maximum total dataset size (useful for quick Colab runs).

        Returns:
            Tuple of (train_samples, val_samples).
        """
        rng = random.Random(self.seed)

        if not cubicasa_samples and not synthetic_samples:
            raise ValueError("Both cubicasa_samples and synthetic_samples are empty!")

        # If only one source is provided, use that source
        if not cubicasa_samples:
            all_samples = list(synthetic_samples)
        elif not synthetic_samples:
            all_samples = list(cubicasa_samples)
        else:
            # Determine mix counts
            if target_total_samples:
                n_cubi = int(round(target_total_samples * self.cubicasa_ratio))
                n_synth = target_total_samples - n_cubi
            else:
                # Scale up to include all available synthetic data or cubicasa
                # Use all synthetic samples and balance cubicasa to match ratio
                n_synth = len(synthetic_samples)
                n_cubi = int(round(n_synth * (self.cubicasa_ratio / self.synthetic_ratio)))
                n_cubi = min(n_cubi, len(cubicasa_samples))

            # Sample (with replacement if a dataset is smaller than needed)
            if len(cubicasa_samples) >= n_cubi:
                sampled_cubi = rng.sample(cubicasa_samples, n_cubi)
            else:
                sampled_cubi = [rng.choice(cubicasa_samples) for _ in range(n_cubi)]

            if len(synthetic_samples) >= n_synth:
                sampled_synth = rng.sample(synthetic_samples, n_synth)
            else:
                sampled_synth = [rng.choice(synthetic_samples) for _ in range(n_synth)]

            all_samples = sampled_cubi + sampled_synth

        # Shuffle thoroughly
        rng.shuffle(all_samples)

        # Split into train and val
        val_count = max(1, int(round(len(all_samples) * self.val_split_ratio)))
        val_samples = all_samples[:val_count]
        train_samples = all_samples[val_count:]

        return train_samples, val_samples

    def save_jsonl(
        self,
        samples: List[Dict[str, Any]],
        output_file: Union[str, Path],
    ) -> Path:
        """Write list of sample dicts to a JSON Lines file."""
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for item in samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return out_p

    def build_and_export(
        self,
        cubicasa_samples: List[Dict[str, Any]],
        synthetic_samples: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        target_total_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mix, split, and save train.jsonl and val.jsonl with a metadata summary.

        Args:
            cubicasa_samples: CubiCasa samples.
            synthetic_samples: Synthetic samples.
            output_dir: Folder where train.jsonl, val.jsonl, and metadata.json are written.
            target_total_samples: Optional size cap.

        Returns:
            Dictionary with dataset summary statistics.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_set, val_set = self.mix_and_split(
            cubicasa_samples=cubicasa_samples,
            synthetic_samples=synthetic_samples,
            target_total_samples=target_total_samples,
        )

        train_path = self.save_jsonl(train_set, out_dir / "train.jsonl")
        val_path = self.save_jsonl(val_set, out_dir / "val.jsonl")

        summary = {
            "total_samples": len(train_set) + len(val_set),
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "cubicasa_input_count": len(cubicasa_samples),
            "synthetic_input_count": len(synthetic_samples),
            "target_cubicasa_ratio": self.cubicasa_ratio,
            "train_file": str(train_path.resolve()),
            "val_file": str(val_path.resolve()),
        }

        with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
