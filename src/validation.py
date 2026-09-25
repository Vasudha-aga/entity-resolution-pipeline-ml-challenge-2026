"""
Validation split infrastructure for Amazon ML Challenge 2026.

Provides deterministic entity-level train/validation splitting
for Source 1 entities while preserving ground-truth relationships.
"""

import os
import random
from typing import List, Set, Dict, Tuple, Optional
from src.data_loader import DataLoader


class ValidationSplitter:
    """Manages deterministic entity-level dataset splitting."""

    def __init__(self, data_loader: Optional[DataLoader] = None, seed: int = 42):
        self.data_loader = data_loader or DataLoader()
        self.seed = seed

    def get_all_source1_entity_ids(self, split: str = "train") -> List[str]:
        """
        Extract all Source 1 entity IDs from the master TSV file.
        """
        s1_file = self.data_loader.get_file_path(split, "source1")
        entity_ids = []
        for batch in self.data_loader.stream_entities(s1_file, chunk_size=100000):
            entity_ids.extend([e.entity_id for e in batch])
        return entity_ids

    def split_entity_ids(
        self,
        entity_ids: List[str],
        val_ratio: float = 0.20
    ) -> Tuple[List[str], List[str]]:
        """
        Deterministically split a list of entity IDs into train and validation sets.
        
        Args:
            entity_ids: List of unique Source 1 entity IDs.
            val_ratio: Proportion of entities allocated to validation (default: 0.20).
            
        Returns:
            Tuple of (train_ids, val_ids).
        """
        # Sort first to guarantee deterministic ordering independent of filesystem/OS
        sorted_ids = sorted(entity_ids)
        
        rng = random.Random(self.seed)
        shuffled_ids = sorted_ids.copy()
        rng.shuffle(shuffled_ids)

        total_count = len(shuffled_ids)
        val_count = int(total_count * val_ratio)
        train_count = total_count - val_count

        train_ids = shuffled_ids[:train_count]
        val_ids = shuffled_ids[train_count:]

        return train_ids, val_ids

    def save_split_ids(self, ids: List[str], file_path: str) -> None:
        """
        Save a list of entity IDs to a text file (one ID per line).
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for eid in ids:
                f.write(f"{eid}\n")

    def load_split_ids(self, file_path: str) -> Set[str]:
        """
        Load a set of entity IDs from a text file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Split file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def filter_ground_truth_for_split(
        self,
        ground_truth: Dict[str, List[str]],
        split_ids: Set[str]
    ) -> Dict[str, List[str]]:
        """
        Extract ground truth mappings corresponding only to entities in split_ids.
        """
        return {s1_id: matches for s1_id, matches in ground_truth.items() if s1_id in split_ids}

    def generate_and_save_splits(
        self,
        output_dir: str = "experiments/splits",
        val_ratio: float = 0.20
    ) -> Tuple[str, str]:
        """
        Generate train and validation splits for training Source 1 entities and save to disk.
        
        Returns:
            Tuple of (train_split_path, val_split_path).
        """
        s1_ids = self.get_all_source1_entity_ids("train")
        train_ids, val_ids = self.split_entity_ids(s1_ids, val_ratio=val_ratio)

        train_path = os.path.join(output_dir, "train_s1_ids.txt")
        val_path = os.path.join(output_dir, "val_s1_ids.txt")

        self.save_split_ids(train_ids, train_path)
        self.save_split_ids(val_ids, val_path)

        return train_path, val_path
