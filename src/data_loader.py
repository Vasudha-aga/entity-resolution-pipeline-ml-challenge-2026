"""
Data loading utilities for the Amazon ML Challenge 2026 Business Entity Resolution.

Supports memory-efficient streaming, chunked processing, and batch loading
for large-scale TSV entity datasets and ground truth mappings.
"""

import os
import csv
from typing import Dict, List, Iterator, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class Entity:
    """Represents a business entity record."""
    entity_id: str
    business_name: str
    business_address: str
    country: str

    @classmethod
    def from_row(cls, row: List[str]) -> "Entity":
        eid = row[0].strip() if len(row) > 0 else ""
        name = row[1].strip() if len(row) > 1 else ""
        addr = row[2].strip() if len(row) > 2 else ""
        country = row[3].strip() if len(row) > 3 else ""
        return cls(entity_id=eid, business_name=name, business_address=addr, country=country)


class DataLoader:
    """DataLoader for Entity Resolution TSV datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, "train")
        self.test_dir = os.path.join(data_dir, "test")

    def get_file_path(self, split: str, source: str) -> str:
        """
        Get the path to a specific source TSV file.
        
        Args:
            split: 'train' or 'test'
            source: 'source1', 'source2', 'source3', or 'ground_truth'
        """
        sub_dir = self.train_dir if split == "train" else self.test_dir
        if source == "ground_truth":
            return os.path.join(sub_dir, f"{split}_ground_truth.tsv")
        return os.path.join(sub_dir, f"{split}_{source}.tsv")

    def stream_entities(self, file_path: str, chunk_size: Optional[int] = None) -> Iterator[List[Entity]]:
        """
        Stream entities from a TSV file in chunks to optimize memory.
        
        Args:
            file_path: Absolute or relative path to TSV file.
            chunk_size: If None, yields single-entity lists. If int, yields batches.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        chunk: List[Entity] = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            _ = next(reader, None)  # Skip header
            for row in reader:
                if not row:
                    continue
                entity = Entity.from_row(row)
                if chunk_size is None:
                    yield [entity]
                else:
                    chunk.append(entity)
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
            if chunk:
                yield chunk

    def load_ground_truth(self, file_path: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Load ground truth mappings into a dictionary.
        
        Returns:
            Dict mapping source1_entity_id -> list of matched S2 and S3 entity IDs.
        """
        if file_path is None:
            file_path = self.get_file_path("train", "ground_truth")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Ground truth file not found: {file_path}")

        gt_mapping: Dict[str, List[str]] = {}
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            _ = next(reader, None)  # Skip header
            for row in reader:
                if not row:
                    continue
                s1_id = row[0].strip()
                matched_str = row[1].strip() if len(row) > 1 else ""
                matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
                gt_mapping[s1_id] = matched_ids
        return gt_mapping

    def split_matches_by_source(self, matched_ids: List[str]) -> Tuple[List[str], List[str]]:
        """
        Partition a list of matched IDs into (S2_matches, S3_matches).
        """
        s2_list = [m for m in matched_ids if m.startswith("S2-")]
        s3_list = [m for m in matched_ids if m.startswith("S3-")]
        return s2_list, s3_list
