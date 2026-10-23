"""Integration checks for the migrated DSpark offline evaluator dependency."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import evaluate_ep  # noqa: E402


class OfflineEvaluatorDependencyTests(unittest.TestCase):
    def test_loads_migrated_evaluator_with_required_interface(self) -> None:
        evaluator = evaluate_ep._load_legacy_evaluator()

        required_names = (
            "_parse_bool_override",
            "_load_vocab_mapping_tensors",
            "_ensure_loaded_vocab_mappings",
            "build_dspark_tree_proposal",
            "build_tree_attention_mask",
            "verify_draft_tree",
            "DSparkOfflineRunner",
            "BaseModelOfflineRunner",
            "resolve_stop_token_ids",
            "_discover_datasets",
            "_split_csv",
            "_evaluate_dataset",
            "_write_outputs",
        )
        for name in required_names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(evaluator, name), name)


if __name__ == "__main__":
    unittest.main()
