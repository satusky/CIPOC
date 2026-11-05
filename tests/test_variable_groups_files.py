import json
import unittest
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

from cipoc.tools import load_group_hierarchy, load_stages, load_variable_groups


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def variable_groups_files() -> list[Path]:
    """Every JSON file in ``config/`` with a top-level ``groups`` list."""
    files = []
    for path in sorted(CONFIG_DIR.glob("*.json")):
        with open(path, "r") as f:
            if "groups" in json.load(f):
                files.append(path)
    return files


class VariableGroupsFileAgreementTests(unittest.TestCase):
    """The variable groups files are copies, so shared groups must not drift apart."""

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.files = variable_groups_files()

    def pairs(self, load):
        loaded = {path.name: load(path) for path in self.files}
        return combinations(loaded.items(), 2)

    def test_finds_the_default_file_and_its_copies(self):
        self.assertIn("variable_groups.json", [path.name for path in self.files])
        self.assertGreater(len(self.files), 1)

    def test_shared_groups_load_identically(self):
        # Compare loaded groups, so a subgroup's inherited stage, gate and note
        # filter count, not just what its own JSON says.
        def load(path):
            return {group.group_id: group.model_dump() for group in load_variable_groups(path)}

        for (file_a, groups_a), (file_b, groups_b) in self.pairs(load):
            for group_id in sorted(groups_a.keys() & groups_b.keys()):
                with self.subTest(files=(file_a, file_b), group_id=group_id):
                    self.assertEqual(groups_a[group_id], groups_b[group_id])

    def test_shared_groups_have_the_same_parent(self):
        def load(path):
            return {node.group_id: asdict(node) for node in load_group_hierarchy(path)}

        for (file_a, nodes_a), (file_b, nodes_b) in self.pairs(load):
            for group_id in sorted(nodes_a.keys() & nodes_b.keys()):
                with self.subTest(files=(file_a, file_b), group_id=group_id):
                    self.assertEqual(nodes_a[group_id], nodes_b[group_id])

    def test_shared_stages_run_in_the_same_order(self):
        # A copy may add a stage, but the stages it shares keep their order.
        for (file_a, stages_a), (file_b, stages_b) in self.pairs(load_stages):
            shared = set(stages_a) & set(stages_b)
            with self.subTest(files=(file_a, file_b)):
                self.assertEqual(
                    [stage for stage in stages_a if stage in shared],
                    [stage for stage in stages_b if stage in shared],
                )


if __name__ == "__main__":
    unittest.main()
