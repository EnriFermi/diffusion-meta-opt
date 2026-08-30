from __future__ import annotations

import ast
import copy
import csv
import math
import os
import stat
import tempfile
import unittest
from pathlib import Path


os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from scripts import (
    review_one_state_exact_selected_trajectory_recovery_salvage as review,
)


def numeric_audit() -> dict[str, object]:
    return {
        "absolute_tolerance": 1e-9,
        "relative_tolerance": 0.0,
        "matrix_from_hessian_max_abs_error": 3.885780586188048e-16,
        "matrix_symmetry_max_abs_error": 0.0,
        "eigvalsh_matrix_max_abs_error": 6.661338147750939e-15,
        "low_basis_orthogonality_max_abs": 2.1163626406917047e-15,
        "low_basis_eigen_residual_relative": 1.4866823708986403e-15,
        "low_projector_basis_max_abs_error": 1.9984014443252818e-15,
        "metric_closure_max_abs_error": 5.684341886080802e-14,
        "source_metric_max_abs_error": 0.0,
        "spectrum_max_abs_error": 0.0,
        "spectrum_bitwise_equal_count": 512,
        "spectrum_representation_count": 5,
        "rank_2_binary64_hex": "3de22a7f787e6c62",
        "state_csv_rows": 101,
        "spectrum_csv_rows": 51_712,
        "scientific_success": False,
        "failed_scientific_success_gates": review.FAILED_SCIENTIFIC_GATES,
        "final_exact_a_per_dim": 0.9321672207645146,
        "continuation_non_top_only_fraction": 0.11301885339451084,
        "total_non_top_only_fraction": 0.07733408144475684,
    }


def mocked_review_evidence() -> dict[str, object]:
    digest = "a" * 64
    failed_gates = {
        "protocol_and_manifests_frozen": True,
        "failed_tree_exact": True,
        "original_source_tree_exact": True,
        "startup_failure_tree_exact": True,
        "packet_hash_graph_exact": True,
        "checkpoint_lineage_exact": True,
        "task_and_input_provenance_exact": True,
        "model_nonmutation_exact": True,
        "geometry_shapes_dtypes_finite": True,
        "matrix_from_hessian_closure": True,
        "eigensystem_closure": True,
        "low_basis_projector_closure": True,
        "metric_closure": True,
        "spectrum_bitwise_closure": True,
        "csv_checkpoint_closure": True,
        "scientific_outcome_unchanged": True,
    }
    freeze = {
        "publisher": {"raw_sha256": digest, "normalized_sha256": digest},
        "reviewer": {
            "raw_sha256": digest,
            "normalized_sha256": review.EXPECTED_NORMALIZED_SOURCE_SHA256,
        },
        "publisher_test": {"sha256": digest},
        "reviewer_test": {"sha256": digest},
    }
    protected = {
        "failed": {"tree_sha256": review.FAILED_TREE_SHA256},
        "original_source": {"tree_sha256": review.ORIGINAL_TREE_SHA256},
        "startup_failure": {"tree_sha256": review.STARTUP_TREE_SHA256},
        "published": {"tree_sha256": review.PUBLISHED_TREE_SHA256},
    }
    publication = {
        "source_tree_sha256": review.FAILED_TREE_SHA256,
        "published_tree_sha256": review.PUBLISHED_TREE_SHA256,
        "file_count": 32,
        "total_size_bytes": 114_949_656,
        "all_bytes_identical": True,
        "all_inodes_distinct": True,
        "all_link_counts_one": True,
        "inode_pairs": {"fixture": {"source_inode": 1, "published_inode": 2}},
    }
    return {
        "freeze": freeze,
        "freeze_sha256": digest,
        "freeze_validation": {"exact": True},
        "publisher_static": {"policy_pass": True},
        "reviewer_static": {
            "policy_pass": True,
            "normalized_sha256": review.EXPECTED_NORMALIZED_SOURCE_SHA256,
        },
        "failed_audit": {"gates": failed_gates, "numeric_audit": numeric_audit()},
        "publication": publication,
        "target_graph": {"artifact_count": 29},
        "execution": {
            "record": {
                "publisher_snapshot_sha256": digest,
                "scientific_execution": review.SCIENTIFIC_EXECUTION_ZERO,
            },
            "sha256": digest,
            "validation": {"valid": True, "scientific_execution_absent": True},
        },
        "protected_before": protected,
        "protected_after": copy.deepcopy(protected),
    }


class StaticPolicyTest(unittest.TestCase):
    def test_reviewer_and_publisher_are_standalone_cpu_sources(self) -> None:
        policy = {
            "allowed_top_level_imports": review.ALLOWED_TOP_LEVEL_IMPORTS,
            "forbidden_tokens": review.FORBIDDEN_POLICY_TOKENS,
        }
        reviewer_raw = review.sha256_file(review.REVIEWER_SOURCE)
        reviewer_normalized = review.normalized_source_sha256()
        self.assertEqual(reviewer_normalized, review.EXPECTED_NORMALIZED_SOURCE_SHA256)
        reviewer = review.audit_static_source(
            review.REVIEWER_SOURCE,
            policy,
            expected_raw=reviewer_raw,
            expected_normalized=reviewer_normalized,
            reviewer=True,
        )
        publisher_raw = review.sha256_file(review.PUBLISHER_SOURCE)
        publisher = review.audit_static_source(
            review.PUBLISHER_SOURCE,
            policy,
            expected_raw=publisher_raw,
            expected_normalized=None,
            reviewer=False,
        )
        self.assertTrue(reviewer["policy_pass"])
        self.assertTrue(publisher["policy_pass"])
        self.assertFalse(review.accelerator_uninitialized() is False)
        tree = ast.parse(review.REVIEWER_SOURCE.read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        self.assertLessEqual(imports, set(review.ALLOWED_TOP_LEVEL_IMPORTS))

    def test_dynamic_import_and_code_execution_evasions_are_rejected(self) -> None:
        cases = {
            "aliased_eval": "import torch\nrunner = eval\nrunner('1 + 1')\n",
            "builtins_getattr": (
                "import torch\n"
                "runner = getattr(__builtins__, '__im' + 'port__')\n"
                "runner('scripts.bad')\n"
            ),
            "dangerous_subscript": (
                "import torch\n"
                "runner = {'eval': lambda value: value}['eval']\n"
                "runner('payload')\n"
            ),
            "nested_dynamic_call": (
                "import torch\ngetattr(torch, '__im' + 'port__')('scripts.bad')\n"
            ),
            "aliased_getattr": (
                "import os\nimport torch\n"
                "lookup = getattr\n"
                "runner = lookup(os, 'sys' + 'tem')\n"
                "runner('true')\n"
            ),
            "os_namespace_lookup": (
                "import os\nimport torch\n"
                "namespace = os.__dict__\n"
                "runner = namespace['system']\n"
                "runner('true')\n"
            ),
            "ctypes_pythonapi": (
                "import ctypes\nimport torch\n"
                "runner = ctypes.pythonapi.PyRun_SimpleString\n"
                "runner(b'pass')\n"
            ),
            "from_os_system": (
                "from os import system as runner\nimport torch\nrunner('true')\n"
            ),
            "from_os_popen": (
                "from os import popen as runner\nimport torch\nrunner('true')\n"
            ),
            "from_ctypes_pythonapi": (
                "from ctypes import pythonapi as api\n"
                "import torch\n"
                "runner = api.PyRun_SimpleString\n"
                "runner(b'pass')\n"
            ),
            "ctypes_cdll_subscript": (
                "import ctypes\nimport torch\n"
                "library = ctypes.CDLL(None)\n"
                "symbol = 'sys' + 'tem'\n"
                "runner = library[symbol]\n"
                "runner(b'true')\n"
            ),
            "aliased_os_module": (
                "import os\nimport torch\n"
                "module = os\n"
                "runner = module.execv\n"
                "runner('/bin/true', ['true'])\n"
            ),
            "sys_modules_lookup": (
                "import sys\nimport torch\n"
                "module = sys.modules['os']\n"
                "runner = module.system\n"
                "runner('true')\n"
            ),
            "sys_modules_builtin_import": (
                "import sys\nimport torch\n"
                "loader = sys.modules['builtins'].__import__\n"
                "loader('scripts.bad')\n"
            ),
            "sys_modules_builtin_eval": (
                "import sys\nimport torch\n"
                "runner = sys.modules['builtins'].eval\n"
                "runner('1 + 1')\n"
            ),
            "sys_modules_get_builtin_exec": (
                "import sys\nimport torch\n"
                "runner = sys.modules.get('builtins').exec\n"
                "runner('value = 1')\n"
            ),
            "platform_os_exec": (
                "import platform\nimport torch\n"
                "module = platform.os\n"
                "runner = module.execv\n"
                "runner('/bin/true', ['true'])\n"
            ),
            "ast_sys_exec": (
                "import ast\nimport torch\n"
                "module = ast.sys.modules.get('os')\n"
                "runner = module.execv\n"
                "runner('/bin/true', ['true'])\n"
            ),
            "unsafe_direct_torch_load": (
                "import torch\n"
                "value = torch.load(\n"
                "    'payload.pt', map_location='cpu', weights_only=False\n"
                ")\n"
            ),
            "unsafe_aliased_torch_load": (
                "import torch\n"
                "runner = torch.load\n"
                "value = runner('payload.pt', map_location='cpu', weights_only=False)\n"
            ),
            "path_write_text": (
                "from pathlib import Path\n"
                "import torch\n"
                "Path('/tmp/protected').write_text('tamper')\n"
            ),
            "builtin_open_write": (
                "import torch\nhandle = open('/tmp/protected', 'w')\n"
            ),
            "raw_os_write": (
                "import os\n"
                "import torch\n"
                "fd = os.open('/tmp/protected', os.O_WRONLY)\n"
                "os.write(fd, b'tamper')\n"
                "os.close(fd)\n"
            ),
            "aliased_backward": (
                "import torch\n"
                "x = torch.tensor([1.0], requires_grad=True)\n"
                "runner = x.backward\n"
                "runner()\n"
            ),
            "cuda_tensor": ("import torch\nx = torch.tensor([1.0], device='cuda')\n"),
        }
        policy = {
            "allowed_top_level_imports": review.ALLOWED_TOP_LEVEL_IMPORTS,
            "forbidden_tokens": review.FORBIDDEN_POLICY_TOKENS,
        }
        for label, source in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / "candidate.py"
                path.write_text(source, encoding="utf-8")
                with self.assertRaises(review.SalvageReviewError):
                    review.audit_static_source(
                        path,
                        policy,
                        expected_raw=review.sha256_file(path, root=root),
                        expected_normalized=None,
                        root=root,
                        reviewer=False,
                    )


class FrozenPacketTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = review.audit_failed_packet(verify_large_inputs=False)

    def test_real_failed_packet_closes_independently(self) -> None:
        self.assertTrue(all(self.audit["gates"].values()))
        numeric = self.audit["numeric_audit"]
        self.assertEqual(numeric["spectrum_bitwise_equal_count"], 512)
        self.assertEqual(numeric["spectrum_max_abs_error"], 0.0)
        self.assertEqual(
            numeric["failed_scientific_success_gates"], review.FAILED_SCIENTIFIC_GATES
        )
        self.assertAlmostEqual(
            numeric["matrix_from_hessian_max_abs_error"],
            3.885780586188048e-16,
        )

    def test_real_failed_tree_is_read_only(self) -> None:
        self.assertEqual(self.audit["protected_before"], self.audit["protected_after"])
        self.assertEqual(
            self.audit["protected_before"]["failed"]["tree_sha256"],
            review.FAILED_TREE_SHA256,
        )


class CsvAuthorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_rows(
        self, rows: list[list[str]], header: list[str] | None = None
    ) -> Path:
        path = self.root / "values.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header or ["rank", "value"])
            writer.writerows(rows)
        return path

    def records(self) -> list[dict[str, object]]:
        return [
            {"rank": 0, "value": 0.0},
            {"rank": 1, "value": 0.25},
            {"rank": 2, "value": 0.125},
        ]

    def test_rank2_one_ulp_is_rejected(self) -> None:
        changed = math.nextafter(0.125, math.inf)
        path = self.write_rows([["0", "0.0"], ["1", "0.25"], ["2", repr(changed)]])
        with self.assertRaisesRegex(review.SalvageReviewError, "row 2 column value"):
            review.validate_rows_csv(self.records(), path, root=self.root)

    def test_nan_negative_zero_and_duplicate_header_are_rejected(self) -> None:
        cases = [
            ([["0", "nan"], ["1", "0.25"], ["2", "0.125"]], None),
            ([["0", "-0.0"], ["1", "0.25"], ["2", "0.125"]], None),
            (
                [["0", "0.0", "0.0"], ["1", "0.25", "0.25"], ["2", "0.125", "0.125"]],
                ["rank", "value", "value"],
            ),
        ]
        for rows, header in cases:
            with self.subTest(rows=rows, header=header):
                path = self.write_rows(rows, header)
                with self.assertRaises(review.SalvageReviewError):
                    review.validate_rows_csv(self.records(), path, root=self.root)

    def test_duplicate_rank_and_read_only_csv(self) -> None:
        path = self.write_rows([["0", "0.0"], ["1", "0.25"], ["1", "0.125"]])
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        with self.assertRaisesRegex(review.SalvageReviewError, "row 2 column rank"):
            review.validate_rows_csv(self.records(), path, root=self.root)
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        good = self.write_rows([["0", "0.0"], ["1", "0.25"], ["2", "0.125"]])
        good.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        self.assertTrue(
            review.validate_rows_csv(self.records(), good, root=self.root)[
                "binary64_exact"
            ]
        )


class FreezeTamperTest(unittest.TestCase):
    def payload(self) -> dict[str, object]:
        digest = "0" * 64
        return {
            "schema_version": 1,
            "protocol_id": review.PROTOCOL_ID,
            "frozen_at_utc": "2026-07-16T09:00:00.000000Z",
            "repository_root": str(review.ROOT),
            "protocol_sha256": review.PROTOCOL_SHA256,
            "failed_manifest_sha256": review.FAILED_MANIFEST_SHA256,
            "derivation_sha256": review.DERIVATION_SHA256,
            "publisher": {
                "path": str(review.PUBLISHER_SOURCE),
                "raw_sha256": digest,
                "normalized_sha256": digest,
            },
            "reviewer": {
                "path": str(review.REVIEWER_SOURCE),
                "raw_sha256": digest,
                "normalized_sha256": digest,
            },
            "publisher_test": {"path": str(review.PUBLISHER_TEST), "sha256": digest},
            "reviewer_test": {"path": str(review.REVIEWER_TEST), "sha256": digest},
            "runtime": review.EXPECTED_RUNTIME,
            "import_policy": {
                "allowed_top_level_imports": review.ALLOWED_TOP_LEVEL_IMPORTS,
                "forbidden_tokens": review.FORBIDDEN_POLICY_TOKENS,
            },
            "paths": review.FREEZE_PATHS,
            "prereview": {"provenance_decision": "GO", "scientific_decision": "GO"},
        }

    def test_exact_synthetic_freeze(self) -> None:
        self.assertTrue(
            review.validate_execution_freeze(self.payload(), verify_live_files=False)[
                "exact"
            ]
        )

    def test_schema_path_gate_and_hash_tamper(self) -> None:
        mutators = [
            lambda value: value.__setitem__("extra", True),
            lambda value: value["paths"].__setitem__("target", "/tmp/wrong"),
            lambda value: value["prereview"].__setitem__(
                "scientific_decision", "NO-GO"
            ),
            lambda value: value.__setitem__("protocol_sha256", "1" * 64),
        ]
        for mutate in mutators:
            payload = self.payload()
            payload["paths"] = dict(payload["paths"])
            payload["prereview"] = dict(payload["prereview"])
            mutate(payload)
            with self.assertRaises(review.SalvageReviewError):
                review.validate_execution_freeze(payload, verify_live_files=False)


class ExecutionRecordValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = review.validate_frozen_documents()

    @staticmethod
    def independent_tree(
        path: Path,
        entries: list[dict[str, object]],
        *,
        tree_sha256: str,
        device: int,
        first_inode: int,
    ) -> dict[str, object]:
        files: dict[str, dict[str, object]] = {}
        for offset, item in enumerate(entries):
            files[str(item["name"])] = {
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
                "device": device,
                "inode": first_inode + offset,
                "link_count": 1,
                "mode": 0o444,
                "mtime_ns": 1,
            }
        return {
            "path": str(path),
            "file_count": len(entries),
            "total_size_bytes": sum(int(item["size_bytes"]) for item in entries),
            "tree_sha256": tree_sha256,
            "files": files,
        }

    @staticmethod
    def historical_tree(
        path: Path,
        entries: list[dict[str, object]],
        destination: dict[str, dict[str, object]],
        *,
        tree_sha256: str,
    ) -> dict[str, object]:
        files = []
        for item in sorted(entries, key=lambda value: os.fsencode(str(value["name"]))):
            metadata = destination[str(item["name"])]
            files.append(
                {
                    "device": metadata["device"],
                    "inode": metadata["inode"],
                    "name": item["name"],
                    "nlink": 1,
                    "sha256": item["sha256"],
                    "size_bytes": item["size_bytes"],
                }
            )
        return {
            "entry_count": len(entries),
            "files": files,
            "path": str(path),
            "regular_file_count": len(entries),
            "total_size_bytes": sum(int(item["size_bytes"]) for item in entries),
            "tree_sha256": tree_sha256,
        }

    def fixture(self) -> tuple[dict[str, object], dict[str, object]]:
        digest = "a" * 64
        failed_entries = self.documents["failed_entries"]
        staged = [item for item in failed_entries if item["name"] != "failure.json"]
        published = [item for item in staged if item["name"] != "INCOMPLETE"]
        failed_tree = self.independent_tree(
            review.FAILED_SOURCE,
            failed_entries,
            tree_sha256=review.FAILED_TREE_SHA256,
            device=11,
            first_inode=1000,
        )
        target_tree = self.independent_tree(
            review.TARGET,
            published,
            tree_sha256=review.PUBLISHED_TREE_SHA256,
            device=12,
            first_inode=2000,
        )
        original_entries = [{"name": "origin", "size_bytes": 1, "sha256": "b" * 64}]
        startup_entries = [{"name": "startup", "size_bytes": 1, "sha256": "c" * 64}]
        original_tree = self.independent_tree(
            review.ORIGINAL_SOURCE,
            original_entries,
            tree_sha256=review.ORIGINAL_TREE_SHA256,
            device=13,
            first_inode=3000,
        )
        startup_tree = self.independent_tree(
            review.STARTUP_FAILURE,
            startup_entries,
            tree_sha256=review.STARTUP_TREE_SHA256,
            device=14,
            first_inode=4000,
        )
        protected = {
            "failed": failed_tree,
            "original_source": original_tree,
            "startup_failure": startup_tree,
        }
        failed_audit = {
            "protected_before": protected,
            "protected_after": copy.deepcopy(protected),
            "numeric_audit": numeric_audit(),
        }
        freeze = {
            "publisher": {
                "path": str(review.PUBLISHER_SOURCE),
                "raw_sha256": digest,
                "normalized_sha256": digest,
            },
            "reviewer": {
                "path": str(review.REVIEWER_SOURCE),
                "raw_sha256": digest,
                "normalized_sha256": digest,
            },
            "publisher_test": {"path": str(review.PUBLISHER_TEST), "sha256": digest},
            "reviewer_test": {"path": str(review.REVIEWER_TEST), "sha256": digest},
        }
        source_projection = review._publisher_tree_projection(
            failed_tree, expected_path=review.FAILED_SOURCE
        )
        original_projection = review._publisher_tree_projection(
            original_tree, expected_path=review.ORIGINAL_SOURCE
        )
        startup_projection = review._publisher_tree_projection(
            startup_tree, expected_path=review.STARTUP_FAILURE
        )
        destination = {
            name: {
                "device": target_tree["files"][name]["device"],
                "inode": target_tree["files"][name]["inode"],
            }
            for name in target_tree["files"]
        }
        destination["INCOMPLETE"] = {"device": 12, "inode": 2999}
        copied_audit = self.historical_tree(
            review.STAGE,
            staged,
            destination,
            tree_sha256=review.STAGED_TREE_SHA256,
        )
        ready_audit = self.historical_tree(
            review.STAGE,
            published,
            destination,
            tree_sha256=review.PUBLISHED_TREE_SHA256,
        )
        source_files = source_projection["files"]
        source_by_name = {item["name"]: item for item in source_files}
        per_file = []
        for item in staged:
            name = str(item["name"])
            per_file.append(
                {
                    "destination_device": destination[name]["device"],
                    "destination_inode": destination[name]["inode"],
                    "destination_nlink": 1,
                    "name": name,
                    "sha256": item["sha256"],
                    "size_bytes": item["size_bytes"],
                    "source_device": source_by_name[name]["device"],
                    "source_inode": source_by_name[name]["inode"],
                    "source_nlink": 1,
                }
            )
        current_copy = [item for item in per_file if item["name"] != "INCOMPLETE"]
        frozen_documents = {
            "derivation": {
                "path": str(review.DERIVATION),
                "sha256": review.DERIVATION_SHA256,
            },
            "failed_manifest": {
                "path": str(review.FAILED_MANIFEST),
                "sha256": review.FAILED_MANIFEST_SHA256,
            },
            "protocol": {
                "path": str(review.PROTOCOL),
                "sha256": review.PROTOCOL_SHA256,
            },
        }
        record = {
            "schema_version": 1,
            "protocol_id": review.PROTOCOL_ID,
            "status": "validated_ready_for_atomic_publish",
            "created_at_utc": "2026-07-16T09:00:00.000000Z",
            "repository_root": str(review.ROOT),
            "protocol_sha256": review.PROTOCOL_SHA256,
            "failed_manifest_sha256": review.FAILED_MANIFEST_SHA256,
            "derivation_sha256": review.DERIVATION_SHA256,
            "execution_freeze_sha256": digest,
            "publisher_source_sha256": digest,
            "publisher_normalized_source_sha256": digest,
            "publisher_snapshot_sha256": digest,
            "reviewer_source_sha256": digest,
            "publisher_test_sha256": digest,
            "reviewer_test_sha256": digest,
            "source": {
                "path": str(review.FAILED_SOURCE),
                "pre_copy_audit": source_projection,
                "post_copy_audit": copy.deepcopy(source_projection),
                "prepublication_audit": copy.deepcopy(source_projection),
            },
            "staging": {
                "path": str(review.STAGE),
                "copied_entry_count": 33,
                "copied_total_size_bytes": 114_949_714,
                "copied_tree_sha256": review.STAGED_TREE_SHA256,
                "copied_audit": copied_audit,
                "incomplete_removed": True,
                "ready_entry_count": 32,
                "ready_total_size_bytes": 114_949_656,
                "ready_tree_sha256": review.PUBLISHED_TREE_SHA256,
                "ready_audit": ready_audit,
                "fsync_complete": True,
            },
            "publication": {
                "target": str(review.TARGET),
                "target_absent_pre_copy": True,
                "target_absent_prepublication": True,
                "method": "renameat2(RENAME_NOREPLACE)",
                "rename_flags": 1,
                "expected_entry_count": 32,
                "expected_total_size_bytes": 114_949_656,
                "expected_tree_sha256": review.PUBLISHED_TREE_SHA256,
            },
            "copy_audit": {
                "copied_names": [item["name"] for item in staged],
                "distinct_inode_count": 33,
                "excluded_names": ["failure.json"],
                "hardlink_pair_count": 0,
                "per_file": per_file,
                "removed_after_validation": ["INCOMPLETE"],
                "source_destination_sha256_equal": True,
                "source_destination_size_equal": True,
            },
            "protected_lineage": {
                "failed_manifest_path": str(review.FAILED_MANIFEST),
                "original_source_pre_copy": original_projection,
                "original_source_prepublication": copy.deepcopy(original_projection),
                "startup_failure_pre_copy": startup_projection,
                "startup_failure_prepublication": copy.deepcopy(startup_projection),
                "frozen_documents": frozen_documents,
                "frozen_sources": copy.deepcopy(freeze),
                "execution_freeze_path": str(review.EXECUTION_FREEZE),
                "publisher_snapshot_path": str(review.PUBLISHER_SNAPSHOT),
            },
            "scientific_execution": copy.deepcopy(review.SCIENTIFIC_EXECUTION_ZERO),
            "numeric_audit": numeric_audit(),
            "acceptance_gates": {name: True for name in review.PUBLISHER_GATES},
        }
        context = {
            "freeze": freeze,
            "freeze_sha256": digest,
            "documents": self.documents,
            "failed_audit": failed_audit,
            "target_tree_audit": target_tree,
            "current_copy_evidence": current_copy,
            "publisher_snapshot_sha256": digest,
            "publisher_snapshot_matches_source": True,
        }
        return record, context

    def validate(
        self, record: dict[str, object], context: dict[str, object]
    ) -> dict[str, object]:
        return review.validate_execution_record_payload(record, **context)

    def test_exact_execution_record_closes(self) -> None:
        record, context = self.fixture()
        result = self.validate(record, context)
        self.assertTrue(result["valid"])
        self.assertEqual(result["gate_count"], 29)
        self.assertTrue(result["scientific_execution_absent"])

    def test_malformed_execution_record_is_rejected(self) -> None:
        record, context = self.fixture()
        variants = []
        extra = copy.deepcopy(record)
        extra["extra"] = True
        variants.append(extra)
        counter = copy.deepcopy(record)
        counter["scientific_execution"]["proposals"] = 1
        variants.append(counter)
        false_gate = copy.deepcopy(record)
        false_gate["acceptance_gates"]["byte_copy_exact"] = False
        variants.append(false_gate)
        bad_inode = copy.deepcopy(record)
        bad_inode["copy_audit"]["per_file"][0]["destination_inode"] += 1
        variants.append(bad_inode)
        bad_source = copy.deepcopy(record)
        bad_source["source"]["post_copy_audit"]["tree_sha256"] = "f" * 64
        variants.append(bad_source)
        for changed in variants:
            with self.subTest(keys=set(changed)):
                with self.assertRaises(review.SalvageReviewError):
                    self.validate(changed, context)
        duplicate_context = copy.deepcopy(context)
        duplicate_context["current_copy_evidence"][-1] = copy.deepcopy(
            duplicate_context["current_copy_evidence"][0]
        )
        with self.assertRaisesRegex(
            review.SalvageReviewError, "current copy evidence filename set"
        ):
            self.validate(record, duplicate_context)


class GeometryTamperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def fixture(self) -> tuple[dict[str, object], list[float], dict[str, object], Path]:
        eig = torch.tensor([0.01, 0.04, 1.0], dtype=torch.float64)
        hessian = torch.diag(eig.sqrt())
        matrix = hessian @ hessian.T
        basis = torch.eye(3, dtype=torch.float64)[:, :2]
        projector = basis @ basis.T
        metrics = review._geometry_metric_closure(hessian, matrix, eig, basis)
        metrics.update(
            {
                "task_loss": 1.0,
                "hessian_sec": 0.1,
                "low_basis_hash": review.tensor_bytes_sha256(projector),
            }
        )
        payload: dict[str, object] = {
            "hessian": hessian,
            "matrix": matrix,
            "eig": eig,
            "current_low_basis": basis,
            "current_low_projector": projector,
            "metrics": metrics,
        }
        payload["tensor_fingerprints"] = {
            name: review.tensor_fingerprint(payload[name])
            for name in (
                "hessian",
                "matrix",
                "eig",
                "current_low_basis",
                "current_low_projector",
            )
        }
        final = dict(metrics)
        authoritative = [float(value) for value in eig]
        replay = self.root / "replay.csv"
        with replay.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank", "stored", "replayed", "abs_error"])
            for rank, value in enumerate(authoritative):
                writer.writerow([rank, repr(value), repr(value), "0.0"])
        return payload, authoritative, final, replay

    def test_valid_synthetic_geometry_and_coordinated_tamper(self) -> None:
        payload, authoritative, final, replay = self.fixture()
        result = review.audit_geometry_payload(
            payload,
            authoritative,
            final,
            replay,
            root=self.root,
            strict_identity=False,
        )
        self.assertEqual(result["spectrum_max_abs_error"], 0.0)
        changed = torch.tensor(
            [0.01, math.nextafter(0.04, math.inf), 1.0], dtype=torch.float64
        )
        payload["eig"] = changed
        payload["hessian"] = torch.diag(changed.sqrt())
        payload["matrix"] = payload["hessian"] @ payload["hessian"].T
        payload["tensor_fingerprints"] = {
            name: review.tensor_fingerprint(payload[name])
            for name in (
                "hessian",
                "matrix",
                "eig",
                "current_low_basis",
                "current_low_projector",
            )
        }
        payload["metrics"] = review._geometry_metric_closure(
            payload["hessian"],
            payload["matrix"],
            payload["eig"],
            payload["current_low_basis"],
        )
        payload["metrics"].update(
            {
                "task_loss": 1.0,
                "hessian_sec": 0.1,
                "low_basis_hash": review.tensor_bytes_sha256(
                    payload["current_low_projector"]
                ),
            }
        )
        with self.assertRaisesRegex(review.SalvageReviewError, "spectrum bit mismatch"):
            review.audit_geometry_payload(
                payload,
                authoritative,
                payload["metrics"],
                replay,
                root=self.root,
                strict_identity=False,
            )


class FileTypeTest(unittest.TestCase):
    def test_symlink_and_hardlink_are_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            original = source / "a"
            original.write_bytes(b"a")
            os.link(original, source / "b")
            tree = review.audit_tree(source, root=root)
            self.assertEqual(tree["files"]["a"]["link_count"], 2)
            (root / "linked").symlink_to(source)
            with self.assertRaisesRegex(review.SalvageReviewError, "symlink"):
                review.audit_tree(root / "linked", root=root)


class ReviewRecordTest(unittest.TestCase):
    def record(self) -> dict[str, object]:
        digest = "a" * 64
        return {
            "schema_version": 1,
            "review_protocol_id": review.REVIEW_PROTOCOL_ID,
            "protocol_id": review.PROTOCOL_ID,
            "status": "valid_independent_salvage_review",
            "created_at_utc": "2026-07-16T09:00:00.000000Z",
            "repository_root": str(review.ROOT),
            "protocol_sha256": review.PROTOCOL_SHA256,
            "failed_manifest_sha256": review.FAILED_MANIFEST_SHA256,
            "derivation_sha256": review.DERIVATION_SHA256,
            "execution_freeze_sha256": digest,
            "publisher_source_sha256": digest,
            "publisher_normalized_source_sha256": digest,
            "publisher_snapshot_sha256": digest,
            "reviewer_source_sha256": digest,
            "reviewer_normalized_source_sha256": digest,
            "publisher_test_sha256": digest,
            "reviewer_test_sha256": digest,
            "execution_record_sha256": digest,
            "failed_tree_sha256": review.FAILED_TREE_SHA256,
            "published_tree_sha256": review.PUBLISHED_TREE_SHA256,
            "reviewed_output": str(review.TARGET),
            "source_staging": str(review.FAILED_SOURCE),
            "valid": True,
            "recovery_valid": True,
            "scientific_success": False,
            "failed_scientific_success_gates": review.FAILED_SCIENTIFIC_GATES,
            "read_only": True,
            "acceptance_gates": {name: True for name in review.REVIEW_GATES},
            "numeric_audit": {
                "absolute_tolerance": 1e-9,
                "spectrum_max_abs_error": 0.0,
            },
            "byte_copy_audit": {"file_count": 32, "all_bytes_identical": True},
            "protected_tree_audits": {"unchanged": True},
            "errors": [],
            "limitations": ["No new scientific replay was performed."],
        }

    def test_exact_schema_and_tamper_rejection(self) -> None:
        record = self.record()
        self.assertTrue(review.validate_review_record(record)["valid"])
        for key, value in (
            ("extra", True),
            ("reviewed_output", "/tmp/wrong"),
            ("execution_record_sha256", "bad"),
        ):
            changed = dict(record)
            changed[key] = value
            with self.assertRaises(review.SalvageReviewError):
                review.validate_review_record(changed)
        changed = dict(record)
        changed["acceptance_gates"] = dict(record["acceptance_gates"])
        changed["acceptance_gates"]["geometry_closure"] = False
        with self.assertRaises(review.SalvageReviewError):
            review.validate_review_record(changed)

    def test_atomic_crash_resume_and_no_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pending = root / ".review.incomplete"
            final = root / "review.json"

            def crash(_source: Path, _target: Path) -> None:
                raise OSError("injected crash")

            with self.assertRaisesRegex(OSError, "injected crash"):
                review.publish_review_record_atomic(
                    pending,
                    final,
                    self.record(),
                    filesystem_root=root,
                    rename_operation=crash,
                )
            self.assertTrue(pending.is_file())
            self.assertFalse(final.exists())
            result = review.publish_review_record_atomic(
                pending,
                final,
                self.record(),
                filesystem_root=root,
            )
            self.assertEqual(result, "published")
            self.assertFalse(pending.exists())
            before = final.read_bytes()
            self.assertEqual(
                review.publish_review_record_atomic(
                    pending,
                    final,
                    self.record(),
                    filesystem_root=root,
                ),
                "existing_valid",
            )
            self.assertEqual(final.read_bytes(), before)
            pending.write_bytes(before)
            with self.assertRaises(review.SalvageReviewError):
                review.publish_review_record_atomic(
                    pending,
                    final,
                    self.record(),
                    filesystem_root=root,
                )


class ProductionOrchestrationTest(unittest.TestCase):
    def test_mocked_read_only_audit_publishes_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pending = root / ".review.incomplete"
            final = root / "review.json"
            prewrite_calls: list[str] = []

            def collect() -> dict[str, object]:
                return mocked_review_evidence()

            def revalidate(_evidence: dict[str, object]) -> None:
                prewrite_calls.append("checked")

            first = review.review_salvage(
                evidence_collector=collect,
                prewrite_validator=revalidate,
                pending=pending,
                final=final,
                filesystem_root=root,
            )
            self.assertEqual(first["status"], "published")
            self.assertTrue(final.is_file())
            self.assertFalse(pending.exists())
            self.assertEqual(stat.S_IMODE(final.stat().st_mode), 0o444)
            first_bytes = final.read_bytes()
            second = review.review_salvage(
                evidence_collector=collect,
                prewrite_validator=revalidate,
                pending=pending,
                final=final,
                filesystem_root=root,
            )
            self.assertEqual(second["status"], "existing_valid")
            self.assertEqual(final.read_bytes(), first_bytes)
            self.assertEqual(prewrite_calls, ["checked"] * 4)

    def test_source_ends_in_no_argument_production_main(self) -> None:
        source = review.REVIEWER_SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        main_functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        ]
        self.assertEqual(len(main_functions), 1)
        calls = {
            review._call_name(node)
            for node in ast.walk(main_functions[0])
            if isinstance(node, ast.Call)
        }
        self.assertIn("review_salvage", calls)
        self.assertIn("sys.argv", source)
        self.assertTrue(
            source.rstrip().endswith('if __name__ == "__main__":\n    main()')
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
