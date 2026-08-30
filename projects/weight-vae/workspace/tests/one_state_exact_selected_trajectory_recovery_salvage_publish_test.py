from __future__ import annotations

import ast
import copy
import errno
import hashlib
import os
import shutil
import stat
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


os.environ["CUDA_VISIBLE_DEVICES"] = ""

from scripts import (  # noqa: E402
    publish_one_state_exact_selected_trajectory_recovery_salvage as publisher,
)


def _write(path: Path, data: bytes, *, mode: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mode is not None:
        path.chmod(mode)
    return path


def _spec(name: str, data: bytes) -> publisher.FileSpec:
    return publisher.FileSpec(name, len(data), hashlib.sha256(data).hexdigest())


def _call_name(node: ast.Call) -> str:
    parts: list[str] = []
    current: ast.AST = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _metadata_snapshot(path: Path) -> tuple[tuple[object, ...], ...]:
    entries: list[tuple[object, ...]] = []
    for entry in sorted(os.scandir(path), key=lambda item: os.fsencode(item.name)):
        metadata = os.lstat(path / entry.name)
        entries.append(
            (
                entry.name,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
            )
        )
    return tuple(entries)


class StaticPolicyTest(unittest.TestCase):
    def test_publisher_is_standalone_cpu_only_source(self) -> None:
        source_path = publisher.PRODUCTION_ROOT / publisher.PUBLISHER_SOURCE_REL
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        imports: set[str] = set()
        calls: list[ast.Call] = []
        string_nodes: list[ast.Constant] = []
        assignments: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                if node.module:
                    imports.add(node.module.split(".", 1)[0])
            elif isinstance(node, ast.Call):
                calls.append(node)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                string_nodes.append(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        assignments.add(target.id)

        self.assertIn("torch", imports)
        self.assertLessEqual(imports, set(publisher.ALLOWED_TOP_LEVEL_IMPORTS))
        self.assertNotIn("EXECUTION_FREEZE_SHA256", assignments)
        self.assertLess(
            source.index('os.environ["CUDA_VISIBLE_DEVICES"] = ""'),
            source.index("import torch"),
        )

        call_names = {_call_name(node) for node in calls}
        disallowed_calls = {
            "__import__",
            "compile",
            "eval",
            "exec",
            "os.popen",
            "os.remove",
            "os.rename",
            "os.replace",
            "os.system",
            "torch.save",
        }
        self.assertTrue(disallowed_calls.isdisjoint(call_names))
        self.assertFalse(
            any(
                name.startswith(("torch.autograd", "torch.cuda", "torch.optim"))
                or name.endswith((".backward", ".grad"))
                for name in call_names
            )
        )
        torch_loads = [node for node in calls if _call_name(node) == "torch.load"]
        self.assertEqual(len(torch_loads), 1)
        keywords = {keyword.arg: keyword.value for keyword in torch_loads[0].keywords}
        self.assertEqual(ast.literal_eval(keywords["map_location"]), "cpu")
        self.assertIs(ast.literal_eval(keywords["weights_only"]), True)

        scan_lines = [list(line) for line in source.splitlines(keepends=True)]
        for node in string_nodes:
            end_line = node.end_lineno or node.lineno
            end_column = node.end_col_offset or node.col_offset
            for line_number in range(node.lineno, end_line + 1):
                line = scan_lines[line_number - 1]
                start = node.col_offset if line_number == node.lineno else 0
                stop = end_column if line_number == end_line else len(line)
                for position in range(start, min(stop, len(line))):
                    if line[position] not in {"\n", "\r"}:
                        line[position] = " "
        policy_source = "".join("".join(line) for line in scan_lines)
        for token in publisher.FORBIDDEN_TOKENS:
            self.assertNotIn(token, policy_source)

        self.assertNotIn("numpy", imports)
        self.assertNotIn("pandas", imports)
        self.assertNotIn("scripts", imports)
        self.assertNotIn("post_train_research", imports)

    def test_production_entry_has_no_overrides(self) -> None:
        main = next(
            node
            for node in ast.parse(
                (publisher.PRODUCTION_ROOT / publisher.PUBLISHER_SOURCE_REL).read_text(
                    encoding="utf-8"
                )
            ).body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        self.assertEqual(len(main.args.args), 0)
        self.assertIsNone(main.args.vararg)
        self.assertIsNone(main.args.kwarg)
        self.assertEqual(publisher.PRODUCTION_LAYOUT.root, Path("/home/coder/project"))


class RealFailedPacketPreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        layout = publisher.PRODUCTION_LAYOUT
        cls.publisher_before = publisher._read_bytes(publisher.LIVE_SOURCE)
        cls.protected_before = {
            "failed": _metadata_snapshot(layout.failed),
            "original": _metadata_snapshot(layout.original_source),
            "startup": _metadata_snapshot(layout.startup_failed),
        }
        cls.cuda_before = publisher._cuda_uninitialized()
        cls.audit = publisher.audit_failed_packet_read_only()
        cls.cuda_after = publisher._cuda_uninitialized()
        cls.protected_after = {
            "failed": _metadata_snapshot(layout.failed),
            "original": _metadata_snapshot(layout.original_source),
            "startup": _metadata_snapshot(layout.startup_failed),
        }
        cls.publisher_after = publisher._read_bytes(publisher.LIVE_SOURCE)

    def test_real_failed_packet_read_only_preflight(self) -> None:
        expected_gates = {
            "failure_and_run_log_exact",
            "packet_hash_graph_exact",
            "execution_budget_exact",
            "checkpoint_lineage_exact",
            "task_and_input_provenance_exact",
            "model_nonmutation_exact",
            "geometry_shapes_dtypes_finite",
            "matrix_from_hessian_closure",
            "eigensystem_closure",
            "low_basis_projector_closure",
            "metric_closure",
            "spectrum_bitwise_closure",
            "csv_checkpoint_closure",
            "scientific_outcome_unchanged",
            "cuda_uninitialized",
            "scientific_execution_absent",
        }
        self.assertEqual(set(self.audit.gates), expected_gates)
        self.assertTrue(all(self.audit.gates.values()))
        self.assertEqual(
            self.audit.details["artifact_link_count"],
            len(publisher.EXPECTED_MANIFESTED_ARTIFACTS),
        )
        self.assertEqual(
            self.audit.details["checkpoint"],
            {
                "active_parameter_count": publisher.ACTIVE_PARAMETER_COUNT,
                "active_tensor_count": publisher.ACTIVE_TENSOR_COUNT,
            },
        )
        self.assertEqual(self.protected_before, self.protected_after)
        self.assertEqual(self.publisher_before, self.publisher_after)

    def test_geometry_checkpoint_and_scientific_closures(self) -> None:
        numeric = self.audit.numeric_audit
        publisher._validate_numeric_record(numeric)
        self.assertEqual(numeric["spectrum_bitwise_equal_count"], 512)
        self.assertEqual(numeric["spectrum_representation_count"], 5)
        self.assertEqual(numeric["spectrum_max_abs_error"], 0.0)
        self.assertLessEqual(
            numeric["matrix_from_hessian_max_abs_error"], publisher.REPLAY_ATOL
        )
        self.assertLessEqual(
            numeric["eigvalsh_matrix_max_abs_error"], publisher.REPLAY_ATOL
        )
        self.assertFalse(numeric["scientific_success"])
        self.assertEqual(
            numeric["failed_scientific_success_gates"],
            publisher.EXPECTED_FAILED_SUCCESS_GATES,
        )

    def test_no_cuda_initialization(self) -> None:
        self.assertTrue(self.cuda_before)
        self.assertTrue(self.cuda_after)
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        module = sys.modules.get("torch.cuda")
        self.assertFalse(bool(getattr(module, "_initialized", False)))

    def test_exact_rank_two_parser_regression(self) -> None:
        packet = publisher.PRODUCTION_LAYOUT.failed
        header, rows = publisher._read_csv(packet / "state_spectra.csv")
        self.assertEqual(
            header,
            ["accepted_update", "rank", "m_eigenvalue", "a_contribution", "phase"],
        )
        endpoint = rows[-publisher.DIMENSION :]
        self.assertEqual(len(endpoint), publisher.DIMENSION)
        self.assertEqual(endpoint[2]["accepted_update"], "100")
        self.assertEqual(endpoint[2]["rank"], "2")
        value = float(endpoint[2]["m_eigenvalue"])
        self.assertEqual(struct.pack(">d", value).hex(), "3de22a7f787e6c62")
        self.assertEqual(
            self.audit.numeric_audit["rank_2_binary64_hex"], "3de22a7f787e6c62"
        )

    def test_live_checkpoint_and_geometry_top_level_schemas(self) -> None:
        packet = publisher.PRODUCTION_LAYOUT.failed
        progress = publisher._load_checkpoint(packet / "source_progress_checkpoint.pt")
        final = publisher._load_checkpoint(packet / "final_checkpoint.pt")
        geometry = publisher._load_checkpoint(packet / "replayed_final_geometry.pt")
        self.assertEqual(set(progress), publisher.PROGRESS_KEYS)
        self.assertEqual(set(final), publisher.FINAL_CHECKPOINT_KEYS)
        self.assertEqual(set(geometry), publisher.GEOMETRY_KEYS)
        self.assertEqual(list(geometry["hessian"].shape), [512, 512])
        self.assertEqual(list(geometry["matrix"].shape), [512, 512])
        self.assertEqual(list(geometry["eig"].shape), [512])
        self.assertEqual(list(geometry["current_low_basis"].shape), [512, 451])
        self.assertEqual(list(geometry["current_low_projector"].shape), [512, 512])
        self.assertTrue(publisher._cuda_uninitialized())


class StrictParserAndSchemaTest(unittest.TestCase):
    def test_strict_json_rejects_duplicate_nonfinite_and_nonobject(self) -> None:
        invalid = (
            b'{"a":1,"a":2}',
            b'{"a":NaN}',
            b'{"a":Infinity}',
            b"[]",
            b"\xff",
        )
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(publisher.SalvageError):
                    publisher._strict_json_bytes(payload, "test")

    def test_strict_csv_rejects_duplicate_extra_and_invalid_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid = {
                "duplicate.csv": b"a,a\n1,2\n",
                "extra.csv": b"a\n1,2\n",
                "invalid.csv": b"a\n\xff\n",
            }
            for name, payload in invalid.items():
                with self.subTest(name=name):
                    path = _write(root / name, payload)
                    with self.assertRaises(publisher.SalvageError):
                        publisher._read_csv(path)

    def test_integer_csv_fields_use_exact_decimal_not_binary_float(self) -> None:
        self.assertTrue(publisher._csv_cell_matches(1, "1"))
        self.assertTrue(publisher._csv_cell_matches(1, "1.0"))
        self.assertTrue(publisher._csv_cell_matches(1, "01"))
        self.assertTrue(publisher._csv_cell_matches(True, "True"))
        self.assertFalse(publisher._csv_cell_matches(True, "1"))
        self.assertEqual(publisher._as_csv_int("100", "value"), 100)
        self.assertEqual(publisher._as_csv_int("100.0", "value"), 100)
        self.assertEqual(publisher._as_csv_int("1e2", "value"), 100)
        for value in ("100.5", "-0", "NaN", "Infinity", ""):
            with self.subTest(value=value):
                with self.assertRaises(publisher.SalvageError):
                    publisher._as_csv_int(value, "value")

    def test_manifest_numeric_geometry_and_hash_graph_schema_tampering(self) -> None:
        layout = publisher.PRODUCTION_LAYOUT
        raw_manifest = publisher._load_json(layout.path(publisher.FAILED_MANIFEST_REL))
        for mutation in ("missing", "extra"):
            tampered = copy.deepcopy(raw_manifest)
            if mutation == "missing":
                tampered.pop("schema_version")
            else:
                tampered["unexpected"] = True
            with self.subTest(kind=mutation):
                with self.assertRaises(publisher.SalvageError):
                    publisher._validate_failed_manifest_payload(tampered)

        with self.assertRaises(publisher.SalvageError):
            publisher._validate_geometry_payload({}, {})

        numeric = {
            "absolute_tolerance": publisher.REPLAY_ATOL,
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
            "failed_scientific_success_gates": publisher.EXPECTED_FAILED_SUCCESS_GATES,
            "final_exact_a_per_dim": 0.9321672207645146,
            "continuation_non_top_only_fraction": 0.11301885339451084,
            "total_non_top_only_fraction": 0.07733408144475684,
        }
        publisher._validate_numeric_record(numeric)
        for mutation in ("missing", "extra", "value"):
            tampered_numeric = copy.deepcopy(numeric)
            if mutation == "missing":
                tampered_numeric.pop("rank_2_binary64_hex")
            elif mutation == "extra":
                tampered_numeric["unexpected"] = 0
            else:
                tampered_numeric["spectrum_bitwise_equal_count"] = 511
            with self.subTest(numeric=mutation):
                with self.assertRaises(publisher.SalvageError):
                    publisher._validate_numeric_record(tampered_numeric)

        artifact_manifest = publisher._load_json(
            layout.failed / "artifact_manifest.json"
        )
        artifact_manifest.pop("artifacts")
        with mock.patch.object(publisher, "_load_json", return_value=artifact_manifest):
            with self.assertRaises(publisher.SalvageError):
                publisher._validate_packet_hash_graph(layout.failed)


class FilesystemValidationTest(unittest.TestCase):
    def _valid_tree(self, root: Path) -> tuple[Path, tuple[publisher.FileSpec, ...]]:
        tree = root / "tree"
        tree.mkdir()
        contents = {"a.bin": b"alpha", "b.bin": b"beta"}
        for name, data in contents.items():
            _write(tree / name, data)
        specs = tuple(_spec(name, contents[name]) for name in sorted(contents))
        return tree, specs

    def test_exact_tree_hash_and_regular_file_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree, specs = self._valid_tree(Path(temporary))
            expected = hashlib.sha256(
                b'{"name":"a.bin","sha256":"8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8","size_bytes":5}\n'
                b'{"name":"b.bin","sha256":"f44e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753","size_bytes":4}\n'
            ).hexdigest()
            self.assertEqual(publisher._tree_hash(specs), expected)
            audit = publisher._audit_tree(
                tree,
                specs,
                expected_tree_sha256=expected,
                expected_total_size=9,
                label="test tree",
            )
            self.assertEqual(audit["entry_count"], 2)
            self.assertEqual(audit["tree_sha256"], expected)
            publisher._validate_record_tree_audit(
                audit, specs, path=tree, label="recorded tree"
            )
            tampered = copy.deepcopy(audit)
            tampered["files"][0]["unexpected"] = True
            with self.assertRaises(publisher.SalvageError):
                publisher._validate_record_tree_audit(
                    tampered, specs, path=tree, label="recorded tree"
                )

    def test_tree_rejects_tamper_missing_extra_symlink_fifo_and_hardlink(self) -> None:
        mutations = ("tamper", "missing", "extra", "symlink", "fifo", "hardlink")
        for mutation in mutations:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                tree, specs = self._valid_tree(root)
                if mutation == "tamper":
                    _write(tree / "a.bin", b"changed")
                elif mutation == "missing":
                    os.unlink(tree / "a.bin")
                elif mutation == "extra":
                    _write(tree / "extra.bin", b"extra")
                elif mutation == "symlink":
                    os.unlink(tree / "a.bin")
                    (tree / "a.bin").symlink_to(tree / "b.bin")
                elif mutation == "fifo":
                    os.unlink(tree / "a.bin")
                    os.mkfifo(tree / "a.bin")
                else:
                    os.link(tree / "a.bin", root / "external-hardlink.bin")
                with self.assertRaises(publisher.SalvageError):
                    publisher._audit_tree(
                        tree,
                        specs,
                        expected_tree_sha256=publisher._tree_hash(specs),
                        expected_total_size=9,
                        label="tampered tree",
                    )

    def test_lstat_rejects_symlink_components_and_lexists_dangling_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            _write(real / "file", b"data")
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(publisher.SalvageError):
                publisher._require_regular_file(alias / "file")
            dangling = root / "dangling"
            dangling.symlink_to(root / "missing")
            self.assertTrue(os.path.lexists(dangling))
            with self.assertRaises(publisher.SalvageError):
                publisher._require_absent(dangling, "dangling test")

    def test_byte_copy_is_independent_and_removes_only_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            contents = {"INCOMPLETE": b"pending\n", "payload.bin": b"payload"}
            specs = tuple(_spec(name, data) for name, data in contents.items())
            for name, data in contents.items():
                _write(source / name, data)
            before = _metadata_snapshot(source)
            stage = root / "stage"
            audit = publisher._copy_projection(source, stage, specs)
            self.assertEqual(audit["distinct_inode_count"], 2)
            self.assertEqual(audit["hardlink_pair_count"], 0)
            for item in audit["per_file"]:
                self.assertNotEqual(
                    (item["source_device"], item["source_inode"]),
                    (item["destination_device"], item["destination_inode"]),
                )
                self.assertEqual(item["source_nlink"], 1)
                self.assertEqual(item["destination_nlink"], 1)
                self.assertEqual(
                    (source / item["name"]).read_bytes(),
                    (stage / item["name"]).read_bytes(),
                )
                self.assertEqual(
                    stat.S_IMODE(os.lstat(stage / item["name"]).st_mode) & 0o222,
                    0,
                )
            publisher._remove_copied_incomplete(stage)
            self.assertFalse(os.path.lexists(stage / "INCOMPLETE"))
            self.assertEqual((stage / "payload.bin").read_bytes(), b"payload")
            self.assertEqual(before, _metadata_snapshot(source))

    def test_copy_rejects_hardlinked_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            path = _write(source / "payload", b"payload")
            os.link(path, root / "other-link")
            with self.assertRaises(publisher.SalvageError):
                publisher._copy_projection(
                    source,
                    root / "stage",
                    (_spec("payload", b"payload"),),
                )

    def test_interrupted_copy_is_not_cleaned_or_accepted(self) -> None:
        class InjectedCrash(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = publisher.Layout.for_root(root)
            layout.failed.mkdir(parents=True)
            _write(layout.failed / "a", b"a")
            _write(layout.failed / "INCOMPLETE", b"pending")
            specs = (_spec("a", b"a"), _spec("INCOMPLETE", b"pending"))

            def crash(stage: str) -> None:
                if stage == "after_copy:a":
                    raise InjectedCrash(stage)

            with self.assertRaises(InjectedCrash):
                publisher._copy_projection(
                    layout.failed, layout.stage, specs, fault_hook=crash
                )
            self.assertTrue(os.path.lexists(layout.stage))
            self.assertTrue(os.path.lexists(layout.stage / "a"))
            self.assertFalse(os.path.lexists(layout.record))
            with self.assertRaisesRegex(publisher.SalvageError, "manual quarantine"):
                publisher._classify_crash_state(layout)
            with self.assertRaises(publisher.SalvageError):
                publisher._audit_tree(
                    layout.stage,
                    specs,
                    expected_tree_sha256=publisher._tree_hash(specs),
                    expected_total_size=8,
                    label="interrupted stage",
                )


class SnapshotAndExclusiveFileTest(unittest.TestCase):
    def test_snapshot_is_byte_identical_immutable_and_not_a_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write(root / "source.py", b"print('frozen')\n")
            before = os.lstat(source)
            snapshot = root / "snapshot.py"
            first = publisher._create_or_verify_snapshot(source, snapshot)
            second = publisher._create_or_verify_snapshot(source, snapshot)
            after = os.lstat(source)
            self.assertTrue(first["created"])
            self.assertFalse(second["created"])
            self.assertEqual(source.read_bytes(), snapshot.read_bytes())
            self.assertNotEqual(
                (before.st_dev, before.st_ino),
                (os.lstat(snapshot).st_dev, os.lstat(snapshot).st_ino),
            )
            self.assertEqual(
                (before.st_ino, before.st_size, before.st_mtime_ns),
                (after.st_ino, after.st_size, after.st_mtime_ns),
            )
            self.assertEqual(stat.S_IMODE(os.lstat(snapshot).st_mode) & 0o222, 0)

    def test_snapshot_tamper_and_hardlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write(root / "source", b"source")
            tampered = _write(root / "tampered", b"other", mode=0o444)
            with self.assertRaises(publisher.SalvageError):
                publisher._create_or_verify_snapshot(source, tampered)
            hardlink = root / "hardlink"
            os.link(source, hardlink)
            source.chmod(0o444)
            with self.assertRaises(publisher.SalvageError):
                publisher._create_or_verify_snapshot(source, hardlink)

    def test_exclusive_file_never_replaces_existing_or_dangling_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = root / "record.json"
            publisher._create_exclusive_file(record, b"{}\n", label="record")
            self.assertEqual(record.read_bytes(), b"{}\n")
            self.assertEqual(stat.S_IMODE(os.lstat(record).st_mode) & 0o222, 0)
            with self.assertRaises(publisher.SalvageError):
                publisher._create_exclusive_file(
                    record, b'{"changed":true}\n', label="record"
                )
            self.assertEqual(record.read_bytes(), b"{}\n")
            dangling = root / "dangling"
            dangling.symlink_to(root / "missing")
            with self.assertRaises(publisher.SalvageError):
                publisher._create_exclusive_file(dangling, b"data", label="record")


class RenameAndCrashStateTest(unittest.TestCase):
    def test_real_renameat2_no_replace_success_and_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            _write(source / "payload", b"source")
            publisher._rename_noreplace(source, target)
            self.assertFalse(os.path.lexists(source))
            self.assertEqual((target / "payload").read_bytes(), b"source")

            second_source = root / "second-source"
            second_source.mkdir()
            _write(second_source / "payload", b"second")
            with self.assertRaises(publisher.SalvageError):
                publisher._rename_noreplace(second_source, target)
            self.assertEqual((target / "payload").read_bytes(), b"source")
            self.assertEqual((second_source / "payload").read_bytes(), b"second")

    def test_dangling_target_and_target_appearance_race_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            dangling = root / "dangling"
            dangling.symlink_to(root / "missing")
            with self.assertRaises(publisher.SalvageError):
                publisher._rename_noreplace(source, dangling)
            self.assertTrue(source.is_dir())

            race_target = root / "race-target"

            class FakeRename:
                argtypes: object = None
                restype: object = None

                def __call__(self, *_args: object) -> int:
                    race_target.mkdir()
                    publisher.ctypes.set_errno(errno.EEXIST)
                    return -1

            class FakeLibc:
                renameat2 = FakeRename()

            with mock.patch.object(publisher.ctypes, "CDLL", return_value=FakeLibc()):
                with self.assertRaisesRegex(publisher.SalvageError, "target appeared"):
                    publisher._rename_noreplace(source, race_target)
            self.assertTrue(source.is_dir())
            self.assertTrue(race_target.is_dir())

    def test_all_crash_state_combinations(self) -> None:
        expected = {
            (False, False, False): "fresh",
            (True, False, True): "resume_pending_rename",
            (True, True, False): "already_published",
        }
        for record, target, stage in (
            (False, False, False),
            (False, False, True),
            (False, True, False),
            (False, True, True),
            (True, False, False),
            (True, False, True),
            (True, True, False),
            (True, True, True),
        ):
            with self.subTest(record=record, target=target, stage=stage):
                with tempfile.TemporaryDirectory() as temporary:
                    layout = publisher.Layout.for_root(Path(temporary))
                    layout.record.parent.mkdir(parents=True)
                    layout.target.parent.mkdir(parents=True, exist_ok=True)
                    if record:
                        _write(layout.record, b"record")
                    if target:
                        layout.target.mkdir()
                    if stage:
                        layout.stage.mkdir()
                    combination = (record, target, stage)
                    if combination in expected:
                        self.assertEqual(
                            publisher._classify_crash_state(layout),
                            expected[combination],
                        )
                    else:
                        with self.assertRaisesRegex(
                            publisher.SalvageError, "manual quarantine"
                        ):
                            publisher._classify_crash_state(layout)

    def test_dangling_target_counts_as_existing_crash_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = publisher.Layout.for_root(Path(temporary))
            layout.target.parent.mkdir(parents=True)
            layout.target.symlink_to(layout.target.parent / "missing")
            self.assertTrue(os.path.lexists(layout.target))
            with self.assertRaisesRegex(publisher.SalvageError, "manual quarantine"):
                publisher._classify_crash_state(layout)


class ExecutionFreezeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.layout = publisher.Layout.for_root(self.root)
        copies = {
            publisher.PROTOCOL_REL: publisher.PRODUCTION_ROOT / publisher.PROTOCOL_REL,
            publisher.FAILED_MANIFEST_REL: (
                publisher.PRODUCTION_ROOT / publisher.FAILED_MANIFEST_REL
            ),
            publisher.DERIVATION_REL: (
                publisher.PRODUCTION_ROOT / publisher.DERIVATION_REL
            ),
            publisher.PUBLISHER_SOURCE_REL: publisher.LIVE_SOURCE,
            publisher.REVIEWER_SOURCE_REL: (
                publisher.PRODUCTION_ROOT / publisher.REVIEWER_SOURCE_REL
            ),
            publisher.PUBLISHER_TEST_REL: Path(__file__).resolve(),
            publisher.REVIEWER_TEST_REL: (
                publisher.PRODUCTION_ROOT / publisher.REVIEWER_TEST_REL
            ),
        }
        for relative, source in copies.items():
            destination = self.layout.path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        publisher_path = self.layout.path(publisher.PUBLISHER_SOURCE_REL)
        reviewer_path = self.layout.path(publisher.REVIEWER_SOURCE_REL)
        publisher_test = self.layout.path(publisher.PUBLISHER_TEST_REL)
        reviewer_test = self.layout.path(publisher.REVIEWER_TEST_REL)
        self.payload = {
            "schema_version": 1,
            "protocol_id": publisher.PROTOCOL_ID,
            "frozen_at_utc": "2026-07-16T10:00:00.000000Z",
            "repository_root": str(self.root),
            "protocol_sha256": publisher.PROTOCOL_SHA256,
            "failed_manifest_sha256": publisher.FAILED_MANIFEST_SHA256,
            "derivation_sha256": publisher.DERIVATION_SHA256,
            "publisher": {
                "path": str(publisher_path),
                "raw_sha256": publisher._sha256_file(publisher_path),
                "normalized_sha256": publisher._normalized_source_sha256(
                    publisher_path, require_marker=True
                ),
            },
            "reviewer": {
                "path": str(reviewer_path),
                "raw_sha256": publisher._sha256_file(reviewer_path),
                "normalized_sha256": publisher._normalized_source_sha256(
                    reviewer_path, require_marker=True
                ),
            },
            "publisher_test": {
                "path": str(publisher_test),
                "sha256": publisher._sha256_file(publisher_test),
            },
            "reviewer_test": {
                "path": str(reviewer_test),
                "sha256": publisher._sha256_file(reviewer_test),
            },
            "runtime": dict(publisher.EXPECTED_RUNTIME),
            "import_policy": {
                "allowed_top_level_imports": publisher.ALLOWED_TOP_LEVEL_IMPORTS,
                "forbidden_tokens": publisher.FORBIDDEN_TOKENS,
            },
            "paths": {
                "failed_source": str(self.layout.failed),
                "stage": str(self.layout.stage),
                "target": str(self.layout.target),
                "publisher_snapshot": str(self.layout.snapshot),
                "execution_record": str(self.layout.record),
                "review_record_temp": str(
                    self.layout.path(publisher.INDEPENDENT_REVIEW_TEMP_REL)
                ),
                "review_record": str(
                    self.layout.path(publisher.INDEPENDENT_REVIEW_REL)
                ),
            },
            "prereview": {
                "provenance_decision": "GO",
                "scientific_decision": "GO",
            },
        }
        self._write_freeze(self.payload)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_freeze(self, payload: dict[str, object]) -> None:
        if os.path.lexists(self.layout.freeze):
            self.layout.freeze.chmod(0o644)
        _write(self.layout.freeze, publisher._json_bytes(payload), mode=0o444)

    def test_exact_schema_live_hashes_and_observed_freeze_sha(self) -> None:
        audit = publisher._validate_execution_freeze(
            self.layout,
            publisher_source=self.layout.path(publisher.PUBLISHER_SOURCE_REL),
        )
        self.assertEqual(set(audit["payload"]), publisher.FREEZE_KEYS)
        self.assertEqual(audit["sha256"], publisher._sha256_file(self.layout.freeze))
        self.assertEqual(
            audit["self_freeze"]["normalized_sha256"],
            publisher.EXPECTED_NORMALIZED_SOURCE_SHA256,
        )
        self.assertFalse(hasattr(publisher, "EXECUTION_FREEZE_SHA256"))

    def test_freeze_rejects_missing_extra_tampered_and_live_source_change(self) -> None:
        for mutation in ("missing", "extra", "hash"):
            payload = copy.deepcopy(self.payload)
            if mutation == "missing":
                payload.pop("prereview")
            elif mutation == "extra":
                payload["unexpected"] = True
            else:
                payload["publisher"]["raw_sha256"] = "0" * 64
            self._write_freeze(payload)
            with self.subTest(mutation=mutation):
                with self.assertRaises(publisher.SalvageError):
                    publisher._validate_execution_freeze(
                        self.layout,
                        publisher_source=self.layout.path(
                            publisher.PUBLISHER_SOURCE_REL
                        ),
                    )
        self._write_freeze(self.payload)
        reviewer = self.layout.path(publisher.REVIEWER_SOURCE_REL)
        with reviewer.open("ab") as handle:
            handle.write(b"\n# tampered\n")
        with self.assertRaises(publisher.SalvageError):
            publisher._validate_execution_freeze(
                self.layout,
                publisher_source=self.layout.path(publisher.PUBLISHER_SOURCE_REL),
            )


class RecordSchemaTest(unittest.TestCase):
    def test_record_has_exact_schema_and_observed_freeze_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = publisher.Layout.for_root(Path(temporary))
            payload = {
                "protocol_sha256": publisher.PROTOCOL_SHA256,
                "failed_manifest_sha256": publisher.FAILED_MANIFEST_SHA256,
                "derivation_sha256": publisher.DERIVATION_SHA256,
                "publisher": {
                    "path": str(layout.path(publisher.PUBLISHER_SOURCE_REL)),
                    "raw_sha256": "1" * 64,
                    "normalized_sha256": "2" * 64,
                },
                "reviewer": {
                    "path": str(layout.path(publisher.REVIEWER_SOURCE_REL)),
                    "raw_sha256": "3" * 64,
                    "normalized_sha256": "4" * 64,
                },
                "publisher_test": {
                    "path": str(layout.path(publisher.PUBLISHER_TEST_REL)),
                    "sha256": "5" * 64,
                },
                "reviewer_test": {
                    "path": str(layout.path(publisher.REVIEWER_TEST_REL)),
                    "sha256": "6" * 64,
                },
            }
            freeze = {"payload": payload, "sha256": "7" * 64}
            audit = {"entry_count": 0, "tree_sha256": "8" * 64}
            protected = {
                "failed": audit,
                "original_source": audit,
                "startup_failure": audit,
            }
            manifest = publisher.FailedManifest({}, (), (), (), ())
            scientific = publisher.ScientificAudit(
                gates={},
                numeric_audit={"marker": True},
                details={},
            )
            gates = {name: True for name in publisher.ACCEPTANCE_GATE_NAMES}
            record = publisher._build_execution_record(
                layout,
                manifest,
                freeze,
                {"sha256": "1" * 64},
                protected,
                protected,
                protected,
                audit,
                audit,
                {
                    "copied_names": [],
                    "distinct_inode_count": 0,
                    "excluded_names": ["failure.json"],
                    "hardlink_pair_count": 0,
                    "per_file": [],
                    "removed_after_validation": ["INCOMPLETE"],
                    "source_destination_sha256_equal": True,
                    "source_destination_size_equal": True,
                },
                scientific,
                gates,
            )
            self.assertEqual(set(record), publisher.EXECUTION_RECORD_KEYS)
            self.assertEqual(len(record), 23)
            self.assertEqual(record["status"], "validated_ready_for_atomic_publish")
            self.assertEqual(record["execution_freeze_sha256"], "7" * 64)
            self.assertEqual(record["publisher_source_sha256"], "1" * 64)
            self.assertEqual(record["reviewer_source_sha256"], "3" * 64)
            self.assertEqual(record["publisher_test_sha256"], "5" * 64)
            self.assertEqual(record["reviewer_test_sha256"], "6" * 64)
            self.assertEqual(
                record["protected_lineage"]["frozen_sources"],
                {
                    name: payload[name]
                    for name in (
                        "publisher",
                        "reviewer",
                        "publisher_test",
                        "reviewer_test",
                    )
                },
            )


class FrozenIdentityTest(unittest.TestCase):
    def test_normalized_self_hash_is_frozen_exactly_once(self) -> None:
        source = publisher._read_bytes(publisher.LIVE_SOURCE)
        normalized, masked = publisher._source_normalized_bytes(source)
        self.assertEqual(masked, 1)
        self.assertEqual(
            hashlib.sha256(normalized).hexdigest(),
            publisher.EXPECTED_NORMALIZED_SOURCE_SHA256,
        )
        self.assertRegex(
            publisher.EXPECTED_NORMALIZED_SOURCE_SHA256, r"\A[0-9a-f]{64}\Z"
        )


if __name__ == "__main__":
    unittest.main()
