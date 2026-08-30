from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
MANIFEST = ROOT / "projects" / "manifest.json"


class RepositoryLayoutTest(unittest.TestCase):
    def test_organized_views_resolve_to_existing_original_paths(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["layout"], "research_domain")
        self.assertEqual(
            set(payload["domains"]),
            {"weight-vae", "cfm", "lsdl", "unclassified", "shared"},
        )
        seen_views: set[str] = set()
        for entry in payload["entries"]:
            with self.subTest(view=entry["view"]):
                view = ROOT / entry["view"]

                self.assertNotIn(entry["view"], seen_views)
                seen_views.add(entry["view"])
                self.assertTrue(view.exists(), f"missing project view: {view}")

                if "target" in entry:
                    target = ROOT / entry["target"]
                    self.assertTrue(target.exists(), f"missing source: {target}")
                    self.assertFalse(target.is_symlink(), f"source became a link: {target}")
                    self.assertTrue(view.is_symlink(), f"view is not a link: {view}")
                    self.assertEqual(view.resolve(), target.resolve())
                elif "compatibility_alias" in entry:
                    alias = ROOT / entry["compatibility_alias"]
                    self.assertFalse(view.is_symlink(), f"project owner is a link: {view}")
                    self.assertTrue(alias.is_symlink(), f"alias is not a link: {alias}")
                    self.assertTrue(alias.exists(), f"dangling alias: {alias}")
                    self.assertEqual(alias.resolve(), view.resolve())
                else:
                    self.assertFalse(view.is_symlink(), f"project owner is a link: {view}")

    def test_obsolete_file_type_categories_are_absent(self) -> None:
        obsolete = {"active", "archive", "external", "papers", "research", "scratch"}
        present = {path.name for path in (ROOT / "projects").iterdir() if path.is_dir()}
        self.assertTrue(obsolete.isdisjoint(present))
        self.assertEqual(
            present,
            {"weight-vae", "cfm", "lsdl", "unclassified", "shared"},
        )

    def test_root_contains_only_repository_entry_points(self) -> None:
        allowed = {".gitignore", "AGENTS.md", "Pipfile", "Pipfile.lock", "README.md"}
        root_files = {path.name for path in ROOT.iterdir() if path.is_file()}
        self.assertEqual(root_files, allowed)


if __name__ == "__main__":
    unittest.main()
