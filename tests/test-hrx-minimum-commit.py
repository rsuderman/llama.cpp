#!/usr/bin/env python3
"""Exercise the HRX configure gate with local Git histories; no GPU is needed."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


CHECK = Path(__file__).resolve().parents[1] / "ggml/src/ggml-hrx/cmake/MinimumRevision.cmake"


class MinimumRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hrx minimum ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.hrx = self.root / "hrx"
        self.git("init", "--quiet", str(self.hrx))
        self.git("config", "user.name", "HRX test", cwd=self.hrx)
        self.git("config", "user.email", "hrx-test@example.invalid", cwd=self.hrx)
        self.git("config", "commit.gpgsign", "false", cwd=self.hrx)
        self.old = self.commit("old")
        self.minimum = self.commit("minimum")
        self.new = self.commit("new")
        self.project = self.root / "project"
        self.project.mkdir()
        self.pin = self.project / "minimum.txt"
        self.pin.write_text(self.minimum + "\n", encoding="utf-8")
        (self.project / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.14)\n'
            'project(hrx_revision_test NONE)\n'
            f'include("{CHECK.as_posix()}")\n'
            'ggml_hrx_check_minimum_revision("${HRX_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/minimum.txt")\n',
            encoding="utf-8",
        )
        self.build = self.root / "build"

    def git(self, *args, cwd=None):
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()

    def commit(self, name):
        self.git("commit", "--quiet", "--allow-empty", "-m", name, cwd=self.hrx)
        return self.git("rev-parse", "HEAD", cwd=self.hrx)

    def configure(self, source=None, *extra):
        return subprocess.run(
            ["cmake", "-S", str(self.project), "-B", str(self.build),
             f"-DHRX_SOURCE_DIR={source or self.hrx}", *extra],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    def assert_passes(self, result):
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"satisfies minimum {self.minimum}", result.stdout)

    def assert_fails(self, result, reason):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(reason, result.stdout)
        self.assertIn(self.minimum, result.stdout)
        self.assertIn("HRX_SOURCE_DIR:", result.stdout)

    def test_exact_minimum(self):
        self.git("checkout", "--quiet", self.minimum, cwd=self.hrx)
        self.assert_passes(self.configure())

    def test_descendant_with_local_edits(self):
        (self.hrx / "local.txt").write_text("local edit\n", encoding="utf-8")
        self.assert_passes(self.configure())

    def test_older_revision(self):
        self.git("checkout", "--quiet", self.old, cwd=self.hrx)
        result = self.configure()
        self.assert_fails(result, "does not have the required commit")
        self.assertIn(self.old, result.stdout)

    def test_divergent_revision(self):
        self.git("checkout", "--quiet", self.old, cwd=self.hrx)
        divergent = self.commit("divergent")
        result = self.configure()
        self.assert_fails(result, "does not have the required commit")
        self.assertIn(divergent, result.stdout)

    def test_unknown_revision(self):
        self.minimum = "1" * 40
        self.pin.write_text(self.minimum + "\n", encoding="utf-8")
        self.assert_fails(self.configure(), "Required HRX commit is unavailable")

    def test_source_archive(self):
        archive = self.root / "archive"
        archive.mkdir()
        self.assert_fails(self.configure(archive), "Cannot identify the HRX Git checkout")

    def test_enclosing_repository(self):
        archive = self.hrx / "unpacked-hrx"
        archive.mkdir()
        self.assert_fails(self.configure(archive), "must be the root of its own Git checkout")

    def test_worktree(self):
        worktree = self.root / "worktree"
        self.git("worktree", "add", "--detach", str(worktree), self.minimum, cwd=self.hrx)
        self.assert_passes(self.configure(worktree))

    def test_submodule(self):
        parent = self.root / "parent"
        self.git("init", "--quiet", str(parent))
        self.git("-c", "protocol.file.allow=always", "submodule", "add", str(self.hrx), "hrx", cwd=parent)
        self.assert_passes(self.configure(parent / "hrx"))

    def test_shallow_history(self):
        shallow = self.root / "shallow"
        self.git("clone", "--depth=1", self.hrx.as_uri(), str(shallow))
        self.assert_fails(self.configure(shallow), "Required HRX commit is unavailable")
        # An available pin object does not prove ancestry through a missing parent chain.
        self.git("fetch", "--depth=1", "origin", self.minimum, cwd=shallow)
        self.assert_fails(self.configure(shallow), "does not have the required commit")
        self.git("checkout", "--quiet", self.minimum, cwd=shallow)
        self.assert_passes(self.configure(shallow))

    def test_git_missing(self):
        self.assert_fails(
            self.configure(self.hrx, "-DCMAKE_DISABLE_FIND_PACKAGE_Git=TRUE"),
            "Git is required",
        )

    def test_reconfigure_after_checkout_changes(self):
        self.assert_passes(self.configure())
        self.git("checkout", "--quiet", self.old, cwd=self.hrx)
        self.assert_fails(self.configure(), "does not have the required commit")

    def test_pin_change_triggers_reconfigure(self):
        earlier = time.time() - 10
        os.utime(self.pin, (earlier, earlier))
        self.git("checkout", "--quiet", self.minimum, cwd=self.hrx)
        self.assert_passes(self.configure())
        self.minimum = self.new
        self.pin.write_text(self.minimum + "\n", encoding="utf-8")
        result = subprocess.run(
            ["cmake", "--build", str(self.build)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self.assert_fails(result, "does not have the required commit")


if __name__ == "__main__":
    if shutil.which("cmake") is None:
        raise SystemExit("cmake is required")
    if shutil.which("git") is None:
        raise SystemExit("git is required")
    unittest.main()
