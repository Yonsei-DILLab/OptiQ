"""Regression checks for fresh worktrees that omit MFPO submodule contents."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from .dependencies import MFPO_PATH, load_mfpo_config, prepare_dependencies, verify_dependencies


class DependencyTests(unittest.TestCase):
    def git(self, root, *args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()

    def repository(self, root):
        root.mkdir()
        self.git(root, 'init')
        self.git(root, 'config', 'user.name', 'Dependency regression')
        self.git(root, 'config', 'user.email', 'test@example.invalid')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        dependency = base / 'mfpo-upstream'
        self.repository(dependency)
        for name, text in [('configs/mfpo_config.py', 'def get_config(): return {"from_pinned_mfpo": True}\n'),
                           ('jaxrl5/agents/mean_flow_learner.py', '# fixture\n')]:
            path = dependency / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.git(dependency, 'add', '.')
        self.git(dependency, 'commit', '-m', 'Pinned MFPO fixture')
        self.revision = self.git(dependency, 'rev-parse', 'HEAD')
        parent = base / 'repository'
        self.repository(parent)
        (parent / '.gitmodules').write_text(
            f'[submodule "{MFPO_PATH}"]\n\tpath = {MFPO_PATH}\n\turl = {dependency}\n')
        self.git(parent, 'add', '.gitmodules')
        self.git(parent, 'update-index', '--add', '--cacheinfo', f'160000,{self.revision},{MFPO_PATH}')
        self.git(parent, 'commit', '-m', 'Record submodule pin')
        self.source = base / 'frozen-source'
        self.git(parent, 'worktree', 'add', '--detach', str(self.source), 'HEAD')

    def prepare(self):
        with patch.dict(os.environ, {'GIT_ALLOW_PROTOCOL': 'file'}):
            return prepare_dependencies(self.source)

    def test_fresh_worktree_is_rejected_then_prepared_at_pinned_revision(self):
        with self.assertRaisesRegex(RuntimeError, 'submodule is not initialized'):
            verify_dependencies(self.source)
        proof = self.prepare()
        self.assertEqual(proof['mfpo']['commit'], self.revision)
        self.assertEqual(self.git(self.source, 'status', '--porcelain'), '')
        self.assertEqual(prepare_dependencies(self.source), proof)

    def test_other_methods_do_not_require_or_initialize_mfpo(self):
        self.assertEqual(prepare_dependencies(self.source, ('optiq', 'sac', 'dipo')), {})
        self.assertFalse((self.source / MFPO_PATH / 'configs/mfpo_config.py').exists())

    def test_conflicting_configs_package_does_not_shadow_mfpo(self):
        self.prepare()
        shadow = types.ModuleType('configs')
        shadow.__path__ = []
        with patch.dict(sys.modules, {'configs': shadow}):
            self.assertEqual(load_mfpo_config(self.source), {'from_pinned_mfpo': True})
            self.assertIs(sys.modules['configs'], shadow)

    def test_prepare_refuses_to_overwrite_dirty_or_wrong_revision(self):
        self.prepare()
        dependency = self.source / MFPO_PATH
        path = dependency / 'configs/mfpo_config.py'
        path.write_text(path.read_text() + '# local change\n')
        with self.assertRaisesRegex(RuntimeError, 'uncommitted changes'):
            prepare_dependencies(self.source)
        self.git(dependency, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                 'commit', '-am', 'Wrong dependency revision')
        with self.assertRaisesRegex(RuntimeError, 'revision mismatch'):
            prepare_dependencies(self.source)


if __name__ == '__main__':
    unittest.main()
