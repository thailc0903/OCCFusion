"""Snapshot filesystem metadata once per sensor file; verify stability on exit.

DrvFS makes repeated Path.resolve/is_file/stat very costly. This changes only
how the original metadata adapter obtains filesystem facts, not its selection,
calibration, timestamps or model preprocessing. No model imports.
"""
from contextlib import contextmanager
from pathlib import Path
import stat


def signature(info):
    return info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns


class DatasetPaths:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.parents={}
        self.files={}
        self.directories={}

    def inside(self,path):
        return path.is_absolute() and path.is_relative_to(self.root)

    def resolve(self,path,strict=False):
        path=Path(path)
        if not self.inside(path):raise ValueError('Sensor path escapes dataset root')
        if path==self.root:return self.root
        if path.parent not in self.parents:
            parent=path.parent.resolve()
            if not parent.is_relative_to(self.root):raise ValueError('Sensor directory escapes dataset root')
            self.parents[path.parent]=parent
            self.directories[path.parent]=signature(path.parent.lstat())
        if path not in self.files:
            try:
                original=path.lstat()
                resolved=path.resolve(strict=strict) if stat.S_ISLNK(original.st_mode) else self.parents[path.parent]/path.name
                if not resolved.is_relative_to(self.root):raise ValueError('Sensor symlink escapes dataset root')
                followed=resolved.stat() if stat.S_ISLNK(original.st_mode) else original
                self.files[path]=(resolved,original,followed)
            except FileNotFoundError:
                if strict:raise
                # Missing files are not cached as success; original adapter will reject them.
                return self.parents[path.parent]/path.name
        return self.files[path][0]

    def stat(self,path,follow_symlinks=True):
        path=Path(path)
        if not self.inside(path):raise ValueError('Sensor stat escapes dataset root')
        if path not in self.files:self.resolve(path,strict=True)
        if path in self.files:return self.files[path][2 if follow_symlinks else 1]
        return path.stat(follow_symlinks=follow_symlinks)

    def verify_unchanged(self):
        for path,expected in self.directories.items():
            if signature(path.lstat())!=expected:raise ValueError('Sensor directory changed during metadata snapshot')
        for path,(_,original,followed) in self.files.items():
            if signature(path.lstat())!=signature(original) or signature(path.stat())!=signature(followed):
                raise ValueError('Sensor file changed during metadata snapshot')

    @contextmanager
    def use(self,catalog,*modules):
        snapshot=self
        class SnapshotPath(type(Path())):
            def resolve(self,strict=False):
                path=Path(str(self))
                return snapshot.resolve(path,strict) if snapshot.inside(path) else path.resolve(strict=strict)
            def stat(self,*,follow_symlinks=True):
                path=Path(str(self))
                return snapshot.stat(path,follow_symlinks) if snapshot.inside(path) else path.stat(follow_symlinks=follow_symlinks)
        original_root=catalog.root
        originals=[(module,module.Path) for module in modules]
        catalog.root=SnapshotPath(str(original_root))
        for module,_ in originals:module.Path=SnapshotPath
        try:
            yield self
            self.verify_unchanged()
        finally:
            catalog.root=original_root
            for module,old in originals:module.Path=old
