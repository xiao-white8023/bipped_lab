#!/usr/bin/env python3
"""Preflight all replacement hashes, back up originals, then install atomically."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    repo = args.repo.expanduser().resolve()
    if not (repo/'legged_lab').is_dir():
        raise SystemExit('目标不是包含 legged_lab 的仓库目录。')
    manifest = json.loads((package/'SHA256SUMS.json').read_text())
    pending, conflicts = [], []
    for item in manifest['files']:
        relative = Path(item['path'])
        source, target = package/'files'/relative, repo/relative
        if relative.is_absolute() or '..' in relative.parts or not target.resolve().is_relative_to(repo):
            raise SystemExit(f'非法替换路径：{relative}')
        if digest(source) != item['sha256']:
            raise SystemExit(f'修改包中的文件校验失败：{relative}')
        if target.exists() and (not target.is_file() or target.is_symlink()):
            conflicts.append(str(relative)); continue
        current = digest(target)
        if current == item['sha256']:
            continue
        if current != item['original_sha256']:
            conflicts.append(str(relative)); continue
        pending.append((relative, source, target, current is not None))
    if conflicts:
        raise SystemExit('这些文件与基准版本不同，请先手动合并，尚未修改任何文件：\n'+'\n'.join(conflicts))
    if not pending:
        print('所有文件已是修改后的版本。'); return
    if args.dry_run:
        print(f'校验通过：将替换/新增 {len(pending)} 个文件。'); return
    backup = repo/'selector_fix_backups'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    # Copy every existing original before modifying the first target.
    for relative, source, target, existed in pending:
        if existed:
            saved = backup/relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
    backup.mkdir(parents=True, exist_ok=True)
    (backup/'install_manifest.json').write_text(json.dumps({
        'files': [str(item[0]) for item in pending],
        'new_files': [str(item[0]) for item in pending if not item[3]],
    }, ensure_ascii=False, indent=2))
    installed = []
    try:
        for relative, source, target, existed in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(prefix='.selector-', dir=target.parent)
            os.close(handle)
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
            installed.append((relative, target, existed))
    except BaseException:
        for relative, target, existed in reversed(installed):
            if existed:
                shutil.copy2(backup/relative, target)
            else:
                target.unlink(missing_ok=True)
        raise
    print(f'已替换/新增 {len(installed)} 个文件。原文件备份：{backup}')


if __name__ == '__main__':
    main()
