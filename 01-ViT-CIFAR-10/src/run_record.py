"""Keep run evidence separate from the teaching code; full records stay on SSD."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def record_run(project, args, config):
    """Snapshot exact code/config/dependencies before running an observation."""
    lab_root = Path(os.environ['LAB_ROOT']).resolve(strict=True)
    for path in (args.data_root, args.split_path, args.run_dir):
        if not path.resolve().is_relative_to(lab_root):
            raise ValueError('Data, splits and full outputs must remain under LAB_ROOT')
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'config.json').exists():
        raise FileExistsError('Use a new run directory')
    for name in ('source', 'metrics', 'outputs', 'logs'):
        (run / name).mkdir(exist_ok=True)
    write_json(run / 'config.json', config)
    write_json(run / 'status.json', {'status': 'running', 'stage': config['stage']})
    sources = [project / name for name in ('README.md', 'PLAN.md')]
    sources += list(project.glob('requirements*.txt'))
    for directory in ('src', 'scripts', 'configs', 'tests'):
        sources += [p for p in (project / directory).rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts]
    manifest = {}
    for source in sorted(sources):
        relative = source.relative_to(project)
        destination = run / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest[str(relative)] = file_hash(destination)
    write_json(run / 'source-manifest.json', manifest)
    shutil.copyfile(args.split_path, run / 'split.json')
    (run / 'requirements-resolved.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    head = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=project, capture_output=True, text=True)
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=project, text=True)
    if head.returncode == 0:
        (run / 'changes.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD', '--', '.'], cwd=project))
    versions = {name: importlib.metadata.version(name)
                for name in ('torch', 'torchvision', 'numpy', 'matplotlib', 'nvidia-cudnn-cu13')}
    write_json(run / 'metadata.local.json', {
        'machine_time_utc': datetime.now(timezone.utc).isoformat(),
        'clock_note': 'Machine wall clock is not calibrated; elapsed time uses perf_counter.',
        'python': sys.version, 'executable': sys.executable, 'command': [sys.executable, *sys.argv],
        'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'), 'device': 'cuda:0',
        'data_root': str(args.data_root), 'split_path': str(args.split_path), 'run_dir': str(run),
        'git_commit': head.stdout.strip() if head.returncode == 0 else None,
        'git_status': status, 'code_status': 'clean' if not status else 'modified; use source snapshot',
        'versions': versions,
        'gpu_snapshot': subprocess.check_output([
            'nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
            '--query-gpu=index,uuid,name,driver_version,memory.used,memory.free', '--format=csv'], text=True),
    })
    return versions
