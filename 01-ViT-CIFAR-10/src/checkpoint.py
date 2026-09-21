"""Epoch-boundary checkpoints: basic containers/tensors, weights_only=True."""
import copy
import os
from pathlib import Path
import random

import numpy as np
import torch


def cpu_snapshot(value):
    """Copy tensors so a saved best state cannot change with later optimizer steps."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def capture_rng(generators):
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all(),
        'loaders': {name: generator.get_state() for name, generator in generators.items()},
    }


def restore_rng(state, generators):
    random.setstate(state['python'])
    algorithm, keys, position, has_gauss, cached_gaussian = state['numpy']
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position,
                         has_gauss, cached_gaussian))
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])
    for name, generator in generators.items():
        generator.set_state(state['loaders'][name])


def atomic_save(payload, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def load_checkpoint(path, expected=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema_version') != 1:
        raise ValueError('Unsupported checkpoint schema')
    if expected is not None and payload['identity'] != expected:
        raise ValueError('Checkpoint configuration, data, code, device or environment differs')
    return payload
