from dataclasses import dataclass
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    runs_root: Path
    dataset_roots: tuple[Path, ...]
    cache_entries: int = 8
    state_root: Path | None = None
    session_ttl_seconds: int = 28800
    cookie_secure: bool = False
    allowed_origins: tuple[str, ...] = (
        'http://localhost:3000', 'http://127.0.0.1:3000',
        'http://localhost:8000', 'http://127.0.0.1:8000',
    )
    evaluation_roots: tuple[Path, ...] | None = None
    benchmark_roots: tuple[Path, ...] | None = None
    job_profiles_path: Path = REPO / 'web/inference_profiles.json'
    gpu_lock_path: Path = REPO / 'work/inference-worker/gpu.lock'
    camera_policy_path: Path = REPO / 'web/camera_policy.json'

    @property
    def benchmark_directories(self):
        return self.benchmark_roots if self.benchmark_roots is not None else (self.runs_root,)

    @property
    def evaluation_directories(self):
        return self.evaluation_roots if self.evaluation_roots is not None else (self.runs_root,)

    @property
    def state_directory(self):
        return (self.state_root or self.runs_root.parent / 'viewer-state').resolve()

    @classmethod
    def load(cls):
        source = Path(os.environ.get('VIEWER_CONFIG', REPO / 'web/viewer.config.json')).resolve()
        value = json.loads(source.read_text(encoding='utf-8'))

        def absolute(path):
            path = Path(path).expanduser()
            return (source.parent / path).resolve() if not path.is_absolute() else path.resolve()

        runs_root = absolute(os.environ.get('VIEWER_RUNS_ROOT', value['runs_root']))
        roots = json.loads(os.environ['VIEWER_DATA_ROOTS']) if 'VIEWER_DATA_ROOTS' in os.environ else value['dataset_roots']
        if not roots or not isinstance(roots, list):
            raise ValueError('dataset_roots must be a nonempty list of configured directories')
        entries = int(value.get('cache_entries', 8))
        if not 1 <= entries <= 32:
            raise ValueError('cache_entries must be in 1..32')
        ttl = int(os.environ.get('VIEWER_SESSION_TTL_SECONDS', value.get('session_ttl_seconds', 28800)))
        if not 60 <= ttl <= 604800:
            raise ValueError('session_ttl_seconds must be in 60..604800')
        secure = os.environ.get('VIEWER_COOKIE_SECURE', str(value.get('cookie_secure', False))).lower()
        if secure not in ('true', 'false', '1', '0'):
            raise ValueError('VIEWER_COOKIE_SECURE must be true/false')
        origins = json.loads(os.environ['VIEWER_ALLOWED_ORIGINS']) if 'VIEWER_ALLOWED_ORIGINS' in os.environ else value.get('allowed_origins', list(cls.allowed_origins))
        if not origins or not isinstance(origins, list) or not all(isinstance(v, str) for v in origins):
            raise ValueError('allowed_origins must be a nonempty list')
        state_root = absolute(os.environ.get('VIEWER_STATE_ROOT', value.get('state_root', '../work/viewer')))
        evaluations = json.loads(os.environ['VIEWER_EVALUATION_ROOTS']) if 'VIEWER_EVALUATION_ROOTS' in os.environ else value.get('evaluation_roots', [str(runs_root)])
        if not isinstance(evaluations, list) or not all(isinstance(v, str) for v in evaluations):
            raise ValueError('evaluation_roots must be a list of configured directories')
        benchmarks = json.loads(os.environ['VIEWER_BENCHMARK_ROOTS']) if 'VIEWER_BENCHMARK_ROOTS' in os.environ else value.get('benchmark_roots', [str(runs_root)])
        if not isinstance(benchmarks, list) or not all(isinstance(v, str) for v in benchmarks):
            raise ValueError('benchmark_roots must be a list of configured directories')
        return cls(runs_root, tuple(absolute(root) for root in roots), entries,
                   state_root, ttl, secure in ('true', '1'), tuple(origins),
                   tuple(absolute(root) for root in evaluations),
                   tuple(absolute(root) for root in benchmarks),
                   absolute(os.environ.get('VIEWER_JOB_PROFILES', value.get('job_profiles_path', 'inference_profiles.json'))),
                   absolute(os.environ.get('VIEWER_GPU_LOCK', value.get('gpu_lock_path', '../work/inference-worker/gpu.lock'))),
                   absolute(os.environ.get('VIEWER_CAMERA_POLICY', value.get('camera_policy_path', 'camera_policy.json'))))
