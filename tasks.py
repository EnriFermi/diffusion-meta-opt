import random
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TaskSpec:
    name: str
    weight: float = 1.0
    params: Optional[dict] = None


class Task:
    def sample(self, device):
        raise NotImplementedError


class SineTask(Task):
    def __init__(self, n_train, n_val, x_min, x_max, x_coef=0.1, noise_std=0.0):
        self.n_train = n_train
        self.n_val = n_val
        self.x_min = x_min
        self.x_max = x_max
        self.x_coef = x_coef
        self.noise_std = noise_std

    def _sample_x(self, n, device):
        return (self.x_max - self.x_min) * torch.rand(n, 1, device=device) + self.x_min

    def _sample_y(self, x):
        y = torch.sin(x) + self.x_coef * x
        if self.noise_std > 0:
            y = y + self.noise_std * torch.randn_like(y)
        return y

    def sample(self, device):
        x_train = self._sample_x(self.n_train, device)
        y_train = self._sample_y(x_train)
        x_val = self._sample_x(self.n_val, device)
        y_val = self._sample_y(x_val)
        return x_train, y_train, x_val, y_val


class CosineTask(Task):
    def __init__(self, n_train, n_val, x_min, x_max, x_coef=0.1, noise_std=0.0):
        self.n_train = n_train
        self.n_val = n_val
        self.x_min = x_min
        self.x_max = x_max
        self.x_coef = x_coef
        self.noise_std = noise_std

    def _sample_x(self, n, device):
        return (self.x_max - self.x_min) * torch.rand(n, 1, device=device) + self.x_min

    def _sample_y(self, x):
        y = torch.cos(x) + self.x_coef * x
        if self.noise_std > 0:
            y = y + self.noise_std * torch.randn_like(y)
        return y

    def sample(self, device):
        x_train = self._sample_x(self.n_train, device)
        y_train = self._sample_y(x_train)
        x_val = self._sample_x(self.n_val, device)
        y_val = self._sample_y(x_val)
        return x_train, y_train, x_val, y_val


class PolyTask(Task):
    def __init__(self, n_train, n_val, x_min, x_max, coeffs, noise_std=0.0):
        self.n_train = n_train
        self.n_val = n_val
        self.x_min = x_min
        self.x_max = x_max
        self.coeffs = coeffs
        self.noise_std = noise_std

    def _sample_x(self, n, device):
        return (self.x_max - self.x_min) * torch.rand(n, 1, device=device) + self.x_min

    def _poly(self, x):
        y = torch.zeros_like(x)
        for power, coef in enumerate(self.coeffs):
            y = y + coef * (x ** power)
        return y

    def _sample_y(self, x):
        y = self._poly(x)
        if self.noise_std > 0:
            y = y + self.noise_std * torch.randn_like(y)
        return y

    def sample(self, device):
        x_train = self._sample_x(self.n_train, device)
        y_train = self._sample_y(x_train)
        x_val = self._sample_x(self.n_val, device)
        y_val = self._sample_y(x_val)
        return x_train, y_train, x_val, y_val


TASK_REGISTRY = {
    "sine": SineTask,
    "cosine": CosineTask,
    "poly": PolyTask,
}


class TaskMixer:
    def __init__(self, tasks, weights=None, strategy="weighted_random"):
        if not tasks:
            raise ValueError("TaskMixer requires at least one task")
        self.tasks = tasks
        self.strategy = strategy
        if weights is None:
            weights = [1.0] * len(tasks)
        self.weights = weights
        self._rr_idx = 0

    def sample_task(self):
        if self.strategy == "round_robin":
            task = self.tasks[self._rr_idx]
            self._rr_idx = (self._rr_idx + 1) % len(self.tasks)
            return task
        if self.strategy == "uniform":
            return random.choice(self.tasks)
        return random.choices(self.tasks, weights=self.weights, k=1)[0]


def _merge_params(defaults, overrides):
    params = dict(defaults or {})
    params.update(overrides or {})
    return params


def build_task_sampler(cfg):
    if hasattr(cfg, "tasks") and cfg.tasks is not None:
        defaults = cfg.tasks.get("defaults", {})
        mix = cfg.tasks.get("mix", [])
        strategy = cfg.tasks.get("strategy", "weighted_random")

        tasks = []
        weights = []
        for spec in mix:
            name = spec.get("name")
            if name not in TASK_REGISTRY:
                raise ValueError(f"Unknown task: {name}")
            params = _merge_params(defaults, spec.get("params", {}))
            tasks.append(TASK_REGISTRY[name](**params))
            weights.append(float(spec.get("weight", 1.0)))
        if not tasks:
            raise ValueError("cfg.tasks.mix is empty")
        return TaskMixer(tasks, weights=weights, strategy=strategy)

    # Backward-compat: use legacy cfg.task with sine task defaults
    defaults = {
        "n_train": cfg.task.n_train,
        "n_val": cfg.task.n_val,
        "x_min": cfg.task.x_min,
        "x_max": cfg.task.x_max,
        "x_coef": 0.1,
        "noise_std": 0.0,
    }
    task = SineTask(**defaults)
    return TaskMixer([task], weights=[1.0], strategy="weighted_random")
