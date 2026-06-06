from __future__ import annotations

from .types import *

class TensorStore(nn.Module):
    def __init__(
        self,
        initial_tensors: dict[str, torch.Tensor],
        *,
        mode: str,
        latent_rank: int,
        latent_delta_scale: float,
        latent_factor_init_std: float,
    ) -> None:
        super().__init__()
        if mode not in {"direct", "latent"}:
            raise ValueError(f"mode must be 'direct' or 'latent', got {mode!r}")
        self.mode = str(mode)
        self._name_to_key: dict[str, str] = {}
        modules: dict[str, nn.Module] = {}
        for idx, (name, initial) in enumerate(initial_tensors.items()):
            key = f"p{idx:04d}"
            self._name_to_key[name] = key
            if mode == "direct":
                modules[key] = DirectTensor(initial)
            else:
                modules[key] = LowRankDecodedTensor(
                    initial,
                    rank=latent_rank,
                    delta_scale=latent_delta_scale,
                    factor_init_std=latent_factor_init_std,
                )
        self.tensors = nn.ModuleDict(modules)

    def tensor(self, name: str) -> torch.Tensor:
        return self.tensors[self._name_to_key[name]]()

    def decoded_numel(self) -> int:
        total = 0
        for module in self.tensors.values():
            if isinstance(module, DirectTensor):
                total += int(module.value.numel())
            elif isinstance(module, LowRankDecodedTensor):
                total += int(module.base.numel())
        return int(total)

