from __future__ import annotations

from typing import Any


class NullProgress:
    def __init__(self, *, total: int | None = None) -> None:
        self.total = total
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += int(n)

    def set_description(self, _desc: str) -> None:
        return None

    def set_postfix(self, _values: dict[str, Any] | None = None, **_kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def make_progress(cfg: Any, *, total: int | None, desc: str, leave: bool = False):
    if not bool(getattr(cfg, "show_progress", True)):
        return NullProgress(total=total)
    try:
        from tqdm.auto import tqdm
    except Exception:
        return NullProgress(total=total)
    return tqdm(total=total, desc=desc, leave=leave)
