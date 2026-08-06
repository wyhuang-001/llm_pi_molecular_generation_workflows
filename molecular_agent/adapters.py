from __future__ import annotations

from typing import Any


class NotConfiguredAdapter:
    def __init__(self, stage: str):
        self.stage = stage

    def run(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": "not_configured",
            "message": f"{self.stage} adapter is intentionally not configured in the minimal first version.",
        }
