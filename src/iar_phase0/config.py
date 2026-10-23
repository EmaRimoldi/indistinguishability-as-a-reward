from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def root(self) -> Path:
        return Path(self.raw["project"]["root"]).expanduser()

    def resolve(self, *parts: str) -> Path:
        if not parts:
            return self.root
        path = Path(*parts)
        if path.is_absolute():
            return path
        return self.root / path


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ProjectConfig(raw=raw, path=config_path)


def ensure_dirs(config: ProjectConfig) -> None:
    paths = config.raw["paths"]
    for key in ["data_cache", "hf_cache", "experiments"]:
        config.resolve(paths[key]).mkdir(parents=True, exist_ok=True)
    config.resolve("data/raw").mkdir(parents=True, exist_ok=True)
    config.resolve("data/processed").mkdir(parents=True, exist_ok=True)
    config.resolve("logs").mkdir(parents=True, exist_ok=True)

