"""Static dependency and size checks for architecture V2 modules."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_FILE = Path(__file__).resolve().parent / "data" / "architecture_rules.yaml"


@dataclass(frozen=True)
class LayerRule:
    name: str
    paths: Tuple[Path, ...]
    forbidden_imports: Tuple[str, ...]
    allowed_imports: Tuple[Tuple[str, str], ...] = ()
    enforce_module_size: bool = True


@dataclass(frozen=True)
class ArchitectureRules:
    repository_root: Path
    max_module_lines: int
    layers: Tuple[LayerRule, ...]


@dataclass(frozen=True)
class ArchitectureViolation:
    rule: str
    path: Path
    line: int
    detail: str

    def format(self) -> str:
        relative = self.path.relative_to(REPOSITORY_ROOT) if self.path.is_relative_to(REPOSITORY_ROOT) else self.path
        return f"{relative}:{self.line}: {self.rule}: {self.detail}"


def load_architecture_rules(
    path: Path | str = DEFAULT_RULES_FILE,
    repository_root: Path | str = REPOSITORY_ROOT,
) -> ArchitectureRules:
    source = Path(path).resolve()
    root = Path(repository_root).resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported architecture rules schema")
    max_lines = payload.get("max_module_lines")
    if not isinstance(max_lines, int) or max_lines <= 0:
        raise ValueError("max_module_lines must be a positive integer")
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, dict) or not raw_layers:
        raise ValueError("layers must be a non-empty mapping")

    layers = []
    for name, raw_rule in raw_layers.items():
        if not isinstance(raw_rule, dict):
            raise ValueError(f"layer {name!r} must be a mapping")
        paths = raw_rule.get("paths")
        forbidden = raw_rule.get("forbidden_imports", [])
        allowed = raw_rule.get("allowed_imports", [])
        enforce_size = raw_rule.get("enforce_module_size", True)
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"layer {name!r} must declare paths")
        if not isinstance(forbidden, list):
            raise ValueError(f"layer {name!r} forbidden_imports must be a list")
        if not isinstance(allowed, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("module"), str)
            for item in allowed
        ):
            raise ValueError(f"layer {name!r} allowed_imports must contain path/module mappings")
        if not isinstance(enforce_size, bool):
            raise ValueError(f"layer {name!r} enforce_module_size must be boolean")
        layers.append(LayerRule(
            name=str(name),
            paths=tuple(root / item for item in paths),
            forbidden_imports=tuple(str(item) for item in forbidden),
            allowed_imports=tuple(
                (item["path"].replace("\\", "/"), item["module"])
                for item in allowed
            ),
            enforce_module_size=enforce_size,
        ))
    return ArchitectureRules(root, max_lines, tuple(layers))


def _module_imports(tree: ast.AST) -> Iterable[Tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.lineno, node.module


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def audit_architecture(
    rules: ArchitectureRules | None = None,
) -> Tuple[ArchitectureViolation, ...]:
    rules = rules or load_architecture_rules()
    violations = []
    for layer in rules.layers:
        used_allowed_imports = set()
        for directory in layer.paths:
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*.py")):
                source = path.read_text(encoding="utf-8")
                line_count = len(source.splitlines())
                if layer.enforce_module_size and line_count > rules.max_module_lines:
                    violations.append(ArchitectureViolation(
                        "module_size",
                        path,
                        1,
                        f"{line_count} lines exceeds {rules.max_module_lines}",
                    ))
                try:
                    tree = ast.parse(source, filename=str(path))
                except SyntaxError as exc:
                    violations.append(ArchitectureViolation(
                        "syntax",
                        path,
                        exc.lineno or 1,
                        exc.msg,
                    ))
                    continue
                for line, module in _module_imports(tree):
                    for prefix in layer.forbidden_imports:
                        if _matches_prefix(module, prefix):
                            relative = path.relative_to(
                                rules.repository_root).as_posix()
                            matched_allowed = next((
                                (allowed_path, allowed_module)
                                for allowed_path, allowed_module in layer.allowed_imports
                                if relative == allowed_path
                                and _matches_prefix(module, allowed_module)
                            ), None)
                            if matched_allowed is not None:
                                used_allowed_imports.add(matched_allowed)
                                continue
                            violations.append(ArchitectureViolation(
                                "forbidden_import",
                                path,
                                line,
                                f"layer {layer.name} imports {module}",
                            ))
        for allowed_path, allowed_module in layer.allowed_imports:
            if (allowed_path, allowed_module) not in used_allowed_imports:
                violations.append(ArchitectureViolation(
                    "stale_allowlist",
                    rules.repository_root / allowed_path,
                    1,
                    f"layer {layer.name} no longer imports {allowed_module}",
                ))
    return tuple(violations)
