"""Static dependency and size checks for architecture V2 modules."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib.util
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
    owns_paths: bool = True


@dataclass(frozen=True)
class ArchitectureRules:
    repository_root: Path
    max_module_lines: int
    layers: Tuple[LayerRule, ...]
    source_root: Path | None = None


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

    raw_source_root = payload.get("source_root")
    if not isinstance(raw_source_root, str) or not raw_source_root.strip():
        raise ValueError("source_root must be a non-empty repository-relative path")
    source_root = (root / raw_source_root).resolve()
    if not source_root.is_relative_to(root) or source_root == root:
        raise ValueError("source_root must stay below repository_root")

    layers = []
    for name, raw_rule in raw_layers.items():
        if not isinstance(raw_rule, dict):
            raise ValueError(f"layer {name!r} must be a mapping")
        paths = raw_rule.get("paths")
        forbidden = raw_rule.get("forbidden_imports", [])
        allowed = raw_rule.get("allowed_imports", [])
        enforce_size = raw_rule.get("enforce_module_size", True)
        owns_paths = raw_rule.get("owns_paths", True)
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
        if not isinstance(owns_paths, bool):
            raise ValueError(f"layer {name!r} owns_paths must be boolean")
        layers.append(LayerRule(
            name=str(name),
            paths=tuple(root / item for item in paths),
            forbidden_imports=tuple(str(item) for item in forbidden),
            allowed_imports=tuple(
                (item["path"].replace("\\", "/"), item["module"])
                for item in allowed
            ),
            enforce_module_size=enforce_size,
            owns_paths=owns_paths,
        ))
    return ArchitectureRules(root, max_lines, tuple(layers), source_root)


def _path_module(path: Path, source_root: Path) -> tuple[str, str]:
    """Return (module, package) names for one file below source_root."""
    package_root = source_root.name
    relative = path.relative_to(source_root)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
        module = ".".join((package_root, *parts))
        return module, module
    module = ".".join((package_root, *parts))
    package = ".".join((package_root, *parts[:-1]))
    return module, package


def _module_imports(
    tree: ast.AST,
    *,
    package: str | None = None,
) -> Iterable[Tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if not package:
                    continue
                relative = "." * node.level + (node.module or "")
                try:
                    resolved = importlib.util.resolve_name(relative, package)
                except (ImportError, ValueError):
                    continue
                yield node.lineno, resolved
            elif node.module:
                yield node.lineno, node.module


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _path_in_scope(path: Path, scope: Path) -> bool:
    return path == scope if scope.suffix == ".py" else path.is_relative_to(scope)


def _python_files(scope: Path) -> Iterable[Path]:
    if scope.is_file() and scope.suffix == ".py":
        yield scope
    elif scope.is_dir():
        yield from sorted(scope.rglob("*.py"))


def _scope_module_prefix(scope: Path, repository_root: Path) -> str:
    relative = scope.relative_to(repository_root)
    if scope.suffix == ".py":
        relative = relative.with_suffix("")
        if relative.name == "__init__":
            relative = relative.parent
    return relative.as_posix().replace("/", ".")


def audit_architecture(
    rules: ArchitectureRules | None = None,
) -> Tuple[ArchitectureViolation, ...]:
    rules = rules or load_architecture_rules()
    violations = []
    source_root = rules.source_root
    owned_files: dict[Path, LayerRule] = {}
    if source_root is not None:
        if not source_root.is_dir():
            violations.append(ArchitectureViolation(
                "source_root", source_root, 1, "source_root does not exist"))
        else:
            ownership_layers = tuple(
                layer for layer in rules.layers if layer.owns_paths)
            for path in sorted(source_root.rglob("*.py")):
                owners = tuple(
                    layer for layer in ownership_layers
                    if any(_path_in_scope(path, directory) for directory in layer.paths)
                )
                if len(owners) == 1:
                    owned_files[path] = owners[0]
                elif not owners:
                    violations.append(ArchitectureViolation(
                        "unmapped_module", path, 1,
                        "module is not owned by any architecture layer"))
                else:
                    violations.append(ArchitectureViolation(
                        "duplicate_layer_ownership", path, 1,
                        "module is owned by layers "
                        + ", ".join(layer.name for layer in owners)))

    layer_edges: dict[str, set[str]] = {
        layer.name: set() for layer in rules.layers if layer.owns_paths}
    ownership_prefixes = tuple(
        (_scope_module_prefix(directory, rules.repository_root), layer)
        for layer in rules.layers if layer.owns_paths
        for directory in layer.paths
    )
    for layer in rules.layers:
        used_allowed_imports = set()
        for directory in layer.paths:
            if not directory.exists():
                continue
            for path in _python_files(directory):
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
                package = None
                if source_root is not None and path.is_relative_to(source_root):
                    _module, package = _path_module(path, source_root)
                for line, module in _module_imports(tree, package=package):
                    if layer.owns_paths:
                        candidates = [
                            (prefix, owner) for prefix, owner in ownership_prefixes
                            if _matches_prefix(module, prefix)
                        ]
                        if candidates:
                            _prefix, target = max(
                                candidates, key=lambda item: len(item[0]))
                            if target.name != layer.name:
                                layer_edges[layer.name].add(target.name)
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
    visiting: set[str] = set()
    visited: set[str] = set()
    reported_cycles: set[tuple[str, ...]] = set()

    def visit(layer_name: str, stack: tuple[str, ...]) -> None:
        if layer_name in visiting:
            start = stack.index(layer_name)
            cycle = stack[start:] + (layer_name,)
            canonical = tuple(sorted(set(cycle)))
            if canonical not in reported_cycles:
                reported_cycles.add(canonical)
                violations.append(ArchitectureViolation(
                    "dependency_cycle",
                    source_root or rules.repository_root,
                    1,
                    "layer dependency cycle: " + " -> ".join(cycle),
                ))
            return
        if layer_name in visited:
            return
        visiting.add(layer_name)
        for target in sorted(layer_edges.get(layer_name, ())):
            visit(target, stack + (layer_name,))
        visiting.remove(layer_name)
        visited.add(layer_name)

    if source_root is not None:
        for layer_name in sorted(layer_edges):
            visit(layer_name, ())
    return tuple(violations)
