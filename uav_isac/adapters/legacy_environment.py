"""Adapter that keeps the existing Gymnasium environment behind a V2 port."""

from __future__ import annotations

from typing import Any, Mapping


def build_legacy_environment(config_path: str | None, seed: int):
    """Construct the legacy environment at the V2 composition boundary."""

    from uav_isac.environment.env_wrapper import UAVISACEnv
    from uav_isac.adapters.configuration import load_resolved_configuration

    resolved = load_resolved_configuration(config_path)
    return LegacyGymEnvironmentAdapter(UAVISACEnv(config=resolved.value, seed=seed))


def build_environment_from_resolved(resolved, seed: int):
    """Compose an environment from an already validated configuration."""

    from uav_isac.environment.env_wrapper import UAVISACEnv

    return LegacyGymEnvironmentAdapter(UAVISACEnv(config=resolved.value, seed=seed))


class LegacyGymEnvironmentAdapter:
    """Expose the current environment through the application port.

    Importing the legacy environment remains the composition root's job. This
    keeps application/domain modules free of Gymnasium and the legacy package.
    """

    def __init__(self, environment: Any):
        required = ("reset", "step")
        missing = [name for name in required if not callable(getattr(environment, name, None))]
        if missing:
            raise TypeError(f"legacy environment is missing methods: {missing}")
        self._environment = environment

    @property
    def legacy_environment(self) -> Any:
        return self._environment

    def reset(self, seed: int):
        return self._environment.reset(seed=seed)

    def step(self, actions: Mapping[str, Any]):
        return self._environment.step(dict(actions))
