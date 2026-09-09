"""Canonical orchestration for a bounded simulation episode."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Tuple

from uav_isac.domain.episode import EpisodeResult, EpisodeSpec, FrameResult


Observations = Mapping[str, Any]
Actions = Mapping[str, Any]


class EpisodeEnvironment(Protocol):
    """Port required by the episode use case."""

    def reset(self, seed: int) -> Tuple[Observations, Mapping[str, Any]]:
        ...

    def step(
        self,
        actions: Actions,
    ) -> Tuple[
        Observations,
        Mapping[str, float],
        Mapping[str, bool],
        Mapping[str, bool],
        Mapping[str, Any],
    ]:
        ...


class ActionPolicy(Protocol):
    """Port for policies used by the episode application service."""

    def actions(self, observations: Observations, frame: int) -> Actions:
        ...


class EpisodeRunner:
    """Run an episode without knowing Gymnasium, Torch, YAML, or storage."""

    def __init__(self, environment: EpisodeEnvironment, policy: ActionPolicy):
        self._environment = environment
        self._policy = policy

    def run(self, spec: EpisodeSpec) -> EpisodeResult:
        observations, initial_info = self._environment.reset(spec.seed)
        frames = []

        for frame_index in range(spec.max_frames):
            actions = self._policy.actions(observations, frame_index)
            observations, rewards, terminated, truncated, info = (
                self._environment.step(actions)
            )
            frame = FrameResult(
                index=frame_index,
                observations=observations,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            frames.append(frame)
            if (
                terminated.get("__all__", False)
                or truncated.get("__all__", False)
            ):
                break

        return EpisodeResult(
            seed=spec.seed,
            frames=tuple(frames),
            initial_info=initial_info,
        )

