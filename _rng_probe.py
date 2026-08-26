"""Runtime probe: does env.reset(seed=...) rebind the cost-aware comm RNG?

Audit finding candidate: env_core.reseed() rebinds self.rng, action_space.rng and
deflection_computer.rng but NOT the InterUAVCommunicationModel generator, so with
comm_finite_blocklength_enabled / burst / shadowing the draws follow the stale
constructor stream.  Replaying the same seed twice on the same env must produce
identical channel Markov states if reseeding is complete.

Usage:  pytrch_ven\\Scripts\\python.exe _rng_probe.py
"""

import sys

import numpy as np

sys.path.insert(0, "D:/BYLW/LD3")

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv


def _roll(env, frames: int) -> None:
    for _ in range(frames):
        obs = env.current_obs if hasattr(env, "current_obs") else None
        if obs is None:
            break
        acts = {k: {"delta_p": np.zeros(2), "role": 0} for k in obs}
        env.step(acts)


def main() -> None:
    cfg = load_config(
        "D:/BYLW/LD3/config/exp_800_k12q12_distributed_v2_fbl_shadow4ms.yaml")
    env = UAVISACEnv(cfg)
    core = env.core
    comm = core._inter_uav_comm

    print("--- diagnostics ---")
    print("comm model created:", comm is not None)
    print("comm mode:", core._comm_mode)
    print("fusion mode:", core._detection_fusion_mode)
    if comm is None:
        print(">>> cannot probe: comm model not created for this config.")
        return
    shadowing_std = getattr(comm, "snr_shadowing_std_db", 0.0)
    burst_enabled = getattr(comm, "burst_loss_enabled", False)
    fbl_enabled = getattr(comm, "finite_blocklength_enabled", False)
    print("shadowing std:", shadowing_std, "burst:", burst_enabled,
          "FBL:", fbl_enabled)

    # Episode 1: reset(seed=91) then roll 3 frames.
    env.reset(seed=91)
    _roll(env, 3)
    burst_a = comm._burst_bad_state.copy()
    shadow_a = comm._snr_shadowing_db.copy()
    print("after ep1: comm.rng is core.rng ->", comm.rng is core.rng)
    print("shadow nonzero (ep1):", int(np.count_nonzero(shadow_a)))

    # Episode 2 on the SAME env: reset(seed=91) again.  Replay contract says
    # the channel stream must be back at the same position as episode 1.
    env.reset(seed=91)
    _roll(env, 3)
    burst_b = comm._burst_bad_state.copy()
    shadow_b = comm._snr_shadowing_db.copy()
    print("after ep2: comm.rng is core.rng ->", comm.rng is core.rng)
    print("shadow nonzero (ep2):", int(np.count_nonzero(shadow_b)))

    bursts_equal = np.array_equal(burst_a, burst_b)
    shadows_equal = np.array_equal(shadow_a, shadow_b)
    print("burst  identical across same-seed resets:", bursts_equal)
    print("shadow identical across same-seed resets:", shadows_equal)
    if shadows_equal and bursts_equal:
        print(">>> PASS: channel stream replays on reset(seed=...) (no leak).")
    else:
        print(">>> CONFIRMED: cost-aware comm RNG NOT rebound by reset(seed=...);")
        print("    shadow/burst draws follow the stale constructor stream,")
        print("    so same-seed replay across episodes diverges.")


if __name__ == "__main__":
    main()