"""Synthetic statistical gate, not an OTFS range/performance certificate."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from uav_isac.physical.multiframe_glrt import trajectory_bank_scores


def audit(samples=50000):
    rows=[]
    for window in (1,2,4,8):
        # Two predefined complete trajectories in two spatial modes per frame.
        # Trajectory 0 stays in mode 0; trajectory 1 alternates. No truth-based
        # bank adaptation. Their common first frame intentionally duplicates W=1.
        designs=[]
        for route in (0,1):
            s=np.zeros((2*window,window),complex)
            for t in range(window): s[2*t+(t%2 if route else 0),t]=1.
            designs.append(s)
        covariance=np.eye(2*window)
        def noise(seed):
            rng=np.random.default_rng(seed)
            return (rng.normal(size=(samples,2*window))+1j*rng.normal(size=(samples,2*window)))/np.sqrt(2)
        null,_,_=trajectory_bank_scores(noise(510000+window),designs,covariance)
        threshold=float(np.quantile(null,.999,method='higher'))
        held,_,_=trajectory_bank_scores(noise(520000+window),designs,covariance)
        rng=np.random.default_rng(530000+window)
        phases=np.exp(2j*np.pi*rng.random((samples,window)))
        h1noise=noise(540000+window)
        for budget in ('fixed_energy_per_frame','fixed_total_energy'):
            total=4.*window if budget=='fixed_energy_per_frame' else 4.
            signal=(phases*np.sqrt(total/window))@designs[1].T
            scores,_,ranks=trajectory_bank_scores(h1noise+signal,designs,covariance)
            rows.append(dict(window=window,budget=budget,total_signal_energy=total,
                pd=float(np.mean(scores>threshold)),pfa=float(np.mean(held>threshold)),
                threshold=threshold,ranks=ranks.tolist()))
    return dict(rows=rows,samples_per_split=samples,physical_detection_certified=False,
        limitations=['synthetic orthogonal observation modes; no OTFS geometry or 800 m claim',
            'single terminal decision per window, globally calibrated over two full trajectories',
            'deterministic unknown frame amplitudes; random test phases only generate test cases',
            'independent noise in this gate; no radio transport or power-to-energy mapping'])


if __name__=='__main__': print(json.dumps(audit(),indent=2))
