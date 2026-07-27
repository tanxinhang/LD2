# Formal paper experiment suite

## Protocol

Top-1/Top-2/Top-4 and all structural ablations use the same 100 fixed test seeds. Perturbations use 50 seeds. SNR and deadline changes are paired with the first 50 nominal seeds; hard geometry comes from a different stress distribution and is therefore reported without a paired significance claim. QoS feasibility requires steady >= 0.80, weak3 >= 0.70, and worst >= 0.60 in the same episode.

## Formal Top-k comparison (100 matched seeds)

| Variant | N | steady | weak3 | worst | LCB | CVaR20 | QoS | QoS-LCB | bit/frame | latency/ms | delivery | Medium |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-1 (deployment) | 100 | 0.9498 | 0.9331 | 0.8186 | 0.7671 | 0.2650 | 0.8300 | 0.7597 | 768.0000 | 0.7715 | 1.0000 | PASS |
| Top-2 (structural anchor) | 100 | 0.9452 | 0.9269 | 0.7984 | 0.7497 | 0.2882 | 0.7800 | 0.7050 | 1280.0000 | 1.1524 | 1.0000 | PASS |
| Top-4 (broadcast control) | 100 | 0.9250 | 0.9000 | 0.7291 | 0.6683 | 0.1259 | 0.7000 | 0.6202 | 2304.0000 | 1.9147 | 1.0000 | PASS |

Top-1 minus Top-2 worst = +0.0202, paired bootstrap 95% CI [-0.0132, +0.0540]. The QoS feasibility difference is +0.05 (6 Top-1-only versus 1 Top-2-only successes; exact McNemar p=0.1250). Top-1 reduces bits/frame by 40.0% and mean latency by 33.1%.

Interpretation: Top-1 is the communication-efficient deployment variant. Top-2 remains the structural-ablation anchor because all component ablations are matched to it. The mean QoS differences between Top-1 and Top-2 are not statistically resolved by 100 seeds.

## Core baselines and ablations (100 matched seeds)

| Variant | N | steady | weak3 | worst | LCB | CVaR20 | QoS | QoS-LCB | bit/frame | latency/ms | delivery | Medium | Delta worst vs full [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full top-2 | 100 | 0.9452 | 0.9269 | 0.7984 | 0.7497 | 0.2882 | 0.7800 | 0.7050 | 1280.0000 | 1.1524 | 1.0000 | PASS | -- |
| No U2U communication | 100 | 0.9177 | 0.8903 | 0.7055 | 0.6443 | 0.0881 | 0.6700 | 0.5891 | 0.0000 | 0.0000 | 1.0000 | PASS | -0.0930 [-0.1424, -0.0454] |
| Zero token payload | 100 | 0.9332 | 0.9109 | 0.7675 | 0.7187 | 0.2567 | 0.7300 | 0.6516 | 1280.0000 | 1.1467 | 1.0000 | PASS | -0.0309 [-0.0688, +0.0067] |
| Permuted sender identity | 100 | 0.9137 | 0.8849 | 0.7041 | 0.6512 | 0.1607 | 0.6800 | 0.5994 | 1280.0000 | 1.1295 | 1.0000 | PASS | -0.0943 [-0.1368, -0.0538] |
| No comm-to-sensing residual | 100 | 0.9435 | 0.9247 | 0.7939 | 0.7464 | 0.2844 | 0.7800 | 0.7050 | 1280.0000 | 1.1522 | 1.0000 | PASS | -0.0045 [-0.0111, +0.0000] |
| No capacity-two matching | 100 | 0.8554 | 0.8072 | 0.5362 | 0.4838 | 0.0788 | 0.3700 | 0.2950 | 1280.0000 | 1.1338 | 1.0000 | FAIL | -0.2622 [-0.3113, -0.2147] |
| No movement consensus | 100 | 0.8599 | 0.8133 | 0.5429 | 0.4911 | 0.0850 | 0.4100 | 0.3325 | 1280.0000 | 1.1369 | 1.0000 | FAIL | -0.2555 [-0.3020, -0.2099] |
| Projected-only capacity bid (beta=0) | 100 | 0.9188 | 0.8918 | 0.7011 | 0.6374 | 0.0540 | 0.6700 | 0.5891 | 1280.0000 | 1.1528 | 1.0000 | PASS | -0.0973 [-0.1597, -0.0389] |
| Intrinsic-only semantic bid (beta=1) | 100 | 0.9046 | 0.8728 | 0.6844 | 0.6348 | 0.2384 | 0.5500 | 0.4679 | 1280.0000 | 1.1538 | 1.0000 | PASS | -0.1140 [-0.1582, -0.0698] |
| Fixed 25% communication power | 100 | 0.9474 | 0.9299 | 0.8074 | 0.7594 | 0.3024 | 0.7900 | 0.7158 | 1280.0000 | 1.1520 | 1.0000 | PASS | +0.0090 [-0.0002, +0.0245] |
| Central movement oracle (upper bound) | 100 | 0.9870 | 0.9827 | 0.9499 | 0.9264 | 0.7497 | 0.9200 | 0.8635 | 1280.0000 | 1.1900 | 1.0000 | PASS | +0.1515 [+0.0903, +0.2143] |

The central movement oracle is an upper bound, not a deployable baseline. Movement consensus and capacity-two matching are the two largest deployable structural contributions. The hybrid bid also outperforms either projected-only or intrinsic-only bidding. Fixed 25% communication power does not underperform the learned total split, so the dynamic total communication/sensing split is not supported as a core claim. The direct communication-to-sensing residual has only a small effect.

## Perturbation and robustness experiments

| Variant | N | steady | weak3 | worst | LCB | CVaR20 | QoS | QoS-LCB | bit/frame | latency/ms | delivery | Medium |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nominal top-1 | 100 | 0.9498 | 0.9331 | 0.8186 | 0.7671 | 0.2650 | 0.8300 | 0.7597 | 768.0000 | 0.7715 | 1.0000 | PASS |
| Hard geometry | 50 | 0.8562 | 0.8083 | 0.5397 | 0.4566 | 0.0480 | 0.4600 | 0.3491 | 768.0000 | 0.7743 | 1.0000 | FAIL |
| SNR threshold = 35 dB | 50 | 0.9412 | 0.9216 | 0.7859 | 0.7067 | 0.1903 | 0.7800 | 0.6707 | 768.0000 | 0.7695 | 0.8780 | PASS |
| Deadline = 1.0 ms (light control) | 50 | 0.9468 | 0.9291 | 0.8188 | 0.7443 | 0.2819 | 0.8200 | 0.7150 | 768.0000 | 0.7702 | 1.0000 | PASS |
| Deadline = 0.8 ms | 50 | 0.9135 | 0.8847 | 0.7030 | 0.6151 | 0.0859 | 0.7000 | 0.5854 | 768.0000 | 0.7618 | 0.5667 | PASS |

At the calibrated 0.8 ms deadline, delivery falls to 0.567 and worst to 0.703; the average Medium thresholds remain satisfied, but tail robustness degrades. Under hard geometry, worst falls to 0.540, which fails the Medium worst threshold. This is the present generalization boundary and must be stated explicitly.

Paired with the first 50 nominal seeds, raising the SNR threshold to 35 dB changes worst by -0.0355 (95% bootstrap CI [-0.0774, -0.0045]); the 1.0 ms control changes it by -0.0026 [-0.0111, 0.0050], whereas the active 0.8 ms deadline changes it by -0.1183 [-0.1837, -0.0610].

## Paper-facing conclusion

The frozen evidence supports a distributed sparse-token U2U-ISAC system whose essential mechanisms are capacity-aware matching and communication-induced movement consensus. Nominal 100-seed results meet the requested Medium average thresholds. The evidence does not yet support strong claims for a learned total power split, a large direct sensing-residual gain, or robust generalization to hard geometry.
