# Experiments E1-E8

Each script imports `common.py`, which puts `../pinn` on the path and runs
from there, so the data and checkpoint paths resolve unchanged. Each writes
`<name>.log` and `<name>.json` here.

| ID | Script | What it measures |
|---|---|---|
| E1 | `e1_config_change.py`, `e1_plot.py` | retraining for a two-point-load deep beam (8 nodes, 13 members): from scratch vs warm start |
| E2 | `e2_weq_sweep.py` | equilibrium-weight sweep, w_eq = 0 to 100, three seeds, four archetypes |
| E3 | `e3_equilibrium_projection.py` | least-squares projection to exact equilibrium; rank of the equilibrium matrix |
| E4 | `../solver/scripts/validateLiBeams.ts`, `e4_plot.py` | solver vs the eight deep beams of Li et al. (2022), log in `e4_li_beams.log` |
| E5 | `e5_censored_loss.py` | ceiling label vs one-sided hinge for non-failing designs |
| E6 | `e6_width_sweep.py` | network width 16 to 256: train and test error |
| E7 | `e7_force_conformal.py` | split-conformal intervals on every member force |
| E8 | `e8_ood_detection_curve.py` | out-of-domain detection: flag rate, AUROC, recall of harmful errors |
