# Datasets

All files are gzipped JSON; the Python loaders read `*.json.gz` directly
(`pinn/data.py: load_json`). `gunzip -k <file>` gives the plain JSON.

| File | Contents |
|---|---|
| `dataset.json.gz` | the training data: 750 Latin-hypercube designs for each of `deepBeam`, `hammerhead`, `multiColumnBent`, `pileCap` and `pierCap`, analysed by the reference solver (seed 1) |
| `dataset_extrap_0.1.json.gz`, `_0.2`, `_0.3` | out-of-domain shells at distance delta = 0.1, 0.2, 0.3 of the parameter range outside the training box, same solver and settings |
| `dataset_deepBeam2P.json.gz` | a changed loading configuration: deep beam under two loads at the quarter points (8 nodes, 13 members), 750 designs |

## Record schema (`designs[]`)

```text
id                   "deepBeam-0000"
archetype            archetype name
params               design parameters (mm, N, MPa), the surrogate's input
truss.nodes[]        {id, x, y, z, fixed[3], label}
truss.members[]      {id, ni, nj, kind, area, E, concreteArea, steelArea, barDia}
truss.loads[]        {node, fx, fy, fz}           reference load set (N)
labels.failureLoadFactor   lambda_f (3.0 = did not fail within the analysed range)
labels.failureMode         solver's governing mechanism (text)
labels.memberForces        member forces at the linear static solution (N)
labels.curve[]             {loadFactor, displacement} load path
labels.csfmStates[]        up to 12 sampled states {loadFactor, strains{}, forces{}};
                           the last one is the failure state (the force-head target)
```

Regenerate with `cd solver && npm install && npm run dataset` (about a second
per 750 designs); the sweep is deterministic for a given seed.
