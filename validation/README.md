# Experimental benchmarks used to check the reference solver

* `piercaps_geevar_menon_2018.json`: the five concentrically loaded pier caps
  S1-S5 of Geevar and Menon (2018) as documented in Kaufmann et al. (2020),
  *Compatible Stress Field Design of Structural Concrete*, Section 6.5:
  geometry, reinforcement, mean measured materials, measured failure loads and
  the book's CSFM predictions. Run `npm run validate:piercaps` in `solver/`.
* The eight full-scale deep beams of Li, Wu, Zhang and Xie, *Materials* 15
  (2022) 6017 (open access) are encoded directly in
  `solver/scripts/validateLiBeams.ts` (Tables 1-3 of Li et al.). Run
  `npm run validate:deepbeams`.
