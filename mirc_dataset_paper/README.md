# MIRC collaborative-perception dataset paper — sections

LaTeX source for the dataset paper's Contribution 3 (trajectory paradigms) and the
results/discussion that support it.

| File | Contents |
|---|---|
| `sections/collection.tex` | §Dataset Design and Collection: environments, roles, trajectory paradigms (predicates + Table), standing parameters, inventory |
| `sections/results.tex` | §Results: canonical sequence table, validation table, three findings |
| `sections/discussion.tex` | §Discussion: what measurement buys, evidence-definition problem, threats to validity, next collection |

## Packages assumed by the preamble

`booktabs`, `amsmath`, `amssymb` (`\checkmark`), `siunitx`, `todonotes`
(`\todo`, `\unsure` as already used in the draft). `results.tex` uses `table*`,
so a two-column class is assumed; in a one-column class change it to `table`.

## Provenance of every number

All values trace to `characterize_stage1.py` / `characterize_stage2.py` run on
`coop2_0828`, once per ego assignment:

```bash
python3 characterize_stage2.py --bag $BAG --map $MAP --objects chairs.json \
    --meta metadata.yaml --out out/coop2_m1 --ego mobile_1
```

- Table (sequence characterization) ← `out/coop2_m{1,2}/table_row.csv`
- Occlusion episodes, rescue counts ← `episodes.csv`
- Intended-vs-measured paradigm agreement ← `summary.json:paradigm_distribution`
- Validation correlations ← `summary.json:validation_rho`, `summary.json:link.validation`
- Finding 1 stratification ← `summary.json:range_binned`, `rescue_by_range`
- Finding 2 (0.008 vs 0.29) ← `summary.json:range_rescue_rate.{primary,rgbd}`
- Finding 3 (β, Δh) ← `summary.json:beta_binned`, stage-1 `pairwise.csv`

Empty cells in the sequence table (Doppler, stress) mean *not computable on this
sequence*, not zero. Paradigms 2 and 5 are computed as metrics (`link.csv`) but
are not yet emitted as labels by the per-frame classifier; the status column of
the paradigm table says so.
