# SCooP dataset paper (split from the SCooP draft)

SCooP: A Real-World Indoor Multi-Agent Dataset for Collaborative SLAM and
Perception over Measured Wireless Links. Target: IEEE T-RO regular paper,
IEEEtran journal class, CODa-style (findings as questions Q1-Q4 with one
controlled experiment each; full training tables in the appendix). Budget
about 12 pages + references; check the current T-RO overlength policy.

Dataset/benchmark paper. Owns: the two questions (link cost, foreseeable
share), related datasets and comparison table, condensed platform/reference
summary (details deferred to the testbed paper), annotations, collection
design and collaboration-geometry characterization, release formats and
devkit, benchmarks (T1/T2 protocol, channels, replay), experiments (Q1
transfer, Q2 heterogeneity, Q3 where the gain comes from, Q4 link cost and
foreseeable share, failure cases), discussion, appendices A-D.

Companion: `../scoop_testbed_paper` (cited as `scoop_testbed`).

- `main.tex`: manuscript (IEEEtran journal class, T-RO header); page budget per section in the header comment
- `references.bib`: bibliography (copy of the SCooP bib + companion entry)
- `tables/`, `figures/`: `.tex` includes; `figs/`: images
- New in this split: `tables/agents.tex`, `tables/gt_summary.tex`,
  `tables/res_transfer.tex` (Q1), `tables/res_hetero.tex` (Q2),
  `figures/gain_strata.tex` (Q3), `figures/replay.tex`, `figures/stats.tex`

Upload the folder to Overleaf as is; `main.tex` is the root file.
The original single-paper draft in `../scoop_paper` is untouched.
