# SCooP-Bed: testbed paper (Paper A, split from the SCooP draft)

SCooP-Bed: An Indoor Multi-Robot Testbed for Collaborative Perception over
Measured Wireless Links. Target: RA-L (6 pages + 2 overlength).

Structure follows `../scoop_paper/split/PAPER_A.md`; the experiments E1-E5
follow `../scoop_paper/split/EXPERIMENTS.md` (pass criteria and the results
recorded there so far are already in the text).

Owns: testbed design and design decisions, time and space alignment (clock,
sniffer bias, error budget, extrinsics), ground truth without motion capture
(map, fiducial chain, validation, scoring capability), link instrumentation,
channel-in-the-loop replay, experiments E1 Headroom / E2 Map Meets Radio /
E3 Twin Run / E4 Same Knob Same Link / E5 Sim Gap, lessons learned.
Releases no sequences, labels, or leaderboard (those are Paper B).

Companion: `../scoop_dataset_paper` (cites this paper as `scoop_testbed`;
this paper cites it as `scoop_dataset`).

- `main.tex`: manuscript (IEEEtran conference template for drafting)
- `references.bib`: SCooP bib + `../scoop_paper/split/testbed_refs.bib` + companion entry
  (note: `itup530` is duplicated in the original bib; remove one copy before submission)
- `tables/`: sensors, topics, ntp, calib, gt from the draft; new: `spec.tex`
  (spec sheet, the central table), `capability.tex`, `bom.tex`
- `figures/`: overview, platforms, refpipeline, refmap, gt_errors, linktrace,
  dataflow from the draft; new placeholders: `replay.tex`, `headroom.tex`,
  `clearance.tex`, `twinrun.tex`, `simgap.tex`
- `figs/`: images

Upload the folder to Overleaf as is; `main.tex` is the root file.
The original single-paper draft in `../scoop_paper` is untouched.
