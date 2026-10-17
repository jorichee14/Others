# SCooP-iT: testbed paper (Paper A, split from the SCooP draft)

SCooP-iT: An Indoor Multi-Robot Testbed for Collaborative Perception over
Measured Wireless Links. Target: IEEE RA-L, IEEEtran journal class, 6 pages +
at most 2 overlength including references. No appendix: `supplementary.tex`
(compile separately) holds the full sensor table, topics, clock table,
reference map, GT error figure, data flow, intrinsics, and setup effort.

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

- `main.tex`: manuscript (IEEEtran journal class, RA-L header); page budget per section in the header comment
- `supplementary.tex`: RA-L multimedia attachment root
- `references.bib`: SCooP bib + `../scoop_paper/split/testbed_refs.bib` + companion entry
  (note: `itup530` is duplicated in the original bib; remove one copy before submission)
- `tables/`: `sensors_compact.tex`, calib, gt, bom in main; sensors, topics, ntp in supplementary;
  new: `spec.tex` (spec sheet, the central table), `capability.tex`
- `figures/`: overview, platforms, refpipeline, linktrace in main; refmap, gt_errors,
  dataflow in supplementary; new placeholders: `replay.tex`, `headroom.tex`,
  `clearance.tex`, `simgap.tex` (two panels: E3 and E5)
- `figs/`: images

Upload the folder to Overleaf as is; `main.tex` is the root file.
The original single-paper draft in `../scoop_paper` is untouched.

IoT-J variant of Sec. II: `iotj/related_work_iotj.tex`, organized by
discipline (wireless communications, robotics, computing and networked
systems, perception, testbeds), each ending with what that community
assumes about the others. `tables/gap_iotj.tex` is the capability table
with columns grouped by discipline; `tables/gap_fill_iotj.tex` gives each
discipline's gaps (W1-W3, R1-R3, C1-C3, P1-P2, T1-T2), what SCooP-iT
provides, and the evidence. To use it, replace Sec. II of `main.tex` with
`\input{iotj/related_work_iotj}`. New bib entries are at the end of
`references.bib` under "IoT-J additions".
