# SCooP-iT: IEEE Internet of Things Journal draft

Full draft of the testbed paper for IEEE IoT-J, built from
`../scoop_testbed_paper` (the RA-L draft, left unchanged). Upload this
folder to Overleaf as is; `main.tex` is the root file and
`supplementary.tex` compiles separately.

Title: *SCooP-iT: An Indoor IoT Testbed for Measuring What a Shared Wi-Fi
Network Does to Multi-Agent Perception*

## What differs from the RA-L draft

- Framing: robots and masts are mobile and fixed IoT nodes on one
  contended Wi-Fi network; cooperative perception is the application whose
  outcome the network decides (title, abstract, keywords, introduction,
  conclusion, a sentence in Sec. III).
- Sec. II Related Work: IoT testbeds, network timeliness (802.11
  contention, age of information, task-oriented communication, edge,
  middleware, clock sync, trace emulation), channel measurement, robots
  as IoT nodes, cooperative perception as the application, the gap.
- Table II (`tables/related_iotj.tex`, label `tab:gap_iotj`): 22 related
  platforms with venue, nodes, indoor, measured link, CSI, pose
  reference, clock, replay, and scoring. Every cell was checked on
  18 Oct 2026 against official repos, project docs, and excerpts of the
  papers; arXiv and IEEE Xplore were not reachable, so re-check against
  the PDFs before submission.
- Table III (`tables/gap_fill_iotj.tex`): gaps by discipline (wireless,
  robotics, computing, perception, testbeds), what SCooP-iT provides,
  evidence.
- Clock claim: other platforms state sensor-sync figures (CooperScene
  claims sub-ms PTP, TUMTraf-V2X 24.9 ms, V2V4Real < 50 ms); SCooP-iT's
  claim is a *measured* link-to-sensor bound.
- `tables/results_q6.tex` and `tables/questions.tex` updated to the
  checked cells; the RA-L capability table is not used.
- `references.bib`: duplicate `itup530` removed; IoT-J entries and the
  checked bib fixes included.

## Still open (search `\todo` in main.tex)

- E3 replay vs live and E4 day repeatability need the payload-node
  session (see `../scoop_testbed_paper/RESULTS_STATUS.md`).
- Sniffer bias, error-budget total, control distances, cross-robot
  consistency, funding, acknowledgments.
- Reference map: the BA + TSDF write-up is in `sections/refmap_main.tex`
  (with `supp/refmap_method.tex`); swap it into Sec. V-A once its numbers
  are in, and update E1 and Sec. IX to match.
- Sec. V-E `sections/synthetic_views.tex` (Table `tab:twin`): synthetic sensor
  views from the reference map in Isaac Sim (`../isaac_coop_sim`). The
  sim-vs-real scan agreement and the detection gap are `\todo` until the
  runs exist; the tooling itself is not yet run in Isaac Sim.
- Length is estimated at about 9.4 pages (about 9.9 with Sec. V-E); IoT-J charges beyond 8. Fold
  the per-question result tables into the spec sheet to save about a page.
