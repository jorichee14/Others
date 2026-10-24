# SCooP-iT: final direction

> **24 Oct:** the section plan and the work plan are now `WRITING_GUIDE.md`
> (section by section) and `../PLAN.md` (train in the Gym, test on the
> recordings, deploy on the testbed). The paper is split into
> `sections/s00`-`s11`. Sections 1-6 below are the history of the decisions.

This file fixes the venue, the claim, the structure, and the closed list of
work left before submission. Anything not on the list in Sec. 5 is out of
this paper.

## 1. Venue: IEEE IoT-J (this folder). The RA-L draft is frozen.

- **The claim belongs to IoT-J.** "The network decides whether the
  application succeeds, so measure the network where the application runs"
  is a networking claim. The evidence that earns trust for it is a
  live-vs-replay error number, as in Noble, Mahimahi, Pantheon and
  Colosseum-as-twin. IoT-J reviewers know that evidence. RA-L reviewers
  would judge the paper as a dataset against CoPeD and Kimera-Multi.
- **Length.** RA-L allows 6 + 2 pages including references. Compiled, this
  draft was 18 pages before the cuts in Sec. 3 and is 15 after them,
  including 2.5 pages of references (89 entries). The "~9.4 pages" in the
  old header comment was an estimate, not a compile. Check the current
  IoT-J length policy in the Author Portal; the remaining gap is closed by
  condensing prose, not by cutting more content.
- **No reviewer overlap with the companion paper.** The T-RO dataset
  paper takes the robotics audience, and the two papers stop competing
  for the same reviewers.
- **Risk:** IoT-J reviewers will say that five nodes is a small IoT
  testbed. The answer, stated once in Limitations: the contribution is
  measurement fidelity per link, not node count, and the design scales
  per AP.

`../scoop_testbed_paper/` stays as it is, for reference only. No more edits there.

## 2. One claim

**Title (keep):** *SCooP-iT: An Indoor IoT Testbed for Measuring What a
Shared Wi-Fi Network Does to Multi-Agent Perception*

**Headline:** each inter-node link is recorded on the sensors' clock and in
the sensors' map frame, and it is replayed message by message into any
collaborative application, **with a measured replay error (E3)**.

**Framing (added 23 Oct):** networked embodied intelligence. Two surveys say
that link indicators (latency, reliability, AoI) and task outcomes must be
measured together, and both list integrated testbeds with real wireless
dynamics and heterogeneous agents as an open problem: MEAN, arXiv:2610.26297,
Sec. II-B and Sec. VII "Benchmarking and evaluation methodology"; Wen et al.,
TechRxiv 2026, Sec. V-A. SCooP-iT is that testbed for the perception half of
the loop. The data already show the surveys' point: on coop_2 the loaded RTT
(25-33 ms) is well inside the 100 ms frame, yet only 59-68 % of 1.5 MB
messages land in it. This is now contribution 4, the abstract's last
result, and a paragraph in E5. Scope stated in Limitations: the loop is open
(the nodes do not act on what they receive).

Everything else supports that headline, and none of it is a headline of
its own:

| Role | Content | Why not the headline |
|---|---|---|
| Enabler | One clock (<= 0.99 ms measured), one frame without mocap or GNSS | CoPeD and Kimera-Multi already have mocap-free GT. Here it is the precondition for one-way delay and for E2 |
| Evidence the design yields | E2: the map predicts the radio (rho 0.60, 5.4 dB) | Original, but rho 0.60 supports a claim; it can't carry one |
| Why it matters | E5: synthetic channels deliver 100 % in-frame, measured 86 % alone and 59-68 % with two robots | "Real links hurt" is already CooperScene's claim |
| Trust | E1 headroom, E4 repeatability | Necessary checks, not novelty |

**E3 is mandatory.** CooperScene already has "measured link beside
sensors", so without E3 the paper has no claim nobody else has. It costs
one session of about an hour (Sec. 5B).

The seven questions stay as the organising device in the introduction and
in Sec. VIII, but as text only (see Sec. 3).

## 3. Structure and cuts (done 23 Oct: 18 -> 15 compiled pages)

| Change | Saves | Where it goes |
|---|---|---|
| Remove Sec. V-E synthetic views (Isaac Sim) and `tab:twin` | 0.5 | The twin came back on 24 Oct inside Sec. IX (twin mode) and E3b, a paragraph shorter than V-E was |
| Remove the seven `results_q*.tex` tables | 1.0 | They are the working ledger. Keep the files; do not `\input` them |
| Fold `tab:questions` into `tab:spec`: one full-width table with columns Question, Guarantee, Bound, Validated by, Pass criterion | 0.5 | The "Answer now / Remaining" columns cannot appear in a submitted paper |
| Move `tab:gap_fill` (gaps by discipline) to the supplementary | 0.4 | `tab:gap_iotj` plus the last paragraph of Sec. II already say it |
| Move `tab:refmap_stages`, `tab:glim_vs_ref` and the BA-vs-scan-to-map paragraph to the supplementary; keep one paragraph and `tab:gt` in Sec. V-A | 0.3 | `supp/refmap_method.tex` |

The four contributions stay: (1) the testbed, with link and CSI on the
sensor clock; (2) mocap-free GT, the enabler; (3) the per-message replay,
validated live (E3); (4) link indicators do not predict the task outcome
(E5).

## 4. Consistency fixes (the reference changed, but parts of the text did not)

The reference map is now the 20261015 `mobile_2` pass. The 20260924
`mobile_1` pass is the cross-robot check. Several numbers still come from
the old `mapping_A` reference:

| Where | Stale | Final |
|---|---|---|
| Abstract, contribution 2, Sec. V-D, E1, E4, conclusion, `tab:gt`, `tab:spec` | 1.06 cm surface agreement and 2e-5 scale (mapping_A vs the Aug map) | Choose one: either rerun `check.py maps --align` between the two final passes for a cm figure, or quote 96.4 % within 10 cm (fit RMS 3.0 cm). Do not mix the two references |
| `tab:gt` thickness seq. A | 7.9 / 2.1 cm | 7.6 / 1.8 cm (20261015 TSDF) |
| `tab:gt` refinement | 0.15 cm, 0.017 deg | BA vs scan-to-map: 1.3 cm ATE, 0.15 deg |
| Abstract "6-133x", `tab:questions` "133x" | 133x (from 0.15 cm) | 15x (0.2 m / 1.3 cm), as E1 already states. The abstract range becomes 6-101x |
| Sec. VIII opening | "mapping pass `mapping_A`" | 20261015 reference + 20260924 cross-check |
| RTT idle/loaded | Two passes quoted inconsistently | Quote 24 Sep (5-10 to 25-33 ms) as primary; 0828 (3.3 to 60-109 ms) as the earlier pass, with its load |

## 5. What is left (closed list)

### A. Compute only, from the existing passes (`RUN_NOW.md` steps 1-5)

1. Confirm which pass produced the CSI rate, capacity and chrony RMS numbers.
2. Board residual for `mobile_1` on coop_2, p95 speed, and the E1 headroom figure.
3. E2: bootstrap CI on rho, pooled over survey_1 and coop_2.
4. E4 mode half: `link_repeat.py survey_1 coop_2`.
5. E5: best-fit synthetic baseline in `sim_gap.py`.
6. Calib table: LiDAR-camera reprojection; radar reflector residuals if the
   reflector data exist. If they don't, the radar rows say "not validated"
   and Limitations says so.
7. Error budget: one latency figure per sensor class, then compute
   `\delta x_max`.

### B. One recording session (about 1 h, same route as survey_1)

A1/B1 survey, A2/B2 cooperation, plus a probe-off run. This gives E3 (the
headline), the E4 day half, the E5 probe-off control, and the sniffer bias.
Before the session: the payload node and `twin_run.py`.

### C. Bookkeeping

BOM costs and setup effort; the cause of the 0.49 m frame offset; the board
split; control distances, only if rangefinder data were logged (otherwise
drop the claim and state absolute scale as a limit); `mobile_2` radar yes or
no; the names of the released modules; funding, second e-mail,
acknowledgments; re-check the `tab:gap_iotj` cells against the PDFs.

### A2. Added 24 Oct: the twin and the gym

8. `CommSchedule-v0` on coop_2 and survey_1: `python -m scoop_gym.run
   ../scoop_gym/scenarios/comm_schedule_coop_2.yaml --links ideal v2xvit
   fit:<pass> logged:<pass>`. That fills the `\todo` in Sec. IX (in-frame
   share, deadline rule vs. always clouds). Compute only.
9. E3b: the twin run of coop_2 (`isaac_coop_sim`), scan-to-map agreement of
   the synthetic scans at the real poses, then one cooperative detector on
   real and twin data under the ideal and logged channels. The detector run
   needs the dataset paper's labels; see `../PLAN.md` for the order.

### D. Figures

Fig. 1 redraw, headroom (E1), clearance (E2), simgap two-panel (E3 + E5),
and the link trace with RTT and CSI added.

### Out of this paper (not to be added back)

The closed-loop twin and the positional link (the gym paper, `../PLAN.md`);
the twin tasks of Table `tab:gym`; out-of-run E2 prediction (it validates
the positional link, so it belongs to the gym paper); a second site; CSI
power as a second E2 signal; agent-to-AP E2 links; more nodes; other
radios. The last three go in one sentence under next steps.

## 5b. Section plan, 24 Oct (16 compiled pages)

I Introduction (IoRT, then the surveys, then the indoor problem, then
SCooP-iT and four contributions); II Related Work (adds Lingjing, ideal
bus); III Design; IV Alignment; V Ground Truth; VI Link; VII Replay;
VIII Experiments (E3 = E3a live, E3b twin); IX SCooP-iT Gym (link engine,
two engines, tasks, `CommSchedule-v0` demonstration, `tab:gym`);
X Lessons and Limitations (the loop closes only in the twin); XI Conclusion.

## 6. If E3 cannot be recorded before the deadline

The headline falls back to "a synchronised, map-anchored measured link,
and what it shows (E2, E5)". Contribution 2 becomes "a replay path whose
inputs are measured", and the abstract says that the output is not yet
validated. This is a weaker paper. Treat it as the fallback, not the plan.
