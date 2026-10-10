# Robotics and Multi-Robot Testbed Papers: Headline Claims and How Each Validated Itself as an Instrument

Context: these notes support an IEEE RA-L paper on an indoor heterogeneous multi-robot testbed. The testbed has a Scout Mini with LiDAR, stereo, mmWave radar and a Nexmon Wi-Fi CSI sniffer; a pushcart with a RealSense camera and visual SLAM; infrastructure masts; and one Wi-Fi AP. Every link is measured on one clock and one map frame, ground truth comes without mocap or GNSS, and the logged link is replayed into collaborative perception.

Method note: arxiv.org, doaj.org and caltech.edu could not be fetched from this environment (DNS failure), so these notes rely on search-result abstracts and metadata rather than full-text reads. Items marked "(unverified, from prior knowledge)" did not come up in a search during this session. Check them before citing.

## Q1. Robotarium (Pickem et al. ICRA 2017; Wilson et al. CSM 2020; Wilson & Egerstedt OJ-CSYS 2023): headline, safety barrier certificates, usage statistics, how they proved value

### Takeaway
The Robotarium's headline claim is **accessibility**: a free, remotely accessible swarm testbed for people who cannot afford to build one. Its key technical move is to put safety into the design itself, using control barrier certificates that minimally change user commands. It showed its value through demonstrations (robots swapping positions, collision-free trajectories) and later through **usage statistics** (hundreds of user groups on every continent but Antarctica). A traditional accuracy benchmark was not the main evidence.

### Cited Findings
- **Paper and venue:** "The Robotarium: A remotely accessible swarm robotics research testbed," arXiv 1609.04730, presented at ICRA 2017. It won the ICRA 2017 Best Multi-Robot Systems Paper Award. — [arXiv PDF](https://arxiv.org/pdf/1609.04730); [aitopics](https://aitopics.org/tag/accessible%20swarm%20robotic%20research)
- **Motivation:** multi-robot testbeds are costly and hard to build and maintain, which limits access for researchers and students. — [arXiv 1609.04730](https://arxiv.org/pdf/1609.04730)
- **Design philosophy:** safety is built in at the design stage through minimal safety routines with provable guarantees, so users can still run arbitrary control code. — [arXiv 1609.04730](https://arxiv.org/pdf/1609.04730)
- **Hardware:** a 12×14 ft arena with wireless charging coils and small differential-drive GRITSbots, so experiments can run autonomously around the clock. — [arXiv 1609.04730](https://arxiv.org/pdf/1609.04730)
- **Companion paper:** "Safe, Remote-Access Swarm Robotics Research on the Robotarium" (arXiv 1604.00640) states three safety principles:
  - All robots are provably collision-free.
  - User commands are modified only when a collision is imminent.
  - Collision avoidance runs in real time at more than 30 Hz.

  The safety mechanism uses control barrier functions, which guarantee the safe set stays forward-invariant, and constrains user inputs to a convex polytope K(x). — [arXiv 1604.00640](https://arxiv.org/pdf/1604.00640)
- **Validation shown:** the papers demonstrate the system rather than benchmark it. Examples include four GRITSbots swapping positions with active barrier certificates and ten robots executing individual trajectories without collisions. — [arXiv 1604.00640](https://arxiv.org/html/1604.00640); [arXiv 1609.04730](https://arxiv.org/pdf/1609.04730)
- **Long-term operation follow-up:** "Data-Driven Robust Barrier Functions for Safe, Long-Term Operation" (arXiv 2104.07592) carries the safety work forward, which suggests that years of 24/7 operation exposed model mismatch in the original certificates. — [arXiv 2104.07592](https://arxiv.org/pdf/2104.07592)
- **2023 journal paper:** S. Wilson and M. Egerstedt, "The Robotarium: A Remotely-Accessible, Multi-Robot Testbed for Control Research and Education," IEEE Open Journal of Control Systems, vol. 2, pp. 12–23, DOI 10.1109/OJCSYS.2022.3231523. It covers remote use since the 2017 opening, the design, and a tutorial. — [DOAJ](https://doaj.org/article/3dc971cc06254c8c8f4d25a7ffa30ad4)
- **Usage statistics (sources disagree and come from different dates):**
  - "over a hundred research groups" (Egerstedt, 2017). — [CACM news](https://cacmb4.acm.org/news/196819-new-lab-to-give-nations-researchers-remote-access-to-robots)
  - "more than 250 groups from every continent but Antarctica" since going live in Aug 2017. — [THE Journal 2017](https://thejournal.com/Articles/2017/08/17/New-Facility-Enables-Remote-Users-to-Control-Robot-Swarms.aspx)
  - "over 210 users" (project flyer). — [GT flyer](https://hg.gatech.edu/sites/default/files/attachments/IEEE%20SPS%20Dr.%20Egerstedt%20flyer.pdf)
- **Users outside robotics:** biologists studying social insects and traffic engineers have used it. — [Atlanta Magazine](https://www.atlantamagazine.com/news-culture-articles/georgia-tech-robotarium-shining-beacon-robotic-awesomeness/)
- **NSF CPS project:** "Safe and Secure Open-Access Multi-Robot Systems," which also addresses security of remote code. — [CPS-VO slides](https://cps-vo.org/sites/cps-vo.org/files/cpsvo_file_nodes/CPS_TTP_Option_Synergy_Safe_and_Secure_Open-Access_Multi-Robot_Systems.pdf)

### Inferences
- **The pattern:** the Robotarium papers show their value with **an enabling guarantee** (provable safety, which makes remote access possible) plus **adoption metrics** (users, groups, countries, experiments run). They do not use error bars on a measured quantity.
- **Lesson for an RA-L testbed paper:** name one property that makes the instrument usable and give it a quantitative guarantee or check. For the target testbed, that property could be "every link sample is time-aligned with sensors to within X ms." Then add evidence that the instrument enables experiments that were not possible before.
- **Usage statistics require operating history.** A new testbed cannot claim them, so it needs to stand on fidelity and reproducibility instead.

### Gaps
- **Wilson et al., IEEE Control Systems Magazine 2020** (unverified, from prior knowledge): I believe the title is "The Robotarium: Globally Impactful Opportunities, Challenges, and Lessons Learned in Remote-Access, Distributed Control of Multirobot Systems," vol. 40, no. 1, pp. 26–44. Two searches did not surface it. It reportedly contains the detailed usage statistics (experiments submitted, user breakdown) and lessons learned. **Verify on IEEE Xplore before citing.**
- **ICRA 2017 details** (unverified): DOI 10.1109/ICRA.2017.7989200 and pages 1699–1706 are from memory.
- **Collision counts:** I could not get the papers' own collision-rate numbers for long-term operation, since full text was not fetchable.
- **No Robotarium RA-L paper found.**

## Q2. Educational and research platforms: Duckietown (ICRA 2017), AI Driving Olympics, F1TENTH, MIT RACECAR

### Takeaway
These platforms lead with **low cost, openness and accessibility**, then **benchmarking and competition**. They validated themselves mainly through capability demonstrations and adoption (courses, learners, competition entries). They did not run metrological validation of the platform.

### Cited Findings
- **Duckietown publication:** L. Paull et al., "Duckietown: An open, inexpensive and flexible platform for autonomy education and research," ICRA 2017, pp. 1497–1504, DOI 10.1109/ICRA.2017.7989179. — [Duke Scholars](https://scholars.duke.edu/publication/1640806); [Duckietown](https://duckietown.com/duckietown-an-open-inexpensive-and-flexible-platform-for-autonomy-education-and-research)
- **Duckietown hardware:**
  - Duckiebots are built from off-the-shelf components, with a single monocular camera and all processing on a Raspberry Pi 2.
  - "Duckietowns" are miniature cities with roads, signage, traffic lights and obstacles.
- **Duckietown validation:** a capability list. Duckiebots can follow lanes, avoid obstacles, localize in a global map, navigate the city, and coordinate with each other to avoid collisions.
- **Duckietown headline claim:** educators and researchers save money and time because they don't have to build the infrastructure themselves. All materials are open source. — [Duckietown](https://duckietown.com/duckietown-an-open-inexpensive-and-flexible-platform-for-autonomy-education-and-research)
- **Duckietown adoption evidence:** the edX MOOC "Self-Driving Cars with Duckietown" (ETH Zurich) had over 7,000 learners from more than 170 countries in its first 2021 cohort. — [Duckietown](https://duckietown.com/?p=17860)
- **AI Driving Olympics (NeurIPS 2018), arXiv 1903.02503:**
  - **Format:** a competition built on Duckietown with tasks from lane-following to fleet management.
  - **Evaluation:** organizers provided simulators, logs, code templates, baselines and low-cost hardware. Submissions were evaluated in simulation, on standardized hardware, and at the live event.
  - **Headline lesson:** "the need for better robotics benchmarks, and for improved ways to bridge the gap between simulation and reality," with "a frank assessment of what worked well and what needs improvement." — [arXiv 1903.02503](https://arxiv.org/abs/1903.02503); [DeepAI](https://deepai.org/publication/the-ai-driving-olympics-at-neurips-2018)
- **Follow-up:** "The AI Driving Olympics: An Accessible Robot Learning Benchmark" appeared at NeurIPS 2019 (competition/demo). — [NeurIPS 2019](https://neurips.cc/virtual/2019/15031)
- **Related paper:** "Accessible Interfaces for the Development and Deployment of Robotic Platforms" (arXiv 2305.09848), a Duckietown-lineage paper on accessibility. — [arXiv 2305.09848](https://arxiv.org/pdf/2305.09848)
- **F1TENTH:**
  - **Publication:** M. O'Kelly, H. Zheng, D. Karthik, R. Mangharam, "F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning," NeurIPS 2019 Competition & Demonstration Track, PMLR vol. 123, pp. 77–89, 2020.
  - **What is built:** 1/10-scale low-cost hardware plus multiple virtual environments, so experiments can run safely in the lab.
  - **Validation:** three benchmark tasks with baselines in autonomous racing, and an OpenAI Gym API that controls a real vehicle.
  - **Headline claim:** an evaluation framework (benchmark) on accessible hardware. — [PMLR](https://proceedings.mlr.press/v123/o-kelly20a.html); [NSF PAR](https://par.nsf.gov/biblio/10221872-f1tenth-open-source-evaluation-environment-continuous-control-reinforcement-learning); [f1tenth-gym docs](https://f1tenth-gym.readthedocs.io)

### Inferences
- **Two headline types.** Education platforms (Duckietown, F1TENTH, Robotarium) lead with cost, accessibility and openness, and back it with adoption numbers. A sensing or communication testbed for RA-L is closer to an **instrument or dataset paper**, where reviewers expect quantitative validation of fidelity: time sync, ground-truth accuracy, calibration.
- **Sim-to-real as motivation.** The AI-DO's sim-to-real lesson supports the target testbed's "replay the logged link" idea: measured real link traces narrow the sim-to-real gap for communication. This parallels the gap AI-DO found for perception and control.

### Gaps
- **MIT RACECAR:** no search was done within the tool budget. I found no peer-reviewed "MIT RACECAR" testbed paper in this session.
- **F1TENTH journal or IFAC versions:** later versions exist (e.g., the 2020 IFAC or IEEE papers on the F1TENTH community and courses), but these were not verified here.
- **Duckietown follow-ups:** citation counts were not retrieved (Google Scholar not queried).

## Q3. Multi-robot collaborative SLAM/perception testbeds and datasets: CoPeD, Kimera-Multi, S3E, GRACO, DARPA SubT communication lessons

### Takeaway
In this group the headline claims are **realism and scale** (number of robots, km covered, indoor plus outdoor, modalities), **heterogeneity** (air-ground, multiple sensor viewpoints) and **released benchmarks with reference ground truth**. Kimera-Multi and SubT are the only ones that treat **communication** as a first-class experimental variable, and even they characterise it mostly qualitatively (intermittent or unreliable). None found here logs link quality alongside sensors as a measured, replayable channel. This is the target testbed's opening.

### Cited Findings
- **CoPeD (RA-L 2024):**
  - **Publication:** Y. Zhou, L. Quang, C. Nieto-Granda, G. Loianno, "CoPeD – Advancing Multi-Robot Collaborative Perception: A Comprehensive Dataset in Real-World Environments," IEEE RA-L vol. 9, no. 7, pp. 6416–6423, July 2024; arXiv 2405.14731. — [arXiv](https://arxiv.org/abs/2405.14731); [NSF PAR](https://par.nsf.gov/biblio/10557786-coped-advancing-multi-robot-collaborative-perception-comprehensive-dataset-real-world-environments)
  - **Headline claim:** heterogeneity. It "leverages the untapped potential of air-ground robot collaboration featuring distinct spatial viewpoints, complementary robot mobilities, coverage ranges, and sensor modalities." It also claims a diverse range and adequate overlap of sensor views, unlike SLAM-focused datasets.
  - **Sensors:** aerial robots carry stereo, front and down RGB, depth, IMU and GPS. Ground robots carry stereo, RGB, depth, 3D LiDAR, IMU and GPS. Sequences are indoor and outdoor.
  - **Ground truth:** pose comes from **fusing GPS with existing SLAM frameworks** (no motion capture). High-level labels come from foundation models via zero-shot automatic annotation. — [arXiv html](https://arxiv.org/html/2405.14731v1)
- **CU-Multi (2025, arXiv 2509.19463):** another multi-robot collaborative perception dataset, showing the category is active. — [arXiv](https://arxiv.org/html/2509.19463v1)
- **Kimera-Multi (T-RO 2022):**
  - **Publication:** Y. Tian, Y. Chang, F. Herrera Arias, C. Nieto-Granda, J. P. How, L. Carlone, "Kimera-Multi: Robust, Distributed, Dense Metric-Semantic SLAM for Multi-Robot Systems," IEEE T-RO vol. 38, no. 4, 2022.
  - **Claims:** the system is "parsimonious in terms of communication bandwidth." Robots run distributed place recognition and robust pose-graph optimization when in communication range.
  - **Validation:** photo-realistic simulation, benchmark datasets, and outdoor ground-robot data (up to about 800 m per robot). — [MIT DSpace](https://dspace.mit.edu/handle/1721.1/145301); [arXiv 2106.14386](https://arxiv.org/pdf/2106.14386)
- **Kimera-Multi field paper:** Y. Tian, Y. Chang, L. Quang, A. Schang, C. Nieto-Granda, J. P. How, L. Carlone, "Resilient and Distributed Multi-Robot Visual SLAM: Datasets, Experiments, and Lessons Learned," arXiv 2304.04362, presented at IROS 2023.
  - **Contributions:** (i) changes that make the system resilient where communication is intermittent or unreliable; (ii) released multi-robot datasets from live MIT-campus experiments with up to 8 robots and up to 8 km, with "accurate reference trajectories and maps" (tunnels, hallways, mixed indoor and outdoor lighting, pedestrians and cars); (iii) lessons learned.
  - **Validation:** resilience tested under different communication scenarios against a centralized baseline. The distributed system reaches comparable accuracy, handles communication failures better, and keeps working in disconnected clusters. — [arXiv 2304.04362](https://arxiv.org/abs/2304.04362v1); [dataset page](https://web.mit.edu/sparklab/datasets/KimeraMultiData); [data GitHub](https://github.com/plusk01/kimera-multi-data)
- **S3E:**
  - **Publication:** D. Feng, Y. Qi et al. (Sun Yat-sen Univ.), "S3E: A Multi-Robot Multimodal Dataset for Collaborative SLAM," arXiv 2210.13723.
  - **What is built:** a fleet of UGVs on four collaborative trajectory paradigms, with 13 outdoor and 5 indoor sequences.
  - **Sensors:** 360° LiDAR, high-resolution stereo, high-rate IMU, and **UWB relative observations**.
  - **Validation:** benchmarks of collaborative and single-robot SLAM.
  - **Status:** accepted October 24, 2024, recommended by Editor Javier Civera. This fits RA-L, but no search result names the journal explicitly. Treat RA-L as likely, not confirmed. — [arXiv](https://arxiv.org/pdf/2210.13723); [IEEE DataPort DOI 10.21227/rrcw-fv27](https://ieee-dataport.org/documents/s3e-multi-robot-multimodal-dataset-collaborative-slam)
- **DARPA SubT, Team CERBERUS:** "Team CERBERUS Wins the DARPA Subterranean Challenge: Technical Overview and Lessons Learned" (arXiv 2207.04914).
  - **Communications:** an ad-hoc 5.8 GHz wireless mesh, with legged and roving robots as mesh nodes and flying robots as clients.
  - **Breadcrumbs:** carrier ANYmals deployed communication "breadcrumbs" on command. The team also used a fiber tether and a dedicated communications robot.
  - **Stated difficulty:** "the combination of narrow settings, very challenging communications, and sensor degradation." — [arXiv 2207.04914](https://arxiv.org/pdf/2207.04914); [IEEE Spectrum](https://spectrum.ieee.org/automaton/robotics/robotics-hardware/how-teams-are-solving-the-biggest-challenge-at-darpa-subt)
- **SubT, other teams** (trade press, 2020–2022):
  - CoSTAR sent in robots carrying deployable nodes.
  - Explorer dropped ranging-radio anchors to form a mesh, up to 10 nodes per robot. Lead Matt Travers: "radio does not penetrate rock."
  - Press summary: teams "encountered the most difficulty in employing communications solutions underground." — [IEEE Spectrum](https://spectrum.ieee.org/how-teams-are-solving-the-biggest-challenge-at-darpa-subt-2650278934); [AFCEA Signal](https://www.afcea.org/signal-media/technology/robots-confront-challenges-underground)

### Inferences
- **Reviewers of RA-L dataset papers expect** three things, judging from CoPeD, S3E and Kimera-Multi:
  - a comparison table against prior datasets (modalities, robot count, heterogeneity, indoor/outdoor);
  - an explicit ground-truth method with an accuracy statement;
  - baseline benchmarks of existing algorithms on the new data.
- **Ground truth without mocap or GNSS is precedented but needs care.** CoPeD fuses GPS with SLAM, Kimera-Multi builds "accurate reference trajectories" offline, and S3E uses UWB relative ranges. Since the target testbed has neither mocap nor GNSS indoors, it must state its ground-truth method (e.g., LiDAR map-based registration, survey markers) and quantify its accuracy.
- **Communication is the differentiating gap.** Kimera-Multi tests "different communication scenarios," and SubT teams name communication as the hardest problem. Yet none of the datasets found here logs measured link quality (RSSI, CSI, throughput, latency) synchronized with sensor data in a common frame. The target testbed's "every inter-agent link measured on one clock and one map frame + replay" directly fills the gap these works expose.

### Gaps
- **GRACO** (unverified, from prior knowledge): believed to be "GRACO: A Multimodal Dataset for Ground and Aerial Cooperative Localization and Mapping," IEEE RA-L 2023. The search did not surface it. **Verify before citing.**
- **CoPeD ground-truth accuracy:** the reported accuracy numbers and the robot count were not retrieved, since the full text was not fetchable.
- **Kimera-Multi field paper details:** the exact IROS 2023 page and DOI, and the concrete communication metrics measured, were not retrieved.
- **SubT CoSTAR and Explorer:** their own field-robotics papers (e.g., the CoSTAR NeBula overview, CMU Explorer) were not retrieved, nor were any quantitative link-quality logs from SubT.

## Q4. Connected-robot / networked-robot testbeds: ROS-NetSim, real wireless measurement, digital twins

### Takeaway
ROS-NetSim (RA-L 2021) is the reference RA-L paper for coupling robot and network simulation. Its headline is **fidelity-tunable, transparent co-simulation** of perception-action-communication loops. It is simulation-only, its code was archived in 2022, and later work criticises its scalability. Real-measurement robotic wireless testbeds date back to the Mostofi lab (around 2009–2011), but none of those is an RA-L multi-robot perception testbed.

### Cited Findings
- **ROS-NetSim publication:** M. Calvo-Fullana, D. Mox, A. Pyattaev, J. Fink, V. Kumar, A. Ribeiro, "ROS-NetSim: A Framework for the Integration of Robotic and Network Simulators," IEEE RA-L vol. 6, no. 2, pp. 1120–1127, 2021; arXiv 2101.10113. — [arXiv](https://www.arxiv.org/abs/2101.10113); [dblp](https://dblp.org/pid/223/6064)
- **ROS-NetSim claims:**
  - Motivation: perception-action-communication (PAC) loops "lack appropriate simulation tools."
  - Design: transparent to the ROS application, agnostic to the network and physics simulator, and "tunable in fidelity and complexity." — [arXiv](https://arxiv.org/pdf/2101.10113)
- **ROS-NetSim status and criticism:**
  - The GitHub repo was archived Mar 16, 2022 and is read-only. — [GitHub](https://github.com/NESTLab/ros-net-sim)
  - The BotNet simulator paper (arXiv 2108.13606) says ROS-NetSim "is not designed to handle high agent counts nor standards compliant communication models." — [arXiv 2108.13606](https://arxiv.org/pdf/2108.13606)
- **Mostofi lab (UCSB):**
  - A 2011 overview of communication-aware networked robots validates a probabilistic channel model "with a robotic testbed, making an extensive number of channel measurements." — [JRobotics11 PDF](https://web.ece.ucsb.edu/~ymostofi/papers/JRobotics11.pdf)
  - Public 2.4 GHz Wi-Fi channel data: a router as transmitter and a Pioneer robot with a Wi-Fi card as receiver, 67 routes. — [Mostofi data page](https://web.ece.ucsb.edu/mostofi-lab/code-data/channel-measurements-2009.html)
  - Wi-Fi imaging data from robot-positioned TX/RX (Pioneer 3-AT, RSSI). — [Mostofi Wi-Fi imaging](https://web.ece.ucsb.edu/mostofi-lab/code-data/WiFiImagingData2014)
- **RoboMNIST (arXiv 2408.16703):** two robot arms with Wi-Fi CSI, video and audio. It targets activity recognition, not link characterisation. — [arXiv](https://arxiv.org/abs/2408.16703v2)
- **iV2V/iV2I+ industrial data:** about 10 h of sidelink communication between 3 AGVs (cellular, not Wi-Fi). — [IEEE DataPort / TH-OWL](https://www.th-owl.de/elsa/record/2375.aref)

### Inferences
- **The target testbed complements ROS-NetSim.** ROS-NetSim models link effects in simulation; the target testbed **measures** them and replays them into collaborative perception. A natural framing is "trace-driven replay grounded in measured links," validated by showing that replayed-link performance predicts live-link performance. ROS-NetSim does not provide that validation.
- **The archived code is a reproducibility lesson.** The target paper should commit to an open release and a maintenance plan, because RA-L reviewers increasingly weigh reproducibility.
- **Nearest neighbours lack the multi-robot perception context.** The Mostofi-lab and RoboMNIST-style CSI datasets show robot-mounted Wi-Fi measurement is established, but they do not pair it with multi-robot perception, a shared map frame, or collaborative-perception replay.

### Gaps
- **Digital-twin testbeds** (Isaac Sim-based multi-robot wireless twins, e.g., NVIDIA Sionna/Aerial-based robot twins): none found or verified in this session's budget.
- **Real-radio RA-L multi-robot testbeds:** no RA-L paper was found that measures real wireless links on a multi-robot platform alongside perception sensors. This may be a genuine gap, or may reflect limited search depth.
- **ROS-NetSim experiments:** the quantitative experiments in the paper (e.g., the specific NS-3 scenarios and timing-overhead measurements) were not retrieved.

## Q5. What these papers emphasise as their contribution, and what readers and reviewers valued (citation patterns, follow-ups)

### Takeaway
Across the set, the headline claims fall into three families, each with its own kind of validation:

1. **Accessibility and cost.** Robotarium, Duckietown and F1TENTH validate with capability demonstrations plus adoption numbers.
2. **Benchmark and competition.** AI-DO and F1TENTH validate with baselines, standardized evaluation, and a candid account of what failed.
3. **Realism, scale and heterogeneity of data.** CoPeD, S3E and Kimera-Multi validate with a dataset comparison table, a ground-truth method, and algorithm benchmarks.

Follow-ups suggest readers valued the **long-lived, reusable infrastructure**: Duckietown's MOOC and the AI-DO series, the Robotarium's years of public operation and its later journal papers, and Kimera-Multi's released datasets. Work that was not maintained, such as ROS-NetSim's archived code, drew criticism.

### Cited Findings
- **Robotarium follow-ups:** the ICRA 2017 Best Multi-Robot Systems award, then a 2023 journal paper (OJ-CSYS) reporting years of remote use, then follow-up safety work for long-term operation. — [aitopics](https://aitopics.org/tag/accessible%20swarm%20robotic%20research); [DOAJ](https://doaj.org/article/3dc971cc06254c8c8f4d25a7ffa30ad4); [arXiv 2104.07592](https://arxiv.org/pdf/2104.07592)
- **Duckietown follow-ups:** the ICRA 2017 paper, then the AI-DO at NeurIPS 2018 and 2019, then an edX MOOC with over 7,000 learners in more than 170 countries (2021). — [Duckietown](https://duckietown.com/?p=17860); [arXiv 1903.02503](https://arxiv.org/abs/1903.02503)
- **Kimera-Multi follow-ups:** the T-RO 2022 algorithm paper, then the IROS 2023 "datasets, experiments, lessons learned" paper with released data. The second paper reframes the work as a field-validated instrument and benchmark. — [arXiv 2304.04362](https://arxiv.org/abs/2304.04362v1)
- **ROS-NetSim follow-up:** BotNet (2021) positions itself against ROS-NetSim's scalability and standards-compliance limits. — [arXiv 2108.13606](https://arxiv.org/pdf/2108.13606)

### Inferences
- **Recommended headline for the target RA-L paper:** **fidelity and synchronised completeness**. Every link is measured with the sensors on one clock and one frame, and the logged link can be replayed. Accessibility is the wrong lead, because the Robotarium and Duckietown own that framing and back it with adoption data a new testbed cannot have.
- **Validation the target paper should show,** to match what this set of papers implies reviewers want:
  1. Time-sync error between modalities and link logs.
  2. Ground-truth accuracy without mocap or GNSS, checked against an independent reference.
  3. Link-measurement fidelity, e.g., CSI or RSSI consistency and coverage map repeatability.
  4. A replay-versus-live experiment showing that collaborative-perception metrics under replayed links match those under live links.
  5. A comparison table against CoPeD, S3E, Kimera-Multi data, GRACO and ROS-NetSim on "link measured?", "heterogeneous agents?", "infrastructure nodes?" and "replay?"
- **Add a "lessons learned" section.** Kimera-Multi's IROS 2023 paper and the CERBERUS SubT paper both include one, and that style appears well received in the field-robotics community.

### Gaps
- **Citation counts:** Google Scholar counts were not retrieved for any paper (budget and fetch limits). Citation-pattern claims above rest on the follow-ups observed, not on counts.
- **Reviewer comments:** these are not public for RA-L. Inferences about what reviewers value are indirect.
