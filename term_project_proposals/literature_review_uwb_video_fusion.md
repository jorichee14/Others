# Literature Review: Does UWB identity reach the video coaches watch?

Purpose. The proposal's problem statement contains this sentence:

> The LocusConnect UWB stream is immune to that occlusion and already carries identity, 3D position, and heart rate, yet none of it appears in the video coaches actually watch.

This review checks each part of that claim against the academic literature and against what commercial sports-tracking vendors ship, then proposes a wording that survives scrutiny. It also collects the prior work the proposal should cite for its three technical pieces: identity fusion, markerless camera-to-UWB calibration, and Z-axis jump detection.

Sourcing caveat. Most publisher and vendor sites were unreachable from this environment, so entries marked [snippet] rest on search-result abstracts and result sentences, not on the full text. Numbers should be checked against the PDF before they go into a submission. Entries marked [fetched] were read first-hand (GitHub READMEs).

---

## 1. Verdict on the sentence

| Claim | Status | Evidence |
|---|---|---|
| UWB tracking is immune to the occlusion that breaks video trackers | Supported, with a caveat | Radio identity drives ID switches to zero in LiDAR-UWB tracking (LUOT, 2025). UWB itself suffers NLOS range bias when bodies block anchors, so "immune" overstates it; "does not lose identity" is accurate. |
| The stream carries identity and 3D position | Supported | LocusConnect public materials list 3D real-time location, MQTT/HTTP/WebSocket/UDP APIs, and tags with motion sensors [snippet, locusconnect.com, docs.locusconnect.com, smartcity.org.tw catalogue]. |
| The stream carries heart rate | Not verified | No public LocusConnect page or API description mentions a heart-rate field. The course deck says "integrated biosignals (heart rate)" for the wearable. Cite the deck or the device documentation, or drop heart rate from the proposal. |
| None of it appears in the video coaches watch | Too strong as written | Kinexon + PlaySight, Catapult Vision / MatchTracker + Vector, STATSports Sonra Video Manager, Hudl Titan, and ShotTracker + Catapult Vision all put wearable or LPS metrics beside video via timeline synchronisation. Zebra MotionWorks UWB data is overlaid on NFL broadcast replays. |

What the vendors do not do, and what the literature has not done, is the narrower claim the proposal should make:

> Commercial tools align wearable telemetry to video only in time, through a manual sync point or a dual timeline, or cut video per athlete. They do not project tag identity, 3D position, or physiology onto the players in the frame. Spatial overlays of identity exist only where the vendor owns a calibrated fixed-camera installation (Second Spectrum with 6 cameras, Hawk-Eye with 14, Zebra with stadium receivers). No vendor offers it from an uncalibrated phone camera, and no published method jointly calibrates camera pose, focal length, and clock offset from UWB tag trajectories with unknown tag-to-person correspondences.

Proposed replacement for the proposal sentence (same length class):

> The LocusConnect UWB stream keeps every player's identity and 3D position through the occlusions that break video trackers, yet commercial tools only align that telemetry to video in time; nothing is projected onto the players in the footage coaches watch, and doing so today needs a surveyed, fixed camera install.

---

## 2. Commercial state of practice

| Vendor / product | What is synchronised | Identity drawn on the video? | Fixed calibrated camera needed? |
|---|---|---|---|
| Kinexon + PlaySight (partnership 2019) | UWB performance data and PlaySight video shown synchronously in the Kinexon app | Not described | Yes, PlaySight venue cameras |
| Catapult Vision / MatchTracker + Vector (incl. ClearSky LPS) | Wearable metric streams offset to the video timeline by a manually chosen sync point | No | No, any uploaded footage |
| STATSports Sonra Video Manager | Single-click sync point; dual timeline; separate 2D pitch re-animation | No, 2D view is separate | No |
| Zebra MotionWorks (NFL Next Gen Stats) | Stadium UWB/RFID receivers; broadcast overlays of routes and speeds | Yes, for broadcast graphics | Yes, stadium install |
| Second Spectrum / Genius Sports | Optical only; identity and stats drawn over players | Yes | Yes, 6 cameras per arena |
| Hawk-Eye (NBA) | Optical skeletal tracking, 29 body points | Yes | Yes, 14 cameras per arena |
| ShotTracker + Catapult Vision | UWB anchors and chipped ball; stats linked to video in one workflow | Not described | Yes for anchors |
| Pixellot + HELIOS (2026) | Wearable data detects on-ice shifts and cuts per-athlete video | No, temporal slicing | Yes, Pixellot camera |
| Hudl Titan GPS | Session metrics shown beside video | No | No |
| Veo Analytics | Video only; identity from jersey-number OCR; support article on fixing wrong-player events | From OCR only | No, tripod camera |

Sources (all [snippet]): kinexon.com/pr/playsight-interactive-and-kinexon-partner-up; vision.catapultsports.com help articles on single-sync-point offset; statsports.com Sonra Video Manager article; investors.zebra.com MotionWorks launch; sportsvideo.org on Second Spectrum and on Pixellot/HELIOS; espn.com on Hawk-Eye in the NBA; insidersport.com on ShotTracker and Catapult; hudl.com Titan reveal; support.veo.com.

Takeaway. Timeline sync is a solved commercial feature. Spatial fusion of radio identity into the frame is only done by vendors who control the cameras. The proposal's niche, any camera plus self-calibration, is not occupied.

---

## 3. Academic work on radio-to-video identity fusion

Directly relevant, UWB plus camera:

1. Lee, Um, Park, Lee. "Moving Object Performance Analysis System Using Multi-camera Video and Position Sensors." IEEE BigComp 2020, pp. 441-445 (ETRI). Four cameras plus IR-UWB tags for player tracking; 16 fps; player identification rate 98.8 percent; mean position error 0.48 m. The closest published sports precedent. [snippet]
2. Ishige, Yoshimura, Yonetani. "Opt-in Camera: Person Identification in Video via UWB Localization." arXiv 2409.19891, 2024. One webcam plus a UWB tag; tag ground track from an unscented Kalman filter; tag-to-tracklet matching by constrained linear optimisation; calibration by a person walking the tag through the view; identifies the tag carrier in over 85 percent of frames among 8 to 23 people at 10 fps. The most reusable method. [snippet]
3. Peng, Yu, Xia, Zheng, Zhao, Chen. "An Indoor Positioning Method Based on UWB and Visual Fusion." Sensors 22(4):1394, 2022. Hungarian matching between UWB and vision tracks, then federated Kalman fusion; 25 percent better accuracy than UWB alone. The baseline association recipe. [snippet]
4. Li et al. "LUOT: LiDAR-UWB Object Tracking With Zero ID Switches and Centimeter-Level Precision." IEEE Internet of Things Journal, 2025. Radio identity anchored to LiDAR boxes; zero ID switches; 8.7 cm accuracy. The metric template for the proposal's identity claim. [snippet]
5. Huang, Gautam, Choi, Saripalli. "WiDEVIEW: An UltraWideBand and Vision Dataset." arXiv 2309.16057, 2023. The only public UWB plus camera pedestrian dataset found. [snippet]

Same association problem with other radios or wearables:

6. Papaioannou, Wen, Markham, Trigoni. "Fusion of Radio and Camera Sensor Data for Accurate Indoor Positioning." IEEE MASS 2014. Multi-hypothesis tracking that uses WiFi to resolve occlusions; error below 1 m. [snippet]
7. Pham, Le, Dao. "Improvement of Person Tracking Accuracy in Camera Network by Fusing WiFi and Visual Information." Informatica 41(2), 2017. Optimal assignment plus Kalman filter between camera observations and WiFi identities. [snippet]
8. Henschel, von Marcard, Rosenhahn. "Simultaneous Identification and Tracking of Multiple People Using Video and IMUs." CVPRW 2019. Graph labelling over detections and IMU orientation; IDF1 91.2 percent; works when clothing is identical, the sports failure case. [snippet]
9. Liu et al. "Vi-Fi: Associating Moving Subjects across Vision and Wireless Sensors." IPSN 2022. Learned affinity matrix plus bipartite matching; 81 to 91 percent association accuracy. [fetched README]
10. Cao et al. "ViTag: Online WiFi Fine Time Measurements Aided Vision-Motion Identity Association." IEEE SECON 2022. Adding a ranging signal to motion cues raised identity precision by 12.6 points on average, which argues for UWB over IMU-only association. [snippet]
11. Huang, Tseng. "Fusing Video and Inertial Sensor Data for Walking Person Identification." arXiv 1802.07021, 2018 (NCTU, now NYCU). Local prior work on the same association problem; 76 percent within 2 s. [snippet]
12. De Marchi, Turetta, Pravadelli, Bombieri. "Real-Time Multi-Person Identification and Tracking via HPE and IMU Data Fusion." DATE 2024. 96.9 percent identification-and-tracking accuracy with pose keypoints plus IMUs. [snippet]

Patents showing the sports use case is recognised but closed:

13. US 12,112,224 B2 (Wiser Systems, 2024). Camera and UWB RTLS network in an arena; the UWB-identified player is highlighted with a rectangle in retrieved video.
14. US 9,795,830 (Isolynx, later Catapult). UWB tag plus RFID binding of tag ID to player ID; football.
15. EP 4579610 A1 (2025). Wearable data "confirming identity of athletes before and after events that obscure athletes from view." The proposal's exact use case, as a filing rather than a product.

Why identity is the bottleneck in sports video tracking:

16. Cui et al. "SportsMOT." ICCV 2023. ByteTrack leaves 3089 ID switches and 71.4 IDF1 on the test set; the best reported tracker (Deep-EIoU, WACVW 2024) still leaves 2659. [fetched README, numbers from snippet]
17. Cioppa et al. "SoccerNet-Tracking." CVPRW 2022. With ground-truth detections, association accuracy stays near 60, so identity, not detection, is the weak link. [fetched README]
18. Scott et al. "SoccerTrack." CVPRW 2022. Uses STATSports GNSS at 10 Hz (0.22 m error) as the per-player identity reference for video, the same role UWB plays in the proposal. [snippet]
19. Van Zandycke et al. "DeepSportradar player re-identification." ACM MMSports 2022. Basketball re-ID baseline 72.7 mAP; jersey numbers visible in only a fraction of crops. [fetched README]

Gap. Only one peer-reviewed sports result fuses UWB with video (ETRI 2020, four fixed cameras). Every method found matches on the 2D ground plane; none uses the tag's height or 3D position, and none reports ID switches on athlete crossings specifically.

---

## 4. Markerless camera-to-radio calibration

Closest to the proposed self-calibration:

20. Herttuainen, Eerola, Lensu, Kalviainen. "Simultaneous Camera Calibration and Temporal Alignment of 2D and 3D Trajectories." VISAPP 2017. Full camera matrix plus time scale and offset from one 2D track and one 3D track with unknown correspondences, by RANSAC over sampled point pairs; over 96 percent success. Single target, Leap Motion reference. The nearest methodological match. [snippet]
21. Yang, Bar-Shalom. "Camera Calibration Using Inaccurate and Asynchronous Discrete GPS Trajectory from Drones." arXiv 2608.26548, 2026. Joint orientation, altitude bias, and camera-GPS time offset by maximum likelihood; time offset recovered to 0.27 ms against 100 ms GPS sampling; estimates reach the Cramer-Rao bound. Known correspondence (one drone). [snippet]
22. Lee, Nishino, Nobuhara. "Spatiotemporal Multi-Camera Calibration using Freely Moving People." IEEE RA-L 2025. Extrinsics, time offsets, and person associations solved jointly from monocular 3D pose, by alternation with soft assignment. Camera to camera, not camera to radio. [snippet]
23. Lee, Shibata, Nonaka, Nobuhara, Nishino. "Extrinsic Camera Calibration From a Moving Person." IEEE RA-L 2022. One walking person's keypoints suffice for multi-camera extrinsics up to scale. [snippet]
24. Guan et al. "Extrinsic Calibration of Camera Networks Based on Pedestrians." Sensors 16(5):654, 2016. Head and feet detections plus RANSAC Procrustes. [snippet]
25. Persic, Petrovic, Markovic, Petrovic. "Spatiotemporal Multisensor Calibration via Gaussian Processes Moving Target Tracking." IEEE T-RO 2021. Time delay from velocity-magnitude profiles, then extrinsics; delay accurate to a fraction of the sampling period. A clean decoupling the project can reuse. [snippet]
26. Song, Richard, Olivares-Mendez. "Joint Spatial-Temporal Calibration for Camera and Global Pose Sensor." 3DV 2024. Intrinsics, extrinsics, and time offset against an external positioning system, with observability analysis. [snippet]
27. Tang, Suri, Ajisafe, Wandt, Rhodin. "CasCalib: Cascaded Calibration for Motion Capture from Sparse Unsynchronized Cameras." arXiv 2405.06845, 2024. Focal length, extrinsics, and time offsets from 2D keypoints alone. Camera to camera. [fetched README]
28. Tsaregorodtsev et al. "Automated Static Camera Calibration with Intelligent Vehicles." IEEE IV 2023. Roadside camera from a GNSS vehicle track with hypothesis filtering over detections. [snippet]
29. Durmaz, Cevikalp. "Targetless Radar-Camera Calibration via Trajectory Alignment." Sensors 25(24), 2025. Sub-degree rotation, 0.12 to 0.27 m translation from drone trajectories; the accuracy benchmark for radio-to-camera trajectory alignment. [snippet]
30. UCF thesis, 2020: "camera calibration using UWB as an auxiliary sensor"; one traversal of a UWB-equipped pedestrian calibrates a roadside camera. No numbers found. [snippet]

Typical accuracy. Centimetre translation and sub-degree rotation when the reference is motion capture or a ranging sensor; 1.5 to 2 m when consumer GNSS is the reference (Ojala et al., IET ITS 2023). Time offsets are recovered to sub-frame precision.

Gap. No paper jointly estimates camera pose, focal length, and clock offset from UWB tag trajectories matched to human-pose keypoints with RANSAC over unknown tag-to-person correspondences. The nearest neighbours are Herttuainen 2017 (single target, Leap Motion), Opt-in Camera 2024 (UWB, identification-oriented), and Yang and Bar-Shalom 2026 (GPS drone, known correspondence). The proposal's calibration piece is a new combination of established parts, which is the right level of novelty for a semester.

---

## 5. Jump detection from the Z axis

31. Blauberger et al. "Validation of Player and Ball Tracking with a Local Positioning System." Sensors 2021. Kinexon UWB versus motion capture: player position RMSE 9 cm; 3D impact errors 17 to 21 cm. The published UWB vertical accuracy. [snippet]
32. IntechOpen review of UWB in team sports: "the major source of error in UWB systems is in the height or along the Z-axis." [snippet]
33. Skazalski et al. Scand J Med Sci Sports 2018. IMU jump counting in elite volleyball: sensitivity 96.8 percent, specificity 100 percent, height underestimated by 2.5 to 4.1 cm against motion capture. [snippet]
34. "AI-assisted Automatic Jump Detection and Height Estimation in Volleyball Using a Waist-worn IMU." arXiv 2505.05907, 2025. Temporal CNN; detection F1 0.90. [snippet]
35. Kinexon IMU versus force plate, 2025: trivial to small differences in flight time. [snippet]

Gap and risk. No paper detects jumps from UWB z-position alone in basketball or volleyball; all validated jump metrics come from IMUs. With a 17 to 21 cm vertical error, a 30 cm jump is near the noise floor and a 60 cm jump is clear. The proposal should present jump detection as "jumps above a threshold" validated against video, state the Z-axis error as a known limitation, and note that the tag's built-in motion sensor could complement z(t) if the API exposes it.

---

## 6. What this means for the proposal

1. Reword the problem sentence as proposed in section 1. The defensible claim is about spatial projection from an uncalibrated camera, not about absence of video integration.
2. Either source the heart-rate field from the device or API documentation, or remove heart rate from sections 3, 4, and 6. The jump clip still works with height alone.
3. Cite the identity bottleneck with numbers: thousands of ID switches on SportsMOT even for the best trackers, and association accuracy near 60 on SoccerNet with perfect detections.
4. Position the calibration piece against Herttuainen 2017, Opt-in Camera 2024, and Yang and Bar-Shalom 2026, and name the new combination: pose plus focal length plus clock offset from UWB tracks and pose keypoints with unknown correspondences.
5. State the Z-axis error (17 to 21 cm) as a limitation and set the jump threshold accordingly.
