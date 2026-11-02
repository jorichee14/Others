const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, BorderStyle, AlignmentType, HeadingLevel, LevelFormat, PageBreak } = require('docx');

const F = "Liberation Serif";
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const run = (t, o={}) => new TextRun({ text: t, font: F, size: o.size||21, bold: o.bold, italics: o.it });
const P = (t, o={}) => new Paragraph({ alignment: o.align||AlignmentType.JUSTIFIED, spacing: { after: o.after??80, line: 270 }, children: Array.isArray(t)?t:[run(t,o)] });
const H = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 160, after: 60 }, children: [run(t,{bold:true,size:23})] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 120, after: 40 }, children: [run(t,{bold:true,size:21})] });
const RH = (h, t) => P([run(h+" ",{bold:true,it:true}), run(t)]);
const bullet = (t) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 40, line: 270 }, alignment: AlignmentType.JUSTIFIED, children: [run(t)] });
const num = (t) => new Paragraph({ numbering: { reference: "n", level: 0 }, spacing: { after: 40, line: 270 }, alignment: AlignmentType.JUSTIFIED, children: [run(t)] });
const quote = (t) => new Paragraph({ indent: { left: 500, right: 500 }, spacing: { after: 100, line: 270 }, alignment: AlignmentType.JUSTIFIED, children: [run(t,{it:true})] });

function table(caption, headers, rows, widths, note) {
  const total = widths.reduce((a,b)=>a+b,0);
  const cell = (t, w, i, hdr, last) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders: { top: hdr?{style:BorderStyle.SINGLE,size:12,color:"000000"}:none, bottom: hdr?{style:BorderStyle.SINGLE,size:6,color:"000000"}:(last?{style:BorderStyle.SINGLE,size:12,color:"000000"}:none), left: none, right: none },
    margins: { top: 40, bottom: 40, left: 60, right: 60 },
    children: [new Paragraph({ alignment: AlignmentType.LEFT, spacing:{after:0,line:240}, children: [run(t,{size:18,bold:hdr})] })] });
  const out = [new Paragraph({ spacing: { before: 120, after: 40 }, alignment: AlignmentType.JUSTIFIED, children: [run(caption.split(".")[0]+".",{bold:true,size:19}), run(caption.slice(caption.indexOf(".")+1),{size:19})] })];
  out.push(new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [ new TableRow({ tableHeader: true, children: headers.map((h,i)=>cell(h,widths[i],i,true,false)) }),
      ...rows.map((r,ri)=>new TableRow({ children: r.map((c,i)=>cell(c,widths[i],i,false,ri===rows.length-1)) })) ] }));
  if (note) out.push(new Paragraph({ spacing:{before:40,after:120,line:240}, alignment: AlignmentType.JUSTIFIED, children:[run(note,{size:18})] }));
  else out.push(new Paragraph({ spacing:{after:100}, children:[] }));
  return out;
}

const W3 = [3300, 1700, 4500];
const lit = (cap, rows, note) => table(cap, ["Paper", "Venue", "Note"], rows, W3, note);


const children = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:40}, children:[run("Short-Term Collaboration — Stretch 4",{size:20})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:40}, children:[run("From Localisation to a Millimetre Grasp over a Shared Link: Scenario and Literature",{bold:true,size:30})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:60}, children:[run("Student ID: 314540011",{size:21})] }),
  new Paragraph({ border:{ bottom:{ style:BorderStyle.SINGLE, size:8, color:"000000" } }, spacing:{after:160}, children:[] }),

  H("1. The Scenario"),
  P("One Stretch 4 mobile manipulator picks a small everyday object (a cup, a pen, a remote control) from a table, a shelf or an open drawer, starting from elsewhere in the building. It carries a base lidar and IMU, a head RGB-D camera on a pan–tilt mount, a wrist RGB-D camera beside the gripper, joint encoders and gripper effort sensing. Its arm is Cartesian (lift, telescoping arm, wrist) with no lateral joint, so sideways correction of the gripper must come from the base. The heavy models (open-vocabulary detection, grasp proposal, verification) run on a server reached over the building's shared Wi-Fi; the robot's own computer runs localisation and the low-level controllers."),
  P("The task spans building scale to millimetres. The robot localises in a prebuilt map, chooses where to stand, finds the object with the head camera, brings the arm close and finally places the fingers around the object within a few millimetres. Each stage hands the target to the next with an error, and the errors add up. The link adds one more: whatever the server returns describes the scene at the moment the image was captured, so while the image travels, waits in the server's queue, is processed and the answer returns, the robot (and possibly the object) keeps moving. A delay d at a relative speed v becomes a position error of about v·d at the gripper, in the same budget as the sensor errors."),
  P("The problem the robot faces is the following."),
  quote("Given an error at every link from map to fingertip, including the error the link's delay creates, which stage should run on the robot and which on the server, what should be sent and when, and when should control pass from the head camera to the wrist camera, so that the gripper closes within the object's tolerance on the first attempt?"),
  P("Four properties make the problem specific."),
  num("Errors compound along the chain, and the link is one of them. Localisation, base execution, head-camera depth, pan–tilt calibration, arm deflection, wrist calibration and the delay of every offloaded answer each add an error at the gripper."),
  num("Tolerance shrinks stage by stage, from about 10 cm at the table to a few millimetres at closure, while the permissible delay shrinks with it: a stage whose tolerance is δ at relative speed v can tolerate at most δ/v of delay."),
  num("Where a stage runs is a choice. Offloading buys a stronger model at the price of delay and of bytes on a shared link; running on the robot buys immediacy at the price of a weaker model. The right choice changes from stage to stage."),
  num("The Cartesian arm needs the base. Lateral and yaw corrections at the gripper are base motions, so the final alignment is a base–arm loop whose rate a Wi-Fi round trip cannot sustain."),
  RH("The quantity.", "The position error of the gripper relative to the object at closure, decomposed into the contribution of each stage and of the link delay (v·d per offloaded answer), measured against fiducials; together with first-attempt success, time per stage and bytes per stage."),
  ...table("Table 1. The stages of a pick: tolerance, delay budget, where each runs and what crosses the link.", ["Stage", "Must be within", "Delay budget at its speed", "Runs on", "Sent over the link"], [
    ["1 Localise and drive","5–10 cm","≈ 0.2–0.5 s at 0.3 m/s","robot: lidar–inertial localisation, base planner; server: semantic goal once","a low-resolution frame on request"],
    ["2 Find the object","2–3 cm","≈ 0.3 s when the base is stopped, near zero while moving","server: detection and grasp proposal","one full-resolution crop of the target region"],
    ["3 Pre-grasp","1–2 cm","≈ 0.1–0.2 s at 0.1 m/s arm speed","robot: arm on the stage 2 estimate; server: optional refinement","changed pixels in the target box, if refined"],
    ["4 Align","2–5 mm","≈ 50 ms at 5 cm/s, below a Wi-Fi round trip under load","robot only: wrist-camera servo on base yaw and arm extension","nothing"],
    ["5 Close and verify","—","≈ 0.3 s","robot: effort; server: verification","events; one crop after the lift"],
  ], [1300, 1250, 2200, 2650, 2100], "Delay budget = tolerance / relative speed. Values are starting points to be replaced by measurement; the loaded round trip measured on the group's own testbed was 25–33 ms, rising to 60–109 ms on a busier pass, before any server queue or inference time."),


  H("2. Existing Techniques That Address the Link"),
  P("Six families of existing technique reduce the error that the link adds at the gripper. Each is described below with the work it draws on; Table 2 lists a cumulative comparison of them that can be run on the robot."),
  RH("2.1 Motion-compensated answers.", "Every image goes up with its capture time and the robot's pose at that instant (base pose from localisation, joint positions from the encoders). The server answers in the frame of that capture. On arrival, the robot transforms the answer by its own motion since the capture, using odometry and forward kinematics. The robot's own motion during the delay is thereby cancelled; what remains is the motion of the scene, close to zero for an object resting on a table. This converts most of the v·d term to zero without changing the link. Related work: the Smith predictor (Smith 1957); predictive displays for teleoperation with time delay (Bejczy et al. 1990)."),
  RH("2.2 Queue discipline.", "Under load most of the delay is queueing, not transmission. The robot publishes with best-effort, keep-last-one quality of service so that only the newest image ever waits; each request carries its stage and deadline; the server drops a request whose stage has ended and serves the rest earliest-deadline-first rather than in arrival order. Related work: ROS 2 and DDS quality-of-service settings; earliest-deadline-first scheduling."),
  RH("2.3 Region, change and timing of the uplink.", "Only the region the stage needs is sent, and within it only what changed since the server's last copy (an RGB difference and a depth difference above about 2 cm). Resolution and depth precision are set per stage. The image for the next stage is requested ahead of time, from the pose the planned motion will reach, so that its answer arrives as that stage begins rather than after. Related work: DDS (Du et al. 2020), Reducto (Li et al. 2020), Glimpse (Chen et al. 2015), edge-assisted AR (Liu et al. 2019)."),
  RH("2.4 Anytime answers.", "The server first returns a fast, coarse answer (a box from a light model) and then the refined one (the full grasp). The robot starts on the first and corrects on the second, so that under load a usable answer still arrives within the stage's delay budget. Related work: split computing and early exiting (Matsubara et al. 2022); ProgressFTX (Lan et al. 2022)."),
  RH("2.5 Speed that follows the link.", "The robot measures the end-to-end delay continuously and limits its speed so that v·d stays within the current stage's tolerance: fast on an idle link, slow only where the tolerance is tight and the link is busy. Motion compensation (2.1) removes the delay the robot can predict; speed adaptation absorbs the delay it cannot. Related work: networked control systems (Hespanha et al. 2007); communication-aware robotics (Muralidharan and Mostofi 2021)."),
  RH("2.6 Link configuration.", "Power saving off on the robot's Wi-Fi interface, the 5 GHz band pinned, request and answer traffic marked for the voice or video access category (WMM, via DSCP), and a datagram transport for images instead of a stream transport. These are configuration changes, measured like the others. Related work: IEEE 802.11e (WMM); FogROS2 (Ichnowski et al.)."),
  ...table("Table 2. The experiment: parts switched on cumulatively, each under the same delay sweep.", ["Step", "Configuration"], [
    ["1","Naive: every frame streamed, the latest answer acted upon"],
    ["2","+ motion-compensated answers (2.1)"],
    ["3","+ queue discipline (2.2)"],
    ["4","+ region and change uplink and early requests (2.3)"],
    ["5","+ anytime answers (2.4)"],
    ["6","+ speed that follows the link (2.5)"],
    ["—","each step with and without the link configuration (2.6)"],
  ], [1200, 8300], "Delay sweep with tc: 0, 50, 100, 200 and 500 ms, each with and without jitter and with and without a background stream. Measured per pick: gripper error at closure and its decomposition, first-attempt success, time per pick, bytes per pick. The main figure is gripper error against delay, one curve per step."),

  RH("What is measured.", "Stamps on one clock (chrony) at capture, send, server receive, after inference and at the robot's command; AprilTags on gripper and object for the error at the end of each stage. Per pick: the error per stage and the share attributable to delay; first-attempt success; time and bytes per stage. Two reference configurations frame the comparison: everything on the server except low-level control, and everything on the robot with lighter models."),
  H("3. The Literature by Category"),
  P("Tags: [F] read in full in this project; [V] venue or headline result checked against a listing or the paper's own page; [S] abstract, summary or memory only, to be checked before citation."),

  H3("3.1 Cloud and fog robotics: whether to offload"),
  P("Covers running robot computation on a remote server and the infrastructure for doing so."),
  ...lit("Table 3. Cloud and fog robotics.", [
    ["Kehoe, Patil, Abbeel, Goldberg, a survey of research on cloud robotics and automation","IEEE T-ASE 2015 [S]","the survey"],
    ["Chinchali et al., network offloading policies for cloud robotics","RSS 2019; Auton. Robots 2021 [V]","when to offload a sensing query, by deep RL; 1.3–2.3 times better task performance than benchmark policies"],
    ["FogROS2 (Ichnowski et al.)","arXiv 2205.09778 [S]","ROS 2 platform for cloud and fog robotics; video compression cuts image round-trip latency by 97 %"],
    ["FogROS2-LS","2024 [S]","selects the server that keeps latency reliable; offloads estimators and controllers"],
    ["Nakanoya et al., task-relevant representation learning for networked robotic perception","RSS 2021 [S]","learned task-relevant compression for robot uplink; record not found in this session"],
  ]),

  H3("3.2 Server-driven and adaptive video analytics: what to send"),
  P("Covers filtering, cropping and encoding a camera stream for a remote model under a bandwidth limit."),
  ...lit("Table 4. Server-driven and adaptive video analytics.", [
    ["DDS, server-driven video streaming for deep learning inference (Du et al.)","SIGCOMM 2020 [V]","the server's model asks for regions to re-send at high quality; up to 59 % less bandwidth at higher accuracy"],
    ["Reducto (Li et al.)","SIGCOMM 2020 [S]","on-camera frame filtering by feature difference"],
    ["Glimpse (Chen et al.)","SenSys 2015 [S]","track on the device, offload only when needed"],
    ["Liu et al., edge-assisted real-time object detection for mobile AR","MobiCom 2019 [S]","region-of-interest encoding and local tracking between server results"],
  ]),

  H3("3.3 Split computing and edge inference: where to cut"),
  P("Covers partitioning a neural network between a device and a server."),
  ...lit("Table 5. Split computing.", [
    ["Neurosurgeon (Kang et al.)","ASPLOS 2017 [S]","layer-level partitioning by latency and energy"],
    ["Matsubara, Levorato, Restuccia, split computing and early exiting","ACM CSUR 2022 [S]","the survey"],
  ]),

  H3("3.4 Base localisation"),
  P("Covers estimating the base pose in a map on the robot."),
  ...lit("Table 6. Base localisation.", [
    ["2Fast-2Lamaa (Le Gentil, Falque, Lisus, Barfoot)","IJRR 2026, arXiv 2410.05433 [F]","lidar–inertial localisation against a prebuilt map at under 16 % of a laptop CPU; evaluated on 3-D lidars only"],
    ["FAST-LIO2 (Xu et al.)","T-RO 2022 [S]","iterated Kalman-filter lidar–inertial odometry"],
    ["Thrun, Burgard, Fox, Probabilistic Robotics","MIT Press 2005","Monte Carlo localisation on a planar lidar, the Stretch default"],
  ]),

  H3("3.5 Grasp detection and visual servoing"),
  P("Covers choosing a grasp from a view and closing the loop from camera to joint."),
  ...lit("Table 7. Grasp detection and visual servoing.", [
    ["Contact-GraspNet (Sundermeyer et al.)","ICRA 2021 [V]","6-DoF parallel-jaw grasps from a depth image"],
    ["AnyGrasp (Fang et al.)","T-RO 2023 [V]","grasp perception in space and time; 93.3 % bin-clearing success on unseen objects"],
    ["GG-CNN (Morrison, Corke, Leitner)","RSS 2018 [S]","fast grasp network run in a closed loop"],
    ["Chaumette, Hutchinson, visual servo control I and II","IEEE RAM 2006, 2007","the tutorial"],
    ["Precise mobile manipulation of small everyday objects","arXiv 2502.13964, 2025 [S]","eye-in-hand servoing guided by vision foundation models on a mobile manipulator"],
  ]),

  H3("3.6 Mobile manipulation systems on Stretch"),
  P("Covers complete mobile-manipulation systems built on Stretch."),
  ...lit("Table 8. Systems on Stretch.", [
    ["Kemp, Edsinger, Clever, Matulevich, The Design of Stretch","ICRA 2022 [V]","the platform"],
    ["OK-Robot (Liu et al.)","arXiv 2401.12202, 2024 [V]","modular zero-shot pick-and-drop with models on a remote workstation; 58.5 % in 10 homes, 82.4 % in less cluttered rooms"],
    ["Dobb-E (Shafiullah et al.)","arXiv 2311.16098, 2023 [V]","imitation from a hand-held tool; 81 % over 109 tasks in 10 homes"],
    ["HomeRobot (Yenamandra et al.)","CoRL 2023 [S]","open-vocabulary mobile manipulation benchmark on Stretch"],
  ]),

  H3("3.7 Timeliness"),
  ...lit("Table 9. Timeliness.", [
    ["Maatouk, Kriouile, Assaad, Ephremides, age of incorrect information","ToN 2020","freshness that counts only while the receiver is wrong"],
    ["Sun, Polyanskiy, Uysal","T-IT 2020 [V]","signal-aware sampling over a random-delay channel"],
  ]),


  H3("3.8 Delay compensation, networked control and the link layer"),
  P("Covers acting on delayed information and prioritising traffic on the link."),
  ...lit("Table 10. Delay compensation, networked control and the link layer.", [
    ["Smith, closer control of loops with dead time","Chem. Eng. Prog. 1957","the Smith predictor: compensate a known delay with a model of the plant"],
    ["Hespanha, Naghshtabrizi, Xu, a survey of recent results in networked control systems","Proc. IEEE 2007","control over delayed, lossy networks"],
    ["Bejczy, Kim, Venema, the phantom robot: predictive displays for teleoperation with time delay","ICRA 1990 [S]","predict the robot's motion over the delay for the operator"],
    ["IEEE 802.11e (WMM)","standard","access categories for priority traffic"],
  ]),

  H("4. What Is Asked of the Collaborators"),
  bullet("The Stretch 4 configuration: lidar (planar or 3-D), head and wrist cameras, gripper, on-board compute, ROS 2 distribution."),
  bullet("The current pipeline: which models run on the server, their inference time and what they return; whether the grasp is open-loop from the head camera; which localisation runs and in which mode."),
  bullet("The object set, the surfaces and the success criterion."),
  bullet("Permission to run chrony and a logger on robot and server, to throttle the robot's link for experiments, and to mount AprilTags; three sessions of about two hours."),
];
const doc = new Document({
  styles: { default: { document: { run: { font: F, size: 21 } } },
    paragraphStyles: [
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: F, size: 23, bold: true, color: "000000" }, paragraph: { spacing: { before: 160, after: 60 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: F, size: 21, bold: true, color: "000000" }, paragraph: { spacing: { before: 120, after: 40 }, outlineLevel: 2 } } ] },
  numbering: { config: [
    { reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 480, hanging: 240 } } } }] },
    { reference: "n", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 480, hanging: 300 } } } }] } ] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 850, bottom: 800, left: 960, right: 960 } } }, children }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync('/home/user/Others/value_scheduling/docs/Stretch_Manipulation_Scenario_and_Literature.docx', b); console.log('ok'); });
