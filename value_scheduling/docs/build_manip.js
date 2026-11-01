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
  RH("The approach.", "Write the pick as a chain of stages with an error budget in which delay is converted to distance. Offload a stage only when its tolerance divided by the relative speed exceeds the measured end-to-end delay with margin; slow the motion when it does not; send to the server only the region the stage needs and only what changed since the server's last copy; keep the alignment and closure loops on the robot, steering base yaw and arm extension from the wrist camera; switch from head to wrist at the distance where head-camera error plus delay error exceeds the remaining tolerance."),

  ...table("Table 1. The stages of a pick: tolerance, delay budget, where each runs and what crosses the link.", ["Stage", "Must be within", "Delay budget at its speed", "Runs on", "Sent over the link"], [
    ["1 Localise and drive","5–10 cm","≈ 0.2–0.5 s at 0.3 m/s","robot: lidar–inertial localisation, base planner; server: semantic goal once","a low-resolution frame on request"],
    ["2 Find the object","2–3 cm","≈ 0.3 s when the base is stopped, near zero while moving","server: detection and grasp proposal","one full-resolution crop of the target region"],
    ["3 Pre-grasp","1–2 cm","≈ 0.1–0.2 s at 0.1 m/s arm speed","robot: arm on the stage 2 estimate; server: optional refinement","changed pixels in the target box, if refined"],
    ["4 Align","2–5 mm","≈ 50 ms at 5 cm/s, below a Wi-Fi round trip under load","robot only: wrist-camera servo on base yaw and arm extension","nothing"],
    ["5 Close and verify","—","≈ 0.3 s","robot: effort; server: verification","events; one crop after the lift"],
  ], [1300, 1250, 2200, 2650, 2100], "Delay budget = tolerance / relative speed. Values are starting points to be replaced by measurement; the loaded round trip measured on the group's own testbed was 25–33 ms, rising to 60–109 ms on a busier pass, before any server queue or inference time."),

  RH("What is measured.", "Stamps on one clock (chrony) at capture, send, server receive, after inference and at the robot's command; AprilTags on gripper and object for the error at the end of each stage; the link throttled with tc and loaded by a second stream to sweep the delay. Per pick: the error per stage and the share attributable to delay; first-attempt success; time and bytes per stage."),
  RH("Comparisons.", "(a) Everything on the server except low-level control; (b) everything on the robot with lighter models; (c) the per-stage split of Table 1. Each under idle and loaded Wi-Fi. Within (c): open-loop grasp from the head camera against wrist servo; hand-off distance swept from 40 cm to 10 cm; full frames against region-and-change uplink."),
  RH("What it delivers.", "A measured error budget from map to fingertip in which the link's delay is a term like any sensor's; the rule for which stage to offload given the measured delay; and the hand-off distance that maximises first-attempt success under a given link."),

  H("2. The Literature by Category"),
  P("Tags: [F] read in full in this project; [V] venue or headline result checked against a listing or the paper's own page; [S] abstract, summary or memory only, to be checked before citation."),

  H3("2.1 Cloud and fog robotics: whether to offload"),
  P("Owns the decision to run a computation remotely and the infrastructure for it. Scores average task performance or latency; does not turn delay into an error at the gripper or let the split change with the motion stage."),
  ...lit("Table 2. Cloud and fog robotics.", [
    ["Kehoe, Patil, Abbeel, Goldberg, a survey of research on cloud robotics and automation","IEEE T-ASE 2015 [S]","the survey"],
    ["Chinchali et al., network offloading policies for cloud robotics","RSS 2019; Auton. Robots 2021 [V]","when to offload a sensing query, by deep RL; 1.3–2.3 times better task performance than benchmark policies"],
    ["FogROS2 (Ichnowski et al.)","arXiv 2205.09778 [S]","ROS 2 platform for cloud and fog robotics; video compression cuts image round-trip latency by 97 %"],
    ["FogROS2-LS","2024 [S]","selects the server that keeps latency reliable; offloads estimators and controllers"],
    ["Nakanoya et al., task-relevant representation learning for networked robotic perception","RSS 2021 [S]","learned task-relevant compression for robot uplink; record not found in this session"],
  ]),

  H3("2.2 Server-driven and adaptive video analytics: what to send"),
  P("Owns how a camera stream is filtered, cropped and encoded for a remote model under a bandwidth limit. Has no actuator: accuracy is per frame, not at a commit."),
  ...lit("Table 3. Server-driven and adaptive video analytics.", [
    ["DDS, server-driven video streaming for deep learning inference (Du et al.)","SIGCOMM 2020 [V]","the server's model asks for regions to re-send at high quality; up to 59 % less bandwidth at higher accuracy"],
    ["Reducto (Li et al.)","SIGCOMM 2020 [S]","on-camera frame filtering by feature difference"],
    ["Glimpse (Chen et al.)","SenSys 2015 [S]","track on the device, offload only when needed"],
    ["Liu et al., edge-assisted real-time object detection for mobile AR","MobiCom 2019 [S]","region-of-interest encoding and local tracking between server results"],
  ]),

  H3("2.3 Split computing and edge inference: where to cut"),
  P("Owns partitioning a model between device and server. One model, one cut, no motion."),
  ...lit("Table 4. Split computing.", [
    ["Neurosurgeon (Kang et al.)","ASPLOS 2017 [S]","layer-level partitioning by latency and energy"],
    ["Matsubara, Levorato, Restuccia, split computing and early exiting","ACM CSUR 2022 [S]","the survey"],
  ]),

  H3("2.4 Base localisation"),
  P("Owns where the base is. Reports trajectory error, not the error left at the gripper; runs on the robot."),
  ...lit("Table 5. Base localisation.", [
    ["2Fast-2Lamaa (Le Gentil, Falque, Lisus, Barfoot)","IJRR 2026, arXiv 2410.05433 [F]","lidar–inertial localisation against a prebuilt map at under 16 % of a laptop CPU; evaluated on 3-D lidars only"],
    ["FAST-LIO2 (Xu et al.)","T-RO 2022 [S]","iterated Kalman-filter lidar–inertial odometry"],
    ["Thrun, Burgard, Fox, Probabilistic Robotics","MIT Press 2005","Monte Carlo localisation on a planar lidar, the Stretch default"],
  ]),

  H3("2.5 Grasp detection and visual servoing"),
  P("Owns where to grasp and the closed loop that gets there. Assumes the model runs where the camera is, or that the link is ideal."),
  ...lit("Table 6. Grasp detection and visual servoing.", [
    ["Contact-GraspNet (Sundermeyer et al.)","ICRA 2021 [V]","6-DoF parallel-jaw grasps from a depth image"],
    ["AnyGrasp (Fang et al.)","T-RO 2023 [V]","grasp perception in space and time; 93.3 % bin-clearing success on unseen objects"],
    ["GG-CNN (Morrison, Corke, Leitner)","RSS 2018 [S]","fast grasp network run in a closed loop"],
    ["Chaumette, Hutchinson, visual servo control I and II","IEEE RAM 2006, 2007","the tutorial"],
    ["Precise mobile manipulation of small everyday objects","arXiv 2502.13964, 2025 [S]","eye-in-hand servoing guided by vision foundation models on a mobile manipulator"],
  ]),

  H3("2.6 Mobile manipulation systems on Stretch"),
  P("Own the whole pipeline in real homes, typically with a GPU workstation; report overall success, not where in the chain or in the link failures arise."),
  ...lit("Table 7. Systems on Stretch.", [
    ["Kemp, Edsinger, Clever, Matulevich, The Design of Stretch","ICRA 2022 [V]","the platform"],
    ["OK-Robot (Liu et al.)","arXiv 2401.12202, 2024 [V]","modular zero-shot pick-and-drop with models on a remote workstation; 58.5 % in 10 homes, 82.4 % in less cluttered rooms"],
    ["Dobb-E (Shafiullah et al.)","arXiv 2311.16098, 2023 [V]","imitation from a hand-held tool; 81 % over 109 tasks in 10 homes"],
    ["HomeRobot (Yenamandra et al.)","CoRL 2023 [S]","open-vocabulary mobile manipulation benchmark on Stretch"],
  ]),

  H3("2.7 Timeliness"),
  ...lit("Table 8. Timeliness.", [
    ["Maatouk, Kriouile, Assaad, Ephremides, age of incorrect information","ToN 2020","freshness that counts only while the receiver is wrong"],
    ["Sun, Polyanskiy, Uysal","T-IT 2020 [V]","signal-aware sampling over a random-delay channel"],
  ]),

  H("3. Where the Gaps Are"),
  ...table("Table 9. The four properties of the scenario against the nearest prior work.", ["Property", "Nearest prior", "What is missing"], [
    ["errors compound, and the link is one of them","localisation (ATE); grasp detection (success on fixed arms); offloading (average accuracy)","one error budget from map to fingertip in which delay is converted to distance at the gripper"],
    ["tolerance and delay budget shrink together","FogROS2-LS latency targets; coarse-then-fine grasping","stage-wise delay budgets derived from tolerance and speed, and the measured point where offloading stops paying"],
    ["where a stage runs is a choice","Chinchali's offloading policy; Neurosurgeon's cut; Glimpse","a split that moves with the motion stage, scored at the grasp rather than per query"],
    ["the Cartesian arm needs the base","whole-body servoing on arms with lateral joints; OK-Robot open-loop grasping","an on-robot base-yaw and arm-extension servo for the stages no round trip can serve, with the server reduced to proposals and verification"],
  ], [2300, 3400, 3800]),
  P("In short: offloading and video-analytics work optimises what crosses the link without an actuator at the other end; manipulation work optimises the grasp without a link in the loop; systems on Stretch use a remote workstation and report overall success. Converting delay into gripper error closes the space between them. Chinchali et al., DDS and OK-Robot are the three to position against first."),

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
