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
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:40}, children:[run("A Mobile Manipulator That Streams to a Server: Scenario and Literature",{bold:true,size:30})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:60}, children:[run("Student ID: 314540011",{size:21})] }),
  new Paragraph({ border:{ bottom:{ style:BorderStyle.SINGLE, size:8, color:"000000" } }, spacing:{after:160}, children:[] }),

  H("1. The Scenario"),
  P("One Stretch 4 mobile manipulator works in a lab or home on a pick task: drive to a room, approach a table, align the gripper, grasp, lift and verify. It carries a head RGB-D camera, a wrist RGB-D camera, a base lidar and an IMU. A server on the same Wi-Fi runs the heavy perception or reasoning model; the robot runs localisation (2Fast-2Lamaa against a prebuilt map) and its low-level controllers. The Wi-Fi is shared with the building's other traffic."),
  P("Every action closes a loop sense → send → server compute → return → act, and every loop has a deadline set by the motion it serves: navigation tolerates about half a second, the final grasp tolerates tens of milliseconds. The link and the server's queue both consume the deadline, and neither is under the robot's control. Streaming RGB-D at the camera's rate makes the server's input late exactly when the robot moves fastest or the network is busiest, and the robot then acts on a stale picture or stops."),
  P("The problem at every moment is the following."),
  quote("For the motion the robot is about to make, what does the server need to see, from which camera, in what form and when, and which parts of the loop should not involve the server at all, so that the robot's picture of the place it acts on is correct at the moment it commits?"),
  P("Four properties make the problem specific."),
  num("The deadline is set by the motion stage. Coarse motion (navigation) has a long deadline and a large region; fine motion (alignment, grasp) has a short deadline and a small region."),
  num("The server already holds most of the scene. Its last processed frame is a copy of the workspace, so only what changed since then carries information."),
  num("The loop can be split. Each stage can run on the robot, on the server, or partly on each, and the right split depends on the deadline, the robot's compute and the link."),
  num("The link and the server are shared and variable. Goodput, round-trip time and queue delay change from second to second."),
  RH("The quantity.", "Whether the robot's picture of the place it acts on is correct at the moment it commits (age of incorrect information at the decision), together with task success, time per stage and bytes per stage."),
  RH("The approach.", "Decide per stage. Offload what tolerates delay and needs the heavy model (navigation goal, grasp proposal, verification); keep on the robot what needs millimetres at tens of milliseconds (alignment servo, grasp closing); send only what differs from the server's copy, inside the region the stage needs, at the coarsest form that finishes before the stage's deadline; drop at the server any frame whose stage has ended."),

  ...table("Table 1. The stages of a pick and the communication each needs.", ["Stage", "Tolerance / deadline", "Server needs", "Sent", "Loop lives"], [
    ["1 Navigate","10–20 cm / 0.5–1 s","where the target is","low-resolution RGB-D about once per second, on change","server: semantic goal; robot: localisation and Nav2"],
    ["2 Approach","5 cm / 300–500 ms","target pose and grasp proposal","full-resolution crop of the table region, on change","server: detection and grasp; robot: base servo"],
    ["3 Align","1–2 cm / 100–300 ms","one refinement","changed pixels in a box around target and gripper, windowed depth","robot: wrist-depth servo; server: at most one refinement per second"],
    ["4 Grasp","2–5 mm / local","nothing","events only (contact, closed, slipped)","robot only: depth centroid and gripper effort"],
    ["5 Verify","1 cm / 300 ms","did it succeed","one crop after the lift","server or robot"],
  ], [1300, 1700, 1900, 2600, 2000], "Deadlines are starting values to be replaced by the collaborators' controller periods. Over one pick the traffic is of the order of 0.5 MB, against about 1 GB for streaming RGB-D at 30 frames per second through all stages."),

  RH("What is measured.", "Per pick: success; time per stage; bytes per stage; frames processed by the server after their stage had ended; alignment error at the moment of commit against the grasp outcome. Baseline: the collaborators' current stream at full rate through all stages."),
  RH("Implementation without training.", "Chrony on robot and server and stamps at send, receive, after inference and at the command; a change mask of the current frame against the server's last copy (RGB difference and a 2 cm depth difference); a region of interest per stage from kinematics and the server's last answer; a rung choice per frame from measured goodput, queue delay and the stage deadline; a stage machine whose transitions the server receives."),

  H("2. The Literature by Category"),
  P("Tags: [F] read in full in this project; [V] venue checked against a listing; [S] abstract or summary only, to be checked before citation."),

  H3("2.1 Cloud and fog robotics: whether to offload"),
  P("Owns the decision to run a computation on the robot or on a remote server, and the infrastructure to do so. Assumes away what part of the data the server already holds, and a deadline that changes with the motion stage."),
  ...lit("Table 2. Cloud and fog robotics.", [
    ["Kehoe, Patil, Abbeel, Goldberg, a survey of research on cloud robotics and automation","IEEE T-ASE 2015 [S]","the survey of the field"],
    ["Chinchali, Sharma, Harrison, Elhafsi, Kang, Pergament, Cidon, Katti, Pavone, network offloading policies for cloud robotics","RSS 2019; Auton. Robots 45(7), 2021 [V]","when to offload a sensing query, by deep RL; 1.3–2.3 times better task performance than benchmark offloading policies. The closest prior formulation"],
    ["Nakanoya et al., task-relevant representation learning for networked robotic perception","RSS 2021 [S]","learned task-relevant compression for robot uplink; record not found in this session"],
    ["FogROS2 (Ichnowski et al.)","arXiv 2205.09778; ICRA 2023 unconfirmed [S]","ROS 2 platform for cloud and fog robotics; video compression cuts image round-trip latency by 97 %"],
    ["FogROS2-LS","2024 [S]","selects the server that keeps latency reliable; offloads estimators and controllers"],
  ]),

  H3("2.2 Server-driven and adaptive video analytics: what to send for a remote model"),
  P("Owns how a camera stream is filtered, cropped and encoded so that a server-side model stays accurate under a bandwidth limit. Assumes away actuation: there is no robot, no motion stage, and no deadline set by an action."),
  ...lit("Table 3. Server-driven and adaptive video analytics.", [
    ["DDS, server-driven video streaming for deep learning inference (Du et al.)","SIGCOMM 2020 [V]","the server's model feeds back the regions to re-send at high quality; up to 59 % less bandwidth at higher accuracy. The closest prior mechanism"],
    ["Reducto, on-camera filtering for real-time video analytics (Li et al.)","SIGCOMM 2020 [S]","drops frames on the camera by low-level feature differences"],
    ["AWStream (Zhang et al.)","SIGCOMM 2018 [S]","adapts stream quality to wide-area bandwidth for analytics"],
    ["Chameleon (Jiang et al.)","SIGCOMM 2018 [S]","adapts the analytics configuration over time"],
    ["Glimpse (Chen et al.)","SenSys 2015 [S]","tracks on the device and offloads to the server only when needed"],
    ["Liu et al., edge-assisted real-time object detection for mobile AR","MobiCom 2019 [S]","encodes the region of interest at higher quality and tracks locally between server results"],
  ]),

  H3("2.3 Split computing and task-oriented edge inference: where to cut the model"),
  P("Owns partitioning a network between device and server and compressing the intermediate representation for a task. Assumes away the server's memory of earlier frames and stage-dependent deadlines."),
  ...lit("Table 4. Split computing and task-oriented edge inference.", [
    ["Neurosurgeon (Kang et al.)","ASPLOS 2017 [S]","layer-level partitioning between mobile and cloud by latency and energy"],
    ["Matsubara, Levorato, Restuccia, split computing and early exiting survey","ACM CSUR 2022 [S]","the survey"],
    ["Shao, Mao, Zhang, task-oriented communication for edge inference","JSAC 2022","information bottleneck for the uplink"],
    ["Lan, Zeng, Popovski, Gündüz, Huang, ProgressFTX","TWC 2022 [S]","progressive feature transmission with threshold stopping"],
  ]),

  H3("2.4 The platform and open-world mobile manipulation on it"),
  P("Owns the robot and the systems that run large models for it. Treats the network as a given and the server as reachable."),
  ...lit("Table 5. Stretch and open-world mobile manipulation.", [
    ["Kemp, Edsinger, Clever, Matulevich, The Design of Stretch","ICRA 2022 [V]","the platform: two-wheeled base, lift, telescoping arm, compliant gripper"],
    ["OK-Robot (Liu et al.)","arXiv 2401.12202, 2024 [S]","open-knowledge models on Stretch with a remote GPU workstation"],
    ["HomeRobot (Yenamandra et al.)","CoRL 2023 [S]","open-vocabulary mobile manipulation benchmark on Stretch"],
  ]),

  H3("2.5 Localisation and on-board state estimation"),
  P("Owns the part of the loop that must run on the robot."),
  ...lit("Table 6. Localisation.", [
    ["2Fast-2Lamaa (Le Gentil, Falque, Lisus, Barfoot)","IJRR 2026, arXiv 2410.05433 [F]","lidar-inertial localisation against a prebuilt map at under 16 % of a laptop CPU and 0.17–0.33 GB RAM; evaluated on 3-D spinning lidars, so results on a planar lidar are not established"],
  ]),

  H3("2.6 Timeliness and event-triggered sending"),
  P("Owns when to send and how to score freshness."),
  ...lit("Table 7. Timeliness.", [
    ["Maatouk, Kriouile, Assaad, Ephremides, age of incorrect information","ToN 2020","freshness that counts only while the receiver is wrong"],
    ["Sun, Polyanskiy, Uysal","T-IT 2020 [V]","signal-aware sampling over a random-delay channel"],
    ["Trimpe, D'Andrea, event-based state estimation","TAC 2014","send on innovation; silence carries information"],
  ]),

  H("3. Where the Gaps Are"),
  ...table("Table 8. The four properties of the scenario against the nearest prior work.", ["Property", "Nearest prior", "What is missing"], [
    ["deadline set by the motion stage","Chinchali's offloading cost; FogROS2-LS latency targets","a deadline that tightens from navigation to grasp and changes what is sent and where the loop runs"],
    ["the server already holds the scene","DDS feedback regions; Reducto frame dropping; Glimpse local tracking","the server's last copy as the reference for a change-based uplink inside a manipulation loop"],
    ["the loop can be split","Neurosurgeon and split computing (one model, one cut); Glimpse","a split that moves per stage, with the fine loop entirely on the robot and the server reduced to events"],
    ["shared, variable link and server","AWStream, Chameleon (bandwidth); FogROS2-LS (server choice)","scoring at the moment the robot commits, rather than by frame accuracy or average latency"],
  ], [2300, 3400, 3800]),
  P("In short: the video-analytics line knows how to send less to a remote model but has no robot and no deadline from an action; the cloud-robotics line knows how to decide whether to offload but treats each query as a whole frame and scores average performance; neither moves the split of the loop with the precision of the motion, and neither scores whether the robot's picture was correct when it acted. DDS and Chinchali et al. are the two papers to position against first."),

  H("4. What Is Asked of the Collaborators"),
  bullet("The Stretch 4 configuration: cameras, lidar (planar or 3-D), on-board compute, ROS 2 distribution, and the topics and encodings streamed today."),
  bullet("The server model or models, their inference time, what they return, and how often their output is used."),
  bullet("The pick task's success criterion and, if defined, the controller period of each stage."),
  bullet("Permission to run chrony and a logger on robot and server, and a short recording of one pick (head and wrist streams, lidar, IMU, joint states, gripper effort)."),
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
Packer.toBuffer(doc).then(b => { fs.writeFileSync('/home/user/Others/value_scheduling/docs/Stretch_Scenario_and_Literature.docx', b); console.log('ok'); });
