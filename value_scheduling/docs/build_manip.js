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
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:40}, children:[run("From Room-Level Localisation to a Millimetre Grasp: Scenario and Literature",{bold:true,size:30})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:60}, children:[run("Student ID: 314540011",{size:21})] }),
  new Paragraph({ border:{ bottom:{ style:BorderStyle.SINGLE, size:8, color:"000000" } }, spacing:{after:160}, children:[] }),

  H("1. The Scenario"),
  P("One Stretch 4 mobile manipulator picks a small everyday object (a cup, a pen, a remote control) from a table, a shelf or an open drawer in a home or a lab, starting from somewhere else in the building. It carries a base lidar and IMU, a head RGB-D camera on a pan–tilt mount, a wrist RGB-D camera next to the gripper, joint encoders and gripper effort sensing. Its arm is Cartesian: a lift, a telescoping arm and a wrist; there is no lateral joint, so any sideways correction of the gripper must come from rotating or driving the base."),
  P("The task spans from building scale to millimetres. The robot localises in a building-scale map to a few centimetres, chooses where to stand, finds the object with the head camera from one to two metres away, moves the arm close to it, and must finally place the fingers around the object to within a few millimetres. Each step hands the target to the next with an error, and the errors add up along the chain."),
  P("The problem the robot faces is the following."),
  quote("Given an error at every link from map to fingertip, which sensor and which loop should be in charge at each stage, when should control pass from one to the next, and how should base and arm share the correction, so that the gripper closes within the object's tolerance on the first attempt?"),
  P("Four properties make the problem specific."),
  num("Errors compound along the chain. Localisation, base execution, head-camera depth, head pan–tilt calibration, arm deflection and wrist-camera calibration each add an error; the grasp only sees their sum."),
  num("Tolerance shrinks stage by stage, from about 10 cm when stopping near the table to a few millimetres when closing the gripper; a stage is good enough only relative to the tolerance of the next one."),
  num("Sensors hand off. The lidar sees the room, the head camera sees the table from afar with depth error that grows with range, the wrist camera sees the object at close range but is occluded by the gripper and loses the object at the last centimetres."),
  num("The Cartesian arm needs the base. Lateral and yaw corrections at the gripper are base motions, so the final alignment is a coordinated base–arm problem on a differential drive."),
  RH("The quantity.", "The position error of the gripper relative to the object at the moment of closure, and its decomposition into the contribution of each stage, measured against a fiducial on the object; together with first-attempt grasp success and time per stage."),
  RH("The approach.", "Treat the pick as a chain of stages with an explicit error budget; give each stage to the sensor and loop that can meet the next stage's tolerance; switch from open-loop to closed-loop control at the distance where the head camera's error exceeds the remaining tolerance; correct laterally with the base and longitudinally with the telescoping arm in one servo loop on the wrist camera; close with effort feedback."),

  ...table("Table 1. The stages of a pick, the error each must stay within, and who is in charge.", ["Stage", "Reaches", "Must be within", "In charge", "Main error sources"], [
    ["1 Localise and drive","standing pose near the target surface","5–10 cm, a few degrees","lidar–inertial localisation in the prebuilt map; base planner","map error, localisation drift, base execution"],
    ["2 Find the object","object pose and a grasp proposal","2–3 cm","head RGB-D, open-vocabulary detector, grasp detector","depth error growing with range, pan–tilt calibration, segmentation edges"],
    ["3 Pre-grasp","gripper 10–20 cm in front of the object","1–2 cm","lift, arm and wrist, open loop on the stage 2 estimate; base for yaw","stage 2 error, arm deflection when extended, base slip"],
    ["4 Align","gripper around the object","2–5 mm","wrist RGB-D closed-loop servo on base yaw and arm extension","wrist calibration, occlusion by the gripper, servo convergence"],
    ["5 Close and verify","object held","—","gripper effort; wrist camera after lift","object compliance, slip"],
  ], [1450, 1950, 1350, 2400, 2350], "Tolerances are starting values for a parallel gripper and objects of 3–8 cm width; each is replaced by a measured value in the first session."),

  RH("What is measured.", "An AprilTag on the object and on the gripper gives, at the end of every stage, the gripper-to-object error, and the reference map gives the base pose error. Per pick: the error at the end of each stage; first-attempt success; time per stage; the distance at which control passed from head to wrist. Comparisons: open-loop grasp from the head camera (the modular baseline) against wrist servo; three localisation sources (wheel odometry only, AMCL on the planar lidar, 2Fast-2Lamaa); hand-off distance swept from 40 cm to 10 cm; arm extension short against long."),
  RH("What it delivers.", "A measured error budget from map to fingertip for a Cartesian mobile manipulator, the hand-off distance that maximises first-attempt success, and the share of failures attributable to each stage."),

  H("2. The Literature by Category"),
  P("Tags: [F] read in full in this project; [V] venue or headline result checked against a listing or the paper's own page; [S] abstract, summary or memory only, to be checked before citation."),

  H3("2.1 Localisation of the base"),
  P("Owns where the base is in the map. Reports trajectory error (ATE, RPE); does not report the error the localisation leaves at the gripper or at the object."),
  ...lit("Table 2. Base localisation.", [
    ["2Fast-2Lamaa (Le Gentil, Falque, Lisus, Barfoot)","IJRR 2026, arXiv 2410.05433 [F]","lidar–inertial localisation against a prebuilt map with continuous distance fields; centimetre-level RMSE on automotive data at under 16 % of a laptop CPU; evaluated on 3-D spinning lidars only"],
    ["FAST-LIO2 (Xu et al.)","T-RO 2022 [S]","iterated Kalman-filter lidar–inertial odometry; the common baseline"],
    ["KISS-ICP (Vizzo et al.)","RA-L 2023 [S]","point-to-point ICP odometry with minimal tuning"],
    ["Thrun, Burgard, Fox, Probabilistic Robotics","MIT Press 2005","Monte Carlo localisation (AMCL) on a planar lidar, the Stretch default"],
    ["Furgale, Barfoot, visual teach and repeat","JFR 2010 [S]","topometric localisation for repeated routes"],
  ]),

  H3("2.2 Where to stand: base placement and whole-body motion"),
  P("Owns the base pose from which the arm can reach the object. Assumes the base pose is achieved exactly."),
  ...lit("Table 3. Base placement and whole-body mobile manipulation.", [
    ["Vahrenkamp, Asfour, Dillmann, robot placement based on reachability inversion","ICRA 2013 [S]","base poses from an inverted reachability map"],
    ["Kemp, Edsinger, Clever, Matulevich, The Design of Stretch","ICRA 2022 [V]","the platform: differential base, lift, telescoping arm; lateral reach through base rotation"],
    ["Whole-body visual servoing of nonholonomic mobile manipulators","2019 [S]","global Jacobian of base and arm with a Kalman filter fusing position and image"],
  ]),

  H3("2.3 Grasp detection"),
  P("Owns where on the object to grasp, from a single view. Assumes the gripper reaches the proposed pose exactly; evaluated mostly on fixed arms."),
  ...lit("Table 4. Grasp detection.", [
    ["Dex-Net 2.0 (Mahler et al.)","RSS 2017 [S]","grasp-quality network trained on synthetic point clouds"],
    ["GG-CNN (Morrison, Corke, Leitner)","RSS 2018 [S]","fast generative grasp network run in a closed loop"],
    ["Contact-GraspNet (Sundermeyer et al.)","ICRA 2021 [V]","6-DoF parallel-jaw grasps from a depth image; trained on 17 million simulated grasps"],
    ["AnyGrasp (Fang et al.)","T-RO 2023 [V]","grasp perception in space and time; 93.3 % bin-clearing success on over 300 unseen objects"],
  ]),

  H3("2.4 Visual servoing and coarse-to-fine grasping"),
  P("Owns the closed loop from camera to joint. Rarely quantifies how much error the preceding open-loop stages leave for it to absorb."),
  ...lit("Table 5. Visual servoing.", [
    ["Chaumette, Hutchinson, visual servo control, parts I and II","IEEE RAM 2006, 2007","the tutorial: image-based and position-based servoing"],
    ["Precise mobile manipulation of small everyday objects (servoing with vision models)","arXiv 2502.13964, 2025 [S]","eye-in-hand servoing guided by vision foundation models on a mobile manipulator; ten real environments; notes occlusion of the scene by the manipulator"],
    ["Coarse positioning then fine grasping with monocular vision","Int. J. Adv. Manuf. Technol. 2022 [S]","base navigates to the area, then detection and visual servoing with PnP"],
    ["Uncalibrated eye-in-hand mobile manipulation","Artif. Life Robot. 2017 [S]","attributes pick-and-place failures to base and object pose error"],
  ]),

  H3("2.5 Open-world mobile manipulation on Stretch"),
  P("Owns the complete system in real homes. Reports overall success; the error budget and the stage at which failures arise are discussed qualitatively."),
  ...lit("Table 6. Mobile manipulation systems on Stretch.", [
    ["OK-Robot (Liu et al.)","arXiv 2401.12202, 2024 [V]","modular zero-shot pick-and-drop with open-knowledge models; 58.5 % in 10 homes, 82.4 % in less cluttered rooms; grasp from the head-camera view"],
    ["Dobb-E, On Bringing Robots Home (Shafiullah et al.)","arXiv 2311.16098, 2023 [V]","imitation from a hand-held tool; 81 % over 109 tasks in 10 homes after five minutes of demonstrations per task"],
    ["Robot Utility Models (Etukuru et al.)","2024 [S]","zero-shot policies on Stretch; about 74 % rising to 90 % with a vision-language success check (secondary source)"],
    ["HomeRobot (Yenamandra et al.)","CoRL 2023 [S]","open-vocabulary mobile manipulation benchmark on Stretch"],
  ]),

  H3("2.6 Learned visuomotor policies"),
  P("Own the final centimetres end to end from demonstrations. Need data per task and do not use the map."),
  ...lit("Table 7. Learned visuomotor policies.", [
    ["Diffusion Policy (Chi et al.)","RSS 2023 [S]","diffusion over action sequences"],
    ["ACT (Zhao et al.)","RSS 2023 [S]","action chunking with transformers"],
    ["OpenVLA (Kim et al.)","2024 [S]","open vision-language-action model"],
  ]),

  H3("2.7 Calibration"),
  P("Owns the fixed transforms between cameras, arm and base."),
  ...lit("Table 8. Calibration.", [
    ["Tsai, Lenz, hand–eye calibration","IEEE T-RA 1989","the classical solution"],
    ["Park, Martin, robot sensor calibration","IEEE T-RA 1994","AX = XB on the Euclidean group"],
  ]),

  H("3. Where the Gaps Are"),
  ...table("Table 9. The four properties of the scenario against the nearest prior work.", ["Property", "Nearest prior", "What is missing"], [
    ["errors compound along the chain","localisation papers (ATE); grasp papers (success on fixed arms)","an error budget from map to fingertip on one robot, with each stage's share measured"],
    ["tolerance shrinks stage by stage","OK-Robot's modular pipeline; coarse-then-fine grasping","stage tolerances derived from the next stage's needs, and the measured hand-off distance that maximises first-attempt success"],
    ["sensors hand off","eye-in-hand servoing with vision models; GG-CNN closed loop","when to pass from the head camera to the wrist camera, given range-dependent depth error and close-range occlusion"],
    ["the Cartesian arm needs the base","whole-body servoing on arms with lateral joints","coordinated base-yaw and arm-extension servoing for a differential base with a telescoping arm"],
  ], [2300, 3400, 3800]),
  P("In short: localisation work stops at the base and grasp work starts at a fixed arm; mobile-manipulation systems on Stretch report overall success but not where in the chain the error came from; servoing work closes the last loop without saying how much error the open-loop stages hand it. The study fills the space between them with a measured budget and a hand-off rule. OK-Robot and the servoing-with-vision-models paper are the two to position against first."),

  H("4. What Is Asked of the Collaborators"),
  bullet("The Stretch 4 configuration: lidar (planar or 3-D), head and wrist cameras, gripper type, on-board compute, ROS 2 distribution."),
  bullet("The current pick pipeline: which detector and grasp method, whether the grasp is open-loop from the head camera, and which localisation they run (AMCL or 2Fast-2Lamaa, and in which mode)."),
  bullet("The object set and the surfaces (table, shelf, drawer) the task should cover, and their success criterion."),
  bullet("Permission to mount AprilTags on the gripper and objects, and three sessions of about two hours: error budget, hand-off sweep, and the final comparison."),
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
