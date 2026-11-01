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
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:40}, children:[run("Communication-Aware Collaborative Robotics",{size:20})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:40}, children:[run("The Scenario, and the Literature by Category",{bold:true,size:30})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:60}, children:[run("Student ID: 314540011",{size:21})] }),
  new Paragraph({ border:{ bottom:{ style:BorderStyle.SINGLE, size:8, color:"000000" } }, spacing:{after:160}, children:[] }),

  H("1. The Scenario"),
  P("A robot team on shared Wi-Fi in a changing building, keeping each robot's picture of the places its next move depends on correct, in time."),
  P("Some robots move through the building on jobs; some fixed cameras watch parts of it; all share one access point with the building's other traffic. People walk, doors open and close, things get moved. Each robot is about to act somewhere: turn a corner, enter a room, cross a junction. To act well it needs a correct picture of the places its next move depends on. Some it will see for itself in time; some are hidden and it can only learn about them from another robot or a camera that sees them now. Every observation is about a place, taken at a time, and stays true for a while: a wall for ever, a parked cart for minutes, a walking person for a second or two."),
  P("The problem the team faces at every moment is the following."),
  quote("Of everything anyone has seen, which observations should go to which robot, in what form, sent by whom, and when, so that every robot has a correct picture of the places its next move depends on by the time it makes that move, given that the link cannot carry everything and the robots keep moving?"),
  P("Four properties make the problem hard, and they are the four axes along which the literature is organised below."),
  num("Deadlines come from motion. When a robot needs a place is set by how fast it moves toward it."),
  num("Validity comes from the world. How long an observation stays true depends on what is in it and how busy the place is."),
  num("Need comes from the receiver's own future. A place the robot will see itself before it matters needs no message."),
  num("The channel is shared and short. One access point; messages collide; bulk traffic delays small urgent messages."),
  RH("The quantity.", "Everything is scored by the age of incorrect information at the decision (AoII at the decision): how long a robot's belief about a place has been wrong at the moment its move depends on that place."),
  RH("The approach.", "Decide what to send by when it is needed and how long it stays true; take turns on the channel; let the robot move to keep its picture right; score at the moment it acts."),
  RH("Experiments, in order of size.", "One blind corner (one mast, one robot, a loading robot; SCENARIO_0.md); the crossing (three masts, two robots, a throttled link; SCENARIO_CROSSING.md); the changed building (a moved cabinet, a closed door); the full rig under pose and control deadlines."),
  RH("Rig.", "Two Scout Mini robots (Jetson Orin NX, Ouster OS1, ZED 2i, two TI radars), three Raspberry Pi masts with cameras, one Asus RT-BE82U access point, one server, chrony (measured offset at most 0.99 ms), one monitor-mode sniffer, and the reference map of site 1 preloaded on every node. No digital twin, no ray tracer, and no labels beyond who was where per trial."),

  H("2. The Literature by Category"),
  P("Each category owns one of the four axes and assumes the others away. Tags: [F] read in full in this project (PDFs in the refs folder); [V] venue checked against a listing; [S] abstract or summary only, to be checked before citation. Entries without a tag are classical results whose records are well known; the journal volume and pages of the 2004 and 2008 information-theory entries were not checked here."),

  H3("2.1 Collaborative perception: what to send"),
  P("Owns which cells, features or objects are sent, and how few bytes. Assumes away the channel (a byte budget or a constant delay), the receiver's future, deadlines, and other senders."),
  ...lit("Table 1. Collaborative perception.", [
    ["Where2comm (Hu, Fang, Lei, Zhong, Chen)","NeurIPS 2022","spatial confidence and request map; the canonical baseline"],
    ["V2X-ViT (Xu et al.)","ECCV 2022","delay-aware fusion; constant 100 ms delay"],
    ["CoBEVT (Xu et al.)","CoRL 2022 [S]","sparse BEV fusion"],
    ["CodeFilling (Hu et al.)","CVPR 2024 [S]","codebook messages"],
    ["CoSDH (Xu et al.)","CVPR 2025 [S]","supply–demand masks; intermediate–late hybrid"],
    ["ERMVP (Zhang et al.)","CVPR 2024 [S]","efficient and robust messages"],
    ["Chiu, Smith, selective communication","2023 [S]","object centres first, detail on request"],
    ["InfoCom","AAAI 2026 [S]","kilobyte messages by information bottleneck"],
    ["RDComm (Liu, Hu, Hu, Zhang, Chen)","ICLR 2026 [V]","pragmatic rate–distortion; ICLR title differs from arXiv"],
    ["WhisperNet (Chen, Zhang, Zhao)","CVPR 2026 [S]","sender importance and receiver coordination at 1 % bandwidth"],
    ["Dense Coverage, Sparse Refinement (Yazgan, Mueller, Zoellner)","arXiv 2609.29456, 2026 [F]","coarse base plus 198-byte patches ranked by learned benefit; strict byte caps; cross-agent redundancy stated as open"],
    ["JigsawComm (Wang, Li, Li, Wen)","arXiv 2511.17843, 2026 [F]","learned per-cell utility, top-1 sender per cell, O(1) payload in N; decision identical at every agent; a second sender per cell lowers accuracy"],
    ["SparseCoop (Wang et al.)","AAAI 2026 [V]","instance queries with explicit position and velocity"],
    ["TraF-Align (Song, Yang, Wen, Li)","CVPR 2025 [V]","trajectory-aware feature alignment under latency"],
    ["CoBEVFlow (Wei et al.)","NeurIPS 2023","BEV flow for asynchrony"],
    ["FFNet (Yu et al.)","NeurIPS 2023 [S]","sender-side feature flow"],
    ["SyncNet (Lei et al.)","ECCV 2022","latency-aware fusion"],
    ["Reason-to-Transmit (Bansal, Gangwani)","arXiv 2603.20308, 2026 [F]","transformer selection with a neighbour-state token; selection policies within about 0.5 AP of each other; fusion dominates"],
  ], "An indoor ground-robot benchmark with a density-guided residual fusion operator was seen in confidence while under review; it is not citable and is omitted from the table."),

  H3("2.2 Task-oriented and semantic communication: how to encode"),
  P("Owns bits that preserve a task loss over a noisy physical channel. Assumes away the medium-access layer (no contention, no queues), the receiver's state, and time."),
  ...lit("Table 2. Task-oriented and semantic communication.", [
    ["Gündüz et al., Beyond Transmitting Bits","JSAC 2023","the survey"],
    ["Shao, Mao, Zhang, task-oriented communication for edge inference","JSAC 2022","information bottleneck"],
    ["Shao, Mao, Zhang, multi-device cooperative edge inference","TWC 2023","distributed information bottleneck"],
    ["Lan, Zeng, Popovski, Gündüz, Huang, ProgressFTX","TWC 2022 [S]","progressive feature transmission with threshold stopping"],
    ["SComCP (Gan et al.)","arXiv 2507.00895, 2025 [V]","semantic communication for collaborative perception over a PHY channel"],
    ["CoDS (Gan, Liang, Zhang, Guo, Jin)","arXiv 2512.22513, 2025 [V]","digital semantic communication for collaborative perception"],
    ["R-ACP (Fang et al.)","JSAC 2025 [V]","capacity-adaptive compression with priority scheduling on real vehicles"],
    ["Token Communications (Qiao et al.)","IEEE Wireless Commun. 2025 [S]","tokens as the transmission unit"],
  ]),
  P("Foundations used by this category and by the present work: Equitz and Cover (T-IT 1991) and Steinberg and Merhav (T-IT 2004) on successive refinement, with and without side information; Tian, Steiner, Shamai and Diggavi (T-IT 2008) on the broadcast approach with expected distortion."),

  H3("2.3 Timeliness: when to update"),
  P("Owns age, freshness, and scheduling theory with bounds. Assumes away content (every update ages alike), the receiver's own sensing (partly addressed by Update the Unseen Only), and decision deadlines."),
  ...lit("Table 3. Timeliness.", [
    ["Kaul, Yates, Gruteser","INFOCOM 2012","the origin of the age of information"],
    ["Yates, Sun, Brown, Kaul, Modiano, Ulukus","JSAC 2021","survey"],
    ["Kadota, Sinha, Uysal-Biyikoglu, Singh, Modiano","ToN 2018","AoI scheduling; Whittle index and max-weight"],
    ["Sun, Polyanskiy, Uysal","T-IT 2020 [V]","signal-aware sampling beats age-optimal sampling"],
    ["Maatouk, Kriouile, Assaad, Ephremides","ToN 2020","age of incorrect information"],
    ["Fresh2comm (Wu, Peng, Yu)","arXiv 2502.07852, 2025 [S]","AoI for collaborative perception; author list from a secondary citation"],
    ["TAMP (Zhu, Sun, Jia, Chen, Ai, Zhou)","arXiv 2601.04542, 2026 [V]","Lyapunov region scheduling with an age-to-accuracy penalty"],
    ["AoI-guaranteed scheduling for CP (Wang/Xu, Sun, Zhou, Niu)","WiOpt 2023 [S]","centre-side resource allocation under age constraints; author list differs between the arXiv and WiOpt versions"],
    ["Han, Wang, Liwang, Niyato","TCCN 2026 [V]","AoI derived from message size and link, weighting fusion"],
    ["Update the Unseen Only (Ma, Zhao, Fang, An, Chen, Fang)","arXiv 2607.20967, 2026 [F]","age over areas users cannot sense; INAR persistence model; LocMW with sublinear regret"],
    ["WiSwarm (Tripathi et al.)","INFOCOM 2023 [S]","AoI scheduling on a real multi-UAV Wi-Fi rig"],
    ["Tripathi et al., computation and communication co-design","WiOpt 2021 [S]","joint compute and channel scheduling by AoI; occupancy-grid mapping"],
  ]),

  H3("2.4 Mobile systems: who sends to whom on a real radio"),
  P("Owns relevance, visibility, bandwidth partitioning and measured latency on real radios. Assumes away spoil times from content, decision-time metrics, and motion as a lever. AutoCast, EMP and AVR come from one line of work (Govindan; Mao), so a single group holds the systems ancestry."),
  ...lit("Table 4. Mobile systems.", [
    ["AutoCast (Qiu, Huang, Asavisanu, Liu, Psounis, Govindan)","MobiSys 2022 [F]","per-object, per-receiver visibility from the current pose and relevance from trajectory collision; greedy by reward per byte; LTE-Direct TDMA; DSRC replay. The closest prior work"],
    ["EMP (Zhang, Zhang, Sun, Zhu, Guo, Qian, Mao)","MobiCom 2021 [S]","edge-assisted multi-vehicle perception; bandwidth-adaptive point-cloud partition"],
    ["AVR (Qiu, Ahmad, Bai, Gruteser, Govindan)","MobiSys 2018 [S]","augmented vehicular reality; vehicle-to-vehicle point clouds"],
    ["Zhang et al., real-time multi-vehicle collaboration on asynchronous sensors","MobiCom 2023 [S]","asynchrony handled in real time"],
    ["Wang, Cao, edge-assisted relevance-aware dissemination","IEEE 2024 [S]","send only objects whose trajectories intersect the receiver's"],
    ["Lusvarghi et al., The Search for Relevance","TVT 2026 [V]","relevance theory for V2X content"],
    ["V2X-ReaLO (Xiang et al.)","arXiv 2503.10034, 2025 [S]","online fusion over a live link"],
    ["CooperScene","arXiv 2606.31219, 2026 [S]","per-scene C-V2X traces outdoors"],
    ["SCooP-iT (this group)","in preparation","every link logged on the sensor clock and in the map frame, indoors, with replay validated; the only entry that supplies axis 4"],
  ]),

  H3("2.5 Wireless access and resource allocation: who gets the channel"),
  P("Owns user scheduling, slots, OFDMA, decentralised congestion control, learned access, and offloading. Assumes away what messages mean and when they are needed."),
  ...lit("Table 5. Wireless access and resource allocation.", [
    ["Bianchi, performance analysis of the 802.11 DCF","JSAC 2000","the CSMA model"],
    ["Heusse, Rousseau, Berger-Sabbatel, Duda, performance anomaly","INFOCOM 2003","slow senders occupy the airtime"],
    ["IEEE 802.11ax, 802.11be","standards","uplink OFDMA, trigger frames, buffer status reports, restricted target wake time"],
    ["ETSI TS 103 324, TR 103 562","standards","CPM generation and redundancy rules; decentralised congestion control"],
    ["Lyu et al., value-of-information prioritisation for CPM","VNC 2025 [S]","value of information in the standard"],
    ["SchedCP","arXiv 2502.10456, 2025 [S]","deep RL user scheduling by CSI and semantics"],
    ["PRISM (Mazandarani, Shokrnezhad, Taleb)","TNSM 2026 [F]","distributed multi-agent RL medium access over shared semantic segments"],
    ["Yahya, Naeem, Alsisi, Butt, Ejaz, robust task offloading in DT-enabled UAV-IoT networks","IEEE OJ-COMS vol. 7, 2026, DOI 10.1109/OJCOMS.2026.3663770 [F]","MINLP offloading with box-uncertain positions; content-free tasks; orthogonal channels"],
    ["Hady et al., multi-agent RL for resource allocation: a survey","AI Review 2025 [S]","the MARL line"],
    ["Neely, Stochastic Network Optimization","2010","drift-plus-penalty"],
    ["Kelly, Maulloo, Tan","JORS 1998","network utility maximisation"],
  ]),

  H3("2.6 Communication-aware robotics: where to move for the link"),
  P("Owns connectivity, throughput and radio maps as inputs to motion. Assumes away what is sent, the validity of content, and perception tasks."),
  ...lit("Table 6. Communication-aware robotics.", [
    ["Muralidharan, Mostofi, Communication-aware robotics","Annu. Rev. Control Robot. Auton. Syst. 2021 [V]","the survey"],
    ["Fink, Ribeiro, Kumar","Proc. IEEE 2012","robust control for mobility and wireless communication"],
    ["Stephan, Fink, Kumar, Ribeiro","T-RO 2017","concurrent control of mobility and communication"],
    ["Mostofi, Malmirchegini, Ghaffarkhah","ICRA 2010","channel estimation in robotic networks"],
    ["FERMI (Luo et al.)","RSS 2025 [V]","multi-robot radio mapping"],
    ["Amigoni, Banfi, Basilico","IEEE Intell. Syst. 2017","communication-restricted exploration survey"],
    ["Gielis, Shankar, Prorok","Current Robotics Reports 2022","critical review of communications in multi-robot systems"],
    ["Yang et al., online learning and connectivity maintenance","arXiv 2410.05798, 2024 [S]","connectivity-aware coordination"],
  ]),

  H3("2.7 Multi-robot mapping and estimation: sharing a map"),
  P("Owns novelty against a shared map, map increments, event triggering, and localisation against a prebuilt map. Assumes away deadlines, the channel, and dynamic objects."),
  ...lit("Table 7. Multi-robot mapping and estimation.", [
    ["Rocha, Dias, Carvalho","ICRA 2005","entropy-utility measurement sharing"],
    ["Kimera-Multi (Tian et al.)","T-RO 2022","bandwidth-parsimonious distributed SLAM"],
    ["Swarm-SLAM (Lajoie, Beltrame)","RA-L 2024","sparse inter-robot sharing"],
    ["Commerge","arXiv 2606.25386, 2026 [S]","scan-subset exchange for merging"],
    ["Trimpe, D'Andrea, event-based state estimation","TAC 2014","send on innovation; silence is information"],
    ["Khronos (Schmid, Abate, Chang, Carlone)","RSS 2024","spatio-temporal metric-semantic SLAM with change"],
    ["2Fast-2Lamaa (Le Gentil, Falque, Lisus, Barfoot)","IJRR 2026, arXiv 2410.05433 [F]","lidar-inertial localisation against a prebuilt map at under 16 % of a laptop CPU; free-space carving of dynamic objects; the cost of a preloaded static map on a robot"],
  ]),

  H3("2.8 Networked control and co-design"),
  P("Owns computation, channel and control under age of information. Assumes away perception content, real contended radios, and heterogeneous teams."),
  ...lit("Table 8. Networked control and co-design.", [
    ["Tripathi et al., co-design for monitoring and control","WiOpt 2021 [S]","Whittle co-scheduling of processing and channel"],
    ["Hahn, Stursberg, uncertain AoI in stochastic control","arXiv 2021 [S]","AoI as a control input"],
    ["AoI-aware task-oriented scheduling for networked control on SDR","arXiv 2022 [S]","argues that AoI needs semantic measures"],
    ["Park, Sahai et al., wireless network design for control systems","COMST 2018","survey"],
  ]),

  H3("2.9 Caching and edge systems"),
  P("Owns time-to-live, invalidation, prefetch and compute reuse. Assumes away validity from the environment and access patterns known from plans."),
  ...lit("Table 9. Caching and edge systems.", [
    ["Belady","IBM Syst. J. 1966","optimal offline eviction"],
    ["Fofack, Nain, Neglia, Towsley, TTL caches","venue to be checked [S]","TTL as a cache model"],
    ["CrossRoI (Guo et al.)","MMSys 2021 [S]","cross-camera region-of-interest masks to cut compute and traffic"],
    ["Spatula (Jain et al.)","SEC 2020 [S]","cross-camera correlations; 8–86 times less compute"],
    ["Mobility-aware proactive caching at roadside units","surveys 2024–2025 [S]","prefetch by predicted trajectory"],
  ]),

  H("3. Where the Gaps Are, Read off the Categories"),
  ...table("Table 10. The four axes of the scenario against the nearest prior work.", ["Axis", "Nearest prior", "What is missing"], [
    ["deadline from motion","AutoCast's time-to-collision relevance; AoI's age","a per-place deadline from the receiver's stopping distance"],
    ["validity from the world","AoII (assumed source model); ETSI dynamics rules (per object)","a per-place change rate measured from revisits, used as the lifetime"],
    ["need from the receiver's future","AutoCast's current visibility; Update the Unseen Only's statistical persistence","the receiver's own future view cast from its plan and the map"],
    ["shared, short channel with content on it","mobile-systems rigs (no content model); medium-access work (no meaning); collaborative perception (no channel)","a measured contended access point where the content decision changes what arrives"],
  ], [2300, 3400, 3800]),
  P("Two consequences no category reports: a clairvoyant bound computed from the known paths and the logged link, and the fraction of the field's traffic that its receivers never needed."),

  H("4. Positioning"),
  P("The computer-vision community fixes the channel, the communication-theory community fixes the content, the age-of-information community fixes the meaning, the mobile-systems community fixes the time, and the robotics community fixes the message; each obtains a clean problem by freezing what its neighbour studies. The scenario above leaves all four axes open at once, and the one quantity that ties them, AoII at the decision, is computable from state the agents already exchange. AutoCast is the paper to position against first; WiSwarm and Tripathi's co-design are the scheduling ancestors; EMP is the systems ancestor; Mostofi and Fink–Kumar are the motion ancestors; Trimpe's event triggering and ETSI's generation rules are the ancestors of sending on change."),
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
Packer.toBuffer(doc).then(b => { fs.writeFileSync('/home/user/Others/value_scheduling/docs/Scenario_and_Literature.docx', b); console.log('ok'); });
