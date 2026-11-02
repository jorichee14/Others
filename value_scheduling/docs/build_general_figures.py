"""General architecture and pipeline with the Gaussian SLAM map, object retrieval,
asset layer, RGB-only site cameras and the comms tiers.

  fig_architecture_general.html
  fig_pipeline_general.html

Render with: node render_general.js
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from figlib import Fig, legend, TEAL, RED  # noqa: E402

K = dict(ts=16, fs=13.5)          # compact card text
KT = dict(border=TEAL, dash=True)  # comms components

# =====================================================================
# Architecture
# =====================================================================
A = Fig(2000, 1500, 'Coarse-to-Fine Pick: Architecture',
        'mapping, retrieval and pick on robot and edge server, and what crosses the link')
DIV = 790
A.text(60, 185, 'ROBOT · mobile manipulator (and mapping robots) · CPU + GPU · ROS 2', 'colh')
A.text(820, 185, 'EDGE SERVER + SITE INFRASTRUCTURE · GPU server(s)', 'colh')

# ---------- robot ----------
A.box(60, 220, 700, 150, 'hw', 'Sensors + actuators')
for i, (t, l) in enumerate([('Head RGB-D', 'e.g. D435 · align'),
                            ('Wrist RGB-D', 'e.g. D405 · grasp'),
                            ('LiDAR + IMU', 'mapping robots / 3-D LiDAR')]):
    A.card(84 + i * 226, 262, 210, 52, t, [l], **K)
A.card(84, 322, 436, 36, '', ['wheel odometry · mobile base · arm · wrist · gripper'], fs=13.5)
A.card(530, 322, 210, 36, '', ['compute: CPU + GPU'], fs=13.5)

A.box(60, 390, 700, 112, 'loc', 'Localization + odometry', 'on the robot')
A.card(84, 430, 330, 60, 'LiDAR-inertial odometry', ['GLIM / FAST-LIO2 · trajectory'], **K)
A.card(430, 430, 310, 60, 'Map-based localization', ['AMCL on the 2-D grid · map frame'], **K)

A.box(60, 522, 700, 100, 'task', 'Task executive', 'stage machine')
A.card(84, 562, 656, 46, '', ['coarse → mid → fine → verify · publishes the stage · '
                              'triggers captures · retry or go'], fs=14.5)

A.box(60, 642, 700, 150, 'per', 'Object fusion', 'on the robot · RGB + depth in')
A.card(84, 682, 210, 96, 'Mask + depth', ['SAM 3 mask from server', 'on the robot’s own depth',
                                           '→ object cloud'], **K)
A.card(308, 682, 210, 96, 'Align', ['object pose in base', 'head view → arm line'], **K)
A.card(532, 682, 208, 96, 'Grasp perception', ['wrist depth, every frame', 'measured surfaces only',
                                                'fine stage, local'], **K)

A.box(60, 812, 700, 112, 'man', 'Motion', 'on the robot')
A.card(84, 852, 210, 58, 'Nav2', ['base · coarse'], **K)
A.card(308, 852, 210, 58, 'Base + arm', ['approach and align · mid'], **K)
A.card(532, 852, 208, 58, 'Arm + wrist + gripper', ['servo · contact by effort'], **K)

A.box(60, 944, 700, 330, 'com', 'Comms bridge · robot', 'ROS 2 nodes')
A.card(84, 984, 210, 124, 'offload_client', ['request tier', 'crop + JPEG on trigger',
                                              'ID · capture time · pose',
                                              'timeout · retry · drop stale'], **KT, **K)
A.card(308, 984, 210, 124, 'keyframe_packer', ['background tier', 'scan + JPEG + pose, ~2 Hz',
                                                'on change · Draco · cap',
                                                'paused in mid / fine'], **KT, **K)
A.card(532, 984, 208, 124, 'recorder', ['offline tier', 'full-rate frames, IMU, bags',
                                         'to disk', '→ upload at the dock'], **KT, **K)
A.card(84, 1122, 320, 136, 'link_monitor', ['heartbeat: RTT, server queue time',
                                             'RSSI + link map', '→ expected answer time'],
       **KT, **K)
A.card(420, 1122, 320, 136, 'policy', ['send requests when stopped',
                                        'wait / retry / onboard fallback',
                                        'pause keyframe packs in mid + fine',
                                        'speed limit → Nav2'], **KT, **K)

# ---------- server ----------
A.box(820, 220, 1120, 190, 'twin', 'Gaussian SLAM map', 'offline · ~16 h per run (RTX 3080)')
gs = [('Poses + geometry', ['GLIM / FAST-LIO2', 'trajectory + LiDAR map']),
      ('Scene build', ['undistorted frames', 'map depth · blur · surfels']),
      ('Training', ['NeST surfels on LiDAR', 'colour · blur · normals']),
      ('Outputs', ['held-out eval, flythrough', 'LOD + Sionna export']),
      ('Link map', ['seeded from Sionna', 'corrected by survey'])]
for i, (t, l) in enumerate(gs):
    x = 842 + i * 218
    A.card(x, 262, 204, 80, t, l, **(KT if t == 'Link map' else {}), **K)
    if i < 3:
        A.line([(x + 204, 302), (x + 216, 302)])
A.line([(842 + 3 * 218 + 204, 302), (842 + 4 * 218, 302)], color=TEAL)
A.text(842, 356, 'next: blur and exposure from odometry · merge sessions and robots into one map',
       'lbl', style='font-style:italic;color:#666')

A.box(820, 440, 360, 190, 'hw', 'Site cameras', 'wired · RGB only')
A.card(842, 482, 316, 132, 'Fixed RGB cameras', ['calibrated in the map frame',
                                                  'depth: rendered once per camera',
                                                  'from the map + triangulation',
                                                  'live object + person detections'], **K)
A.box(1200, 440, 360, 190, 'task', 'Task')
A.card(1222, 482, 316, 80, 'Request → query', ['“give me the Pringles”', '→ class to look for'],
       **K)
A.box(1580, 440, 360, 190, 'twin', 'Asset layer', 'background · ≥32 GB GPU')
A.card(1602, 482, 316, 132, 'SAM 3D asset per object', ['textured, metric scale',
                                                         'from robot keyframes with depth',
                                                         'each surface checked vs cloud:',
                                                         'measured / generated'], **K)

A.box(820, 650, 1120, 190, 'per', 'Object retrieval', 'online · coarse: where is it?')
orr = [('Keyframe search', ['Perception Encoder', 'map keyframes + site cams']),
       ('SAM 3 masks', ['text prompt → masks']),
       ('Object cloud', ['robot depth / LiDAR', 'site cams: map depth']),
       ('Object cache', ['pose, class, ID, age', 'CRISP geometry']),
       ('Target + view pose', ['with age · refreshed', 'by site cams if old'])]
for i, (t, l) in enumerate(orr):
    x = 842 + i * 218
    A.card(x, 692, 204, 92, t, l, dash=True, **K)
    if i < 4:
        A.line([(x + 204, 738), (x + 216, 738)])

A.box(820, 860, 1120, 160, 'com', 'Comms bridge · server', 'ROS 2 nodes')
srv = [('offload_server', ['crop → SAM 3 · answer', 'by request ID · drop', 'stale requests']),
       ('keyframe_ingest', ['receive packs, backfill', '→ SLAM correction,', 'keyframe search']),
       ('GPU scheduler', ['online first: SAM 3,', 'CRISP · then SAM 3D,', 'training']),
       ('logger', ['per request + pack:', 'times, size, stage,', 'RSSI, RTT → link map'])]
for i, (t, l) in enumerate(srv):
    A.card(842 + i * 272, 900, 258, 104, t, l, **KT, **K)

A.box(820, 1040, 1120, 258, 'hw', 'Link robot ↔ server', 'Wi-Fi · ROS 2 · DSCP per tier')
tiers = [('Request', 'high', ['↓ target + view pose + age', '↑ JPEG crop · ↓ SAM 3 mask',
                              '↑ robot pose · ↕ heartbeat']),
         ('Background', 'low', ['↑ keyframe packs ~2 Hz', 'on change · rate-capped',
                                'paused in mid + fine']),
         ('Offline', 'lowest', ['↑ bags, full-rate frames,', 'IMU, trajectory', 'at the dock / overnight']),
         ('Once', '—', ['↑ calibration', '↓ 2-D grid per map version', ''])]
for i, (t, pr, l) in enumerate(tiers):
    x = 842 + i * 272
    A.card(x, 1082, 258, 120, f'{t} <span style="font-weight:500;color:#666">· {pr}</span>', l,
           **K, border=TEAL if i < 3 else '#bbb')
A.text(842, 1216, '✕ never sent: raw image streams, depth images, raw LiDAR scans during a task',
       'lv', style='font-weight:600')
A.text(842, 1244, 'requests best effort keep-last-1 · answers reliable · '
       'DDS tuned for large messages', 'lv', style=f'color:{TEAL};font-weight:600')

legend(A, 820, 1322, [('hw', 'hardware, link'), ('loc', 'localization'), ('task', 'task'),
                      ('per', 'perception'), ('man', 'motion'), ('twin', 'map, assets'),
                      ('com', 'comms')])
A.svg.append(f'<line x1="820" y1="1376" x2="850" y2="1376" stroke="{RED}" stroke-width="3" '
             f'stroke-dasharray="9,6"/>')
A.text(860, 1365, 'network boundary', 'lg')
A.out.append('<div class="sw" style="left:1060px;top:1365px;background:#fff;'
             'border:2px dashed #999"></div>')
A.text(1100, 1365, 'dashed: not built yet', 'lg')

# robot arrows
A.line([(470, 370), (470, 428)])
A.line([(410, 502), (410, 520)])
A.line([(410, 622), (410, 640)])
A.line([(410, 792), (410, 810)])
A.line([(150, 942), (150, 926)])
A.line([(560, 370), (560, 380), (775, 380), (775, 730), (742, 730)])
# bridges <-> link
A.line([(1380, 1020), (1380, 1038)], color=TEAL, both=True)
A.line([(820, 1150), (765, 1150), (765, 1190), (742, 1190)], color=TEAL, both=True)
A.svg.append(f'<line x1="{DIV}" y1="175" x2="{DIV}" y2="1460" stroke="{RED}" stroke-width="3" '
             f'stroke-dasharray="14,9"/>')

# =====================================================================
# Pipeline
# =====================================================================
P = Fig(2000, 1500, 'Coarse-to-Fine Pick: Pipeline',
        'what moves, how it is perceived, and what crosses the link at each stage')
X0, CW, G = 250, 330, 14
cols = [('0 · Task', 'request', '#efeafb'), ('1 · Coarse', 'find + drive · room', '#e4eefb'),
        ('2 · Mid', 'locate + align · object', '#fdf3e4'), ('3 · Fine', 'grasp · gripper', '#fde3d6'),
        ('4 · Verify', 'lift + check', '#eef0f3')]

# before any pick band
P.rect(40, 178, 1916, 104, '#eef7f1', rx=10, stroke='#9fd3ae')
P.text(60, 188, 'Before any pick', 'lane')
P.text(60, 214, 'offline / background', 'lanes')
pre = [('Mapping run', 'LiDAR-inertial poses · bags → dock'),
       ('Gaussian SLAM map', 'scene build + training, ~16 h (3080)'),
       ('Site-camera depth', 'rendered once per camera from the map'),
       ('Assets', 'SAM 3D per object · measured / generated'),
       ('Link map', 'Sionna export from the map + survey')]
for i, (t, s) in enumerate(pre):
    x = 300 + i * 332
    P.card(x, 192, 318, 76, t, [s], dash=True, ts=16, fs=13.5)

Y = 300
for i, (t, s, bg) in enumerate(cols):
    x = X0 + i * (CW + G)
    P.rect(x, Y, CW, 66, bg, rx=10)
    P.text(x, Y + 6, t, 'col', w=CW)
    P.text(x, Y + 38, s, 'cols', w=CW)
    if i < 4:
        P.line([(x + CW + 1, Y + 33), (x + CW + G - 1, Y + 33)])

# movement hierarchy
P.text(40, 390, 'Movement', 'lane')
P.text(40, 416, 'hierarchy', 'lane')
P.text(40, 444, 'scope shrinks,', 'lanes')
P.text(40, 462, 'precision rises', 'lanes')
P.rect(X0, 382, CW, 120, '#f6f3fd', rx=8)
P.text(X0, 418, 'no motion', 'cols', w=CW)
P.text(X0, 440, 'server parses the request', 'cols', w=CW)
steps = [(1, 'Room', 'base · Nav2', '~10 cm · 0.3–0.5 m/s', 382, 120),
         (2, 'Object', 'base + arm, head RGB-D', '~1–2 cm · stop to look', 402, 100),
         (3, 'Gripper', 'arm + wrist + gripper, wrist RGB-D', '~5 mm · servo, contact', 422, 80),
         (4, 'Gripper', 'lift + retract', 'check, then return', 402, 100)]
for i, title, what, prec, y, h in steps:
    x = X0 + i * (CW + G)
    P.rect(x, y, CW, h, cols[i][2], rx=8, stroke='#ccc')
    P.text(x, y + 8, f'<b>{title}-level</b>', 'cols', w=CW, style='font-size:18px;color:#222')
    P.text(x, y + 33, what, 'cols', w=CW)
    P.text(x, y + 54, prec, 'cols', w=CW, style='font-weight:600;color:#333')

lanes = [
    ('Perception', 'what is seen, by which model', 520, 112, False, [
        ['text → class to look for', 'keyframe search', '(Perception Encoder)'],
        ['site cams + map keyframes', '→ SAM 3 → object cloud', '(map depth / robot depth)',
         '→ object cache → target'],
        ['head RGB-D crop → SAM 3 mask', 'mask × robot depth', '→ object cloud → align'],
        ['wrist RGB-D, every frame', 'grasp on measured', 'surfaces only'],
        ['gripper aperture + effort', 'wrist view check']]),
    ('Runs on', 'where the loop closes', 650, 46, False, [
        ['server'], ['server finds, robot drives'], ['server segments, robot fuses'],
        ['robot only'], ['robot (server optional)']]),
    ('Request tier', 'task traffic, high priority', 714, 92, True, [
        ['↑ request text', '↓ class'],
        ['↓ target + view pose + age', '(refreshed by site cams', 'if old) · ↑ pose 10 Hz'],
        ['robot stops · ↑ 1 JPEG crop', '(~50 KB) · ↓ 1 mask'],
        ['nothing'],
        ['↑ 1 crop, optional', '↓ ok / retry']]),
    ('Background tier', 'keyframe packs, low priority', 824, 66, True, [
        ['~2 Hz on change'], ['reduced rate'], ['paused · buffered on robot'],
        ['paused'], ['resume · backfill']]),
    ('Server GPU', 'online before background', 908, 66, True, [
        ['keyframe search (online)'], ['SAM 3 on site cams (online)', 'SAM 3D / training: background'],
        ['SAM 3 mask first'], ['background jobs run'], ['—']]),
    ('Answer due', 'timeout from the link', 992, 46, True, [
        ['seconds'], ['seconds (robot drives)'], ['~0.5–1 s (robot stopped)'],
        ['local loop ~30 Hz'], ['~1 s']]),
    ('If late', 'fallback', 1056, 66, True, [
        ['retry'], ['keep last valid target', 'slow down'], ['retry once', 'then onboard detector'],
        ['— (local)'], ['local check only']]),
]
for name, sub, y, h, com, cells in lanes:
    P.text(40, y + 6, name, 'lane')
    P.text(40, y + 32, sub, 'lanes', w=200)
    for i, lines in enumerate(cells):
        x = X0 + i * (CW + G)
        P.card(x, y, CW, h, '', lines, border=TEAL if com else '#c8c8c8', dash=com,
               fs=14.5, bg='#f2fbfa' if com else '#fff')
    P.svg.append(f'<line x1="30" y1="{y - 9}" x2="1960" y2="{y - 9}" stroke="#e3e3e3" stroke-width="1"/>')

y = 1140
P.text(40, y + 4, 'Link', 'lane')
P.text(40, y + 30, 'dependence', 'lanes', style='font-size:15px;font-weight:700;color:#333')
dep = [('high', '#0b7f74'), ('high: target from the server', '#2a9d8f'),
       ('medium: one request, then local', '#76c7bd'), ('none', '#e3f4f2'),
       ('low', '#cfece8')]
for i, (s, c) in enumerate(dep):
    x = X0 + i * (CW + G)
    P.rect(x, y, CW, 44, c, rx=6)
    P.text(x, y + 11, s, 'cols', w=CW,
           style=f'color:{"#fff" if i < 2 else "#123"};font-weight:700;font-size:16px')
P.text(40, 1206, 'Precision', 'lane')
P.text(40, 1232, 'needed', 'lanes', style='font-size:15px;font-weight:700;color:#333')
for i, s in enumerate(['—', '~10 cm', '~1–2 cm', '~5 mm', 'grasp held']):
    x = X0 + i * (CW + G)
    P.rect(x, 1202, CW, 44, ['#f6f3fd', '#dbe8fb', '#fbe6c4', '#f8cdb6', '#eef0f3'][i], rx=6)
    P.text(x, 1213, s, 'cols', w=CW, style='color:#222;font-weight:700;font-size:16px')

P.line([(X0 + 345, 1280), (X0 + 4 * (CW + G) + CW - 10, 1280)], color='#555', width=3)
P.text(X0 + 345, 1290, 'finer motion → smaller region → fewer, smaller messages → '
       'loop moves from the server to the robot', 'lbl', w=1300,
       style='font-size:17px;font-weight:600;color:#333')
P.text(40, 1340, 'Raw image streams, depth images and raw LiDAR scans never cross the link '
       'during a task; full-rate data goes at the dock. Every request carries an ID and capture '
       'time; every answer echoes them and carries its age.', 'lbl', w=1900,
       style='font-size:16px;color:#555;font-style:italic')

here = Path(__file__).parent
(here / 'fig_architecture_general.html').write_text(A.html('Pick Architecture'), encoding='utf-8')
(here / 'fig_pipeline_general.html').write_text(P.html('Pick Pipeline'), encoding='utf-8')
print('ok')
