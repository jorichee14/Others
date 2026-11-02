"""Coarse-to-fine pick architecture with the communication layer.

Builds stretch_comms_architecture.html; render with render_diagram.js.
Layout follows the group's original diagram; additions are in teal.
"""
from pathlib import Path

W, H = 2000, 1360
C = {
    'hw': ('#eef0f3', '#c3c7cf'),
    'loc': ('#e4eefb', '#a9c3ec'),
    'man': ('#fdf3e4', '#efcb96'),
    'twin': ('#e4f4e9', '#9fd3ae'),
    'task': ('#efeafb', '#c3b2ec'),
    'com': ('#ddf3f0', '#5fbfb3'),
}
TEAL = '#0b7f74'
out = []


def box(x, y, w, h, kind, title, right=''):
    bg, bd = C[kind]
    out.append(
        f'<div class="box" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px;'
        f'background:{bg};border-color:{bd}"><div class="bt">{title}</div>'
        f'<div class="br">{right}</div></div>')


def card(x, y, w, h, title, lines, dashed=True, solid_border=False, new=False,
         comm_line=None, title_size=19):
    style = 'dashed' if dashed else 'solid'
    bd = TEAL if new else ('#c9c9c9' if dashed else '#b8bcc4')
    body = ''.join(f'<div class="cl">{l}</div>' for l in lines)
    if comm_line:
        body += f'<div class="cl comm">{comm_line}</div>'
    out.append(
        f'<div class="card" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px;'
        f'border:2px {style} {bd}"><div class="ct" style="font-size:{title_size}px">'
        f'{title}</div>{body}</div>')


def text(x, y, s, cls='lbl', w=None, align='left'):
    ws = f'width:{w}px;' if w else ''
    out.append(f'<div class="{cls}" style="left:{x}px;top:{y}px;{ws}text-align:{align}">{s}</div>')


svg = []


def line(pts, arrow=True, color='#222', dash=None, width=2.2):
    d = ' '.join(f'{x},{y}' for x, y in pts)
    da = f'stroke-dasharray="{dash}"' if dash else ''
    m = f'marker-end="url(#ah{"t" if color == TEAL else ""})"' if arrow else ''
    svg.append(f'<polyline points="{d}" fill="none" stroke="{color}" '
               f'stroke-width="{width}" {da} {m}/>')


# ---------------- header ----------------
out.append('<div class="hdr">Mobile Manipulation: Coarse-to-Fine Pick</div>')
out.append('<div class="sub">system architecture with the communication layer</div>')
out.append('<div class="stripe"></div>')

# ---------------- left: robot ----------------
text(62, 212, 'ROBOT · Hello Robot Stretch · onboard computer · ROS 2', 'colh')
box(62, 245, 688, 143, 'hw', 'Sensors + robot')
for i, (a, b) in enumerate([('2D LiDAR (base)', 'Diff-drive base'),
                            ('Head RGB-D · D435', 'Lift + telescoping arm'),
                            ('Gripper RGB-D · D405', '3-DoF wrist + gripper')]):
    x = 88 + i * 218
    card(x, 290, 200, 36, a, [], dashed=False, title_size=17)
    card(x, 337, 200, 36, b, [], dashed=False, title_size=17)

box(62, 410, 688, 125, 'loc', 'Localization', '2D grid ←')
card(88, 452, 312, 68, 'AMCL', ['2D LiDAR · map-frame pose · 10 Hz'], dashed=False)
card(430, 452, 296, 68, 'Map frame = twin frame', ['grid from the ED305 reference map'])

box(62, 560, 688, 515, 'man', 'Coarse-to-fine pick', 'precision target')
stages = [
    ('1 · Coarse', 'room · ~10 cm', 'Global navigation',
     ['Nav2 to the view pose from the server', 'AMCL pose · out: target in the head view'],
     'link: ↓ target + view pose', False),
    ('2 · Mid', 'object · ~2 cm', 'Look + align',
     ['head D435: SAM 3 mask → object cloud in base',
      'rotate–translate–rotate: object on arm line'],
     'link: ↑ JPEG crop · ↓ mask · depth stays on the robot', True),
    ('3 · Fine', 'grasp · ~5 mm', 'Grasp',
     ['gripper D405 at 10–30 cm: antipodal grasp', 'servo lift, arm, wrist · contact by effort'],
     'link: none', True),
    ('4 · Verify', 'retry or go', 'Lift, retract, verify',
     ['grasp check: gripper aperture + D405 view', 'fail → back to 2 · ok → return, hand over'],
     'link: ↑ D405 crop (optional)', True),
]
row_y = [602, 718, 834, 950]
for (lab, l2, title, lines, comm, dashed), y in zip(stages, row_y):
    out.append(
        f'<div class="card stage" style="left:88px;top:{y}px;width:134px;height:104px;'
        f'padding-top:24px"><div class="ct">{lab}</div><div class="cl">{l2}</div></div>')
    card(238, y, 488, 104, title, lines, dashed=dashed, comm_line=comm)
for y0, y1 in [(706, 718), (822, 834), (938, 950)]:
    line([(482, y0), (482, y1)])

# comms bridge, robot side
box(62, 1110, 688, 195, 'com', 'Comms bridge · robot', 'ROS 2 nodes')
card(88, 1152, 200, 128, 'offload_client',
     ['capture on trigger', 'resize + JPEG', 'ID · capture time · pose', 'timeout · retry · drop stale'],
     new=True, title_size=18)
card(306, 1152, 200, 128, 'link_monitor',
     ['heartbeat: RTT, queue', 'RSSI (iw) + link map', '→ expected answer time'],
     new=True, title_size=18)
card(524, 1152, 202, 128, 'policy',
     ['send when stopped', 'wait or onboard fallback', 'speed limit → Nav2'],
     new=True, title_size=18)
line([(306, 1216), (290, 1216)], color=TEAL)
line([(506, 1216), (522, 1216)], color=TEAL)
line([(300, 1077), (300, 1108)], color=TEAL)
text(308, 1084, 'stage · capture trigger', 'lbl teal')
line([(625, 1108), (625, 1077)], color=TEAL)
text(470, 1084, 'results', 'lbl teal', w=148, align='right')
text(633, 1084, 'speed limit', 'lbl teal')

# ---------------- right: server ----------------
text(800, 212, 'ED305 SERVER + INFRASTRUCTURE · RTX workstation · edge PC: '
     'AMD Ryzen AI Max+ 395 (requested)', 'colh')
box(800, 245, 352, 210, 'twin', 'ED305 twin', 'offline')
card(820, 290, 312, 82, 'Reference map',
     ['OS1 + GLIM, anchored', 'keyframes, objects · 2D grid export'], dashed=False)
card(820, 382, 312, 60, 'Link map',
     ['RSSI · RTT · goodput per cell, map frame'], new=True, title_size=18)

box(1176, 245, 352, 210, 'hw', 'Camera network', '360° coverage')
card(1196, 290, 312, 100, 'Infrastructure cameras',
     ['calibrated in the map frame', 'live object + person detections'])
text(1196, 405, 'wired to the server', 'lbl', w=312)

box(1552, 245, 388, 210, 'task', 'Task')
card(1572, 290, 348, 100, 'Request → object query',
     ['“give me the Pringles”', '→ class, instance, where to deliver'])

box(800, 480, 1140, 222, 'man', 'Object retrieval', 'coarse: where is it?')
oc = [(825, 'Target + view pose', ['object pose in map', '→ base pose that sees it']),
      (1100, 'Object cache', ['cloud per keyframe, class, ID', 'fused across views']),
      (1375, 'Segment + depth', ['SAM 3 mask', 'metric RGB-D depth']),
      (1650, 'Keyframe search', ['Perception Encoder embeddings',
                                 'twin keyframes + live cameras'])]
for x, t, l in oc:
    card(x, 530, 250, 98, t, l)
for xa, xb in [(1650, 1627), (1375, 1352), (1100, 1077)]:
    line([(xa, 578), (xb, 578)])
line([(1118, 455), (1118, 528)])
text(1128, 462, 'keyframes, objects', 'lbl')
line([(1437, 455), (1437, 528)])
text(1447, 462, 'images, detections', 'lbl')
line([(1700, 455), (1700, 528)])
text(1710, 462, 'object query', 'lbl')

# comms bridge, server side
box(800, 728, 1140, 175, 'com', 'Comms bridge · server', 'ROS 2 nodes')
card(1376, 773, 540, 112, 'offload_server',
     ['decode JPEG → model input topic (SAM 3, YOLO, keypoints)',
      'match result to request ID · add server timestamps',
      'drop requests past their deadline · report queue length'],
     new=True, title_size=18)
card(822, 773, 530, 112, 'logger',
     ['per request: capture, send, receive, answer times',
      'size, stage, link state (RSSI, RTT)',
      '→ link map of ED305'],
     new=True, title_size=18)
line([(1500, 773), (1500, 630)], color=TEAL)
text(1508, 700, 'crop → image', 'lbl teal')
line([(1460, 630), (1460, 771)], color=TEAL)
text(1340, 700, 'mask', 'lbl teal', w=112, align='right')

box(800, 930, 1140, 210, 'hw', 'Link robot ↔ server', 'Wi-Fi 5 GHz · ROS 2 topics')
rows_l = [('↓ 2D grid', 'map frame · per map version'),
          ('↓ Target + view pose', 'per request · ID + capture time'),
          ('↓ Object mask', 'per crop · small'),
          ('↑ Map-frame pose', '10 Hz · ~1 KB/s')]
rows_r = [('↑ RGB crop (JPEG)', 'on request · depth stays on robot'),
          ('↑ D405 crop', 'optional, verify'),
          ('↕ Heartbeat', '5–10 Hz · RTT, server queue')]
for i, (a_, b_) in enumerate(rows_l):
    text(822, 975 + i * 28, a_, 'lk')
    text(1025, 977 + i * 28, b_, 'lv')
for i, (a_, b_) in enumerate(rows_r):
    text(1390, 975 + i * 28, a_, 'lk')
    text(1590, 977 + i * 28, b_, 'lv')
text(822, 1100, 'QoS: images best effort, keep last 1 · answers reliable · DSCP priority '
     'for requests · DDS tuned for large messages · no raw image streams', 'lv teal', w=1100)

# legend
lg = [('hw', 'hardware, infra, link'), ('loc', 'localization'), ('man', 'manipulation, perception'),
      ('twin', 'offline twin'), ('task', 'task'), ('com', 'comms bridge')]
LX = [800, 1010, 1185, 1450, 1615, 1745]
for i, (k, s) in enumerate(lg):
    bg, bd = C[k]
    x = LX[i]
    out.append(f'<div class="sw" style="left:{x}px;top:1172px;background:{bg};border-color:{bd}"></div>')
    text(x + 40, 1170, s, 'lg')
svg.append('<line x1="800" y1="1222" x2="830" y2="1222" stroke="#d6262c" '
           'stroke-width="3" stroke-dasharray="9,6"/>')
text(840, 1211, 'network: robot ↔ server', 'lg')
out.append('<div class="sw" style="left:1090px;top:1211px;background:#fff;'
           'border:2px dashed #aaa"></div>')
text(1130, 1211, 'not built yet', 'lg')
text(1300, 1211, 'solid: built in the cabinet-door demo (Nav2, AMCL)', 'lg')

# ---------------- cross arrows ----------------
line([(820, 340), (765, 340), (765, 487), (729, 487)])
line([(375, 522), (375, 600)])
text(386, 538, 'map-frame pose', 'lbl')
line([(825, 600), (785, 600), (785, 652), (729, 652)])
text(806, 710, 'target + view pose', 'lbl')
line([(262, 388), (262, 450)])
line([(726, 1000), (742, 1000), (742, 760), (729, 760)])
# robot bridge <-> link <-> server bridge
line([(800, 1035), (762, 1035), (762, 1195), (752, 1195)], color=TEAL)

# divider
svg.append(f'<line x1="775" y1="200" x2="775" y2="{H - 40}" stroke="#d6262c" '
           f'stroke-width="3" stroke-dasharray="14,9"/>')
text(742, H - 34, 'network', 'netl')

css = f"""
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ width:{W}px; height:{H}px; position:relative; background:#fff;
        font-family:'Inter', 'DejaVu Sans', sans-serif; color:#1d1d1f }}
.hdr {{ position:absolute; left:0; top:0; width:{W}px; height:150px; background:#0a8fd8;
        color:#fff; font-size:64px; font-weight:800; text-align:center; padding-top:30px }}
.sub {{ position:absolute; left:0; top:112px; width:{W}px; text-align:center; color:#d9f0ff;
        font-size:22px; font-weight:500 }}
.stripe {{ position:absolute; left:0; top:150px; width:{W}px; height:24px; background:#bfe6f7 }}
.colh {{ position:absolute; font-size:19px; font-weight:700; color:#333 }}
.box {{ position:absolute; border:2px solid; border-radius:10px; box-shadow:7px 7px 0 #e9e9ee }}
.bt {{ position:absolute; left:18px; top:10px; font-size:24px; font-weight:700 }}
.br {{ position:absolute; right:16px; top:15px; font-size:16px; color:#555 }}
.card {{ position:absolute; background:#fff; border-radius:6px; text-align:center;
         padding-top:7px }}
.stage {{ border:2px solid #ddd }}
.ct {{ font-size:19px; font-weight:700; margin-bottom:3px }}
.cl {{ font-size:15px; color:#444; line-height:20px }}
.comm {{ color:{TEAL}; font-weight:600; font-size:14.5px; margin-top:3px }}
.lbl {{ position:absolute; font-size:15px; color:#333 }}
.teal {{ color:{TEAL} !important; font-weight:600 }}
.note {{ position:absolute; font-size:15.5px; font-style:italic; color:#666 }}
.lk {{ position:absolute; font-size:16px; font-weight:700 }}
.lv {{ position:absolute; font-size:15px; color:#444 }}
.lg {{ position:absolute; font-size:16px; color:#333 }}
.sw {{ position:absolute; width:30px; height:20px; border:2px solid; border-radius:4px }}
.netl {{ position:absolute; font-size:17px; font-weight:700; color:#d6262c }}
.new {{ display:inline-block; font-size:12px; font-weight:700; color:#fff; background:{TEAL};
        border-radius:4px; padding:1px 6px; vertical-align:middle; margin-left:4px }}
svg {{ position:absolute; left:0; top:0 }}
"""
markers = (
    '<defs><marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="5" '
    'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#222"/></marker>'
    f'<marker id="aht" markerWidth="10" markerHeight="10" refX="8" refY="5" '
    f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{TEAL}"/></marker></defs>')
html = (f'<!doctype html><html><head><meta charset="utf-8"><title>Coarse-to-Fine Pick</title>'
        f'<style>{css}</style></head><body>{"".join(out)}'
        f'<svg width="{W}" height="{H}">{markers}{"".join(svg)}</svg></body></html>')
p = Path(__file__).with_name('stretch_comms_architecture.html')
p.write_text(html, encoding='utf-8')
print(p)
