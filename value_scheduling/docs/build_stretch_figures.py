"""Two figures for the Stretch 4 coarse-to-fine pick.

  fig_architecture.html  components on robot and server, the comms bridge, the link
  fig_pipeline.html      stages coarse -> fine: movement hierarchy, perception, comms

Render with: node render_figures.js
"""
from pathlib import Path

C = {
    'hw': ('#eef0f3', '#c3c7cf'),
    'loc': ('#e4eefb', '#a9c3ec'),
    'man': ('#fdf3e4', '#efcb96'),
    'per': ('#fff7e0', '#e9c46a'),
    'twin': ('#e4f4e9', '#9fd3ae'),
    'task': ('#efeafb', '#c3b2ec'),
    'com': ('#ddf3f0', '#5fbfb3'),
}
TEAL = '#0b7f74'
RED = '#d6262c'


class Fig:
    def __init__(self, w, h, title, sub):
        self.w, self.h, self.out, self.svg, self.bg = w, h, [], [], []
        self.out.append(f'<div class="hdr">{title}</div><div class="sub">{sub}</div>'
                        '<div class="stripe"></div>')

    def box(self, x, y, w, h, kind, title, right=''):
        bg, bd = C[kind]
        self.out.append(
            f'<div class="box" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px;'
            f'background:{bg};border-color:{bd}"><div class="bt">{title}</div>'
            f'<div class="br">{right}</div></div>')

    def card(self, x, y, w, h, title, lines=(), border='#b8bcc4', dash=False, ts=18, fs=15,
             bg='#fff', align='center'):
        st = 'dashed' if dash else 'solid'
        body = ''.join(f'<div class="cl" style="font-size:{fs}px">{l}</div>' for l in lines)
        t = f'<div class="ct" style="font-size:{ts}px">{title}</div>' if title else ''
        self.out.append(
            f'<div class="card" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px;'
            f'border:2px {st} {border};background:{bg};text-align:{align}">{t}{body}</div>')

    def text(self, x, y, s, cls='lbl', w=None, align='left', style=''):
        if cls in ('col', 'cols') and align == 'left':
            align = 'center'
        ws = f'width:{w}px;' if w else ''
        self.out.append(f'<div class="{cls}" style="left:{x}px;top:{y}px;{ws}'
                        f'text-align:{align};{style}">{s}</div>')

    def line(self, pts, color='#222', arrow=True, dash=None, width=2.2, both=False):
        d = ' '.join(f'{x},{y}' for x, y in pts)
        da = f'stroke-dasharray="{dash}"' if dash else ''
        mid = {'#222': 'k', TEAL: 't', RED: 'r'}.get(color, 'k')
        m = f'marker-end="url(#a{mid})"' if arrow else ''
        ms = f'marker-start="url(#s{mid})"' if both else ''
        self.svg.append(f'<polyline points="{d}" fill="none" stroke="{color}" '
                        f'stroke-width="{width}" {da} {m} {ms}/>')

    def rect(self, x, y, w, h, fill, rx=8, stroke='none'):
        self.bg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                        f'fill="{fill}" stroke="{stroke}"/>')

    def html(self, title):
        W, H = self.w, self.h
        css = f"""
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ width:{W}px; height:{H}px; position:relative; background:#fff;
        font-family:'Inter','DejaVu Sans',sans-serif; color:#1d1d1f }}
.hdr {{ position:absolute; left:0; top:0; width:{W}px; height:140px; background:#0a8fd8;
        color:#fff; font-size:58px; font-weight:800; text-align:center; padding-top:22px }}
.sub {{ position:absolute; left:0; top:98px; width:{W}px; text-align:center; color:#d9f0ff;
        font-size:22px; font-weight:500 }}
.stripe {{ position:absolute; left:0; top:140px; width:{W}px; height:20px; background:#bfe6f7 }}
.colh {{ position:absolute; font-size:19px; font-weight:700; color:#333 }}
.box {{ position:absolute; border:2px solid; border-radius:10px; box-shadow:7px 7px 0 #e9e9ee }}
.bt {{ position:absolute; left:18px; top:9px; font-size:23px; font-weight:700 }}
.br {{ position:absolute; right:16px; top:14px; font-size:15px; color:#555 }}
.card {{ position:absolute; border-radius:6px; padding:6px 6px 0 6px }}
.ct {{ font-weight:700; margin-bottom:3px }}
.cl {{ color:#444; line-height:20px }}
.lbl {{ position:absolute; font-size:15px; color:#333 }}
.teal {{ color:{TEAL}; font-weight:600 }}
.lk {{ position:absolute; font-size:16px; font-weight:700 }}
.lv {{ position:absolute; font-size:15px; color:#444 }}
.lg {{ position:absolute; font-size:16px; color:#333 }}
.sw {{ position:absolute; width:30px; height:20px; border:2px solid; border-radius:4px }}
.netl {{ position:absolute; font-size:17px; font-weight:700; color:{RED} }}
.lane {{ position:absolute; font-size:19px; font-weight:800; color:#333 }}
.lanes {{ position:absolute; font-size:14px; color:#666 }}
.col {{ position:absolute; font-size:22px; font-weight:800; text-align:center }}
.cols {{ position:absolute; font-size:15px; color:#555; text-align:center }}
svg {{ position:absolute; left:0; top:0 }}
"""
        mk = ''
        for k, c in (('k', '#222'), ('t', TEAL), ('r', RED)):
            mk += (f'<marker id="a{k}" markerWidth="10" markerHeight="10" refX="8" refY="5" '
                   f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
                   f'<marker id="s{k}" markerWidth="10" markerHeight="10" refX="2" refY="5" '
                   f'orient="auto"><path d="M10,0 L0,5 L10,10 z" fill="{c}"/></marker>')
        return (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
                f'<style>{css}</style></head><body>'
                f'<svg width="{W}" height="{H}"><defs>{mk}</defs>{"".join(self.bg)}</svg>'
                f'{"".join(self.out)}'
                f'<svg width="{W}" height="{H}"><defs>{mk}</defs>{"".join(self.svg)}</svg>'
                f'</body></html>')


def legend(f, x, y, items):
    xx = x
    for k, s in items:
        bg, bd = C[k]
        f.out.append(f'<div class="sw" style="left:{xx}px;top:{y + 2}px;background:{bg};'
                     f'border-color:{bd}"></div>')
        f.text(xx + 40, y, s, 'lg')
        xx += 60 + 9 * len(s)


# =====================================================================
# Figure 1: architecture
# =====================================================================
A = Fig(2000, 1230, 'Stretch 4 Pick: Architecture',
        'components on the robot and the server, and what crosses the link')
DIV = 790
A.text(60, 185, 'ROBOT · Hello Robot Stretch 4 · Intel NUC 15 + Jetson Orin NX · ROS 2', 'colh')
A.text(820, 185, 'ED305 SERVER + INFRASTRUCTURE · RTX workstation · edge PC (requested)', 'colh')

# robot column
A.box(60, 220, 700, 150, 'hw', 'Sensors + actuators')
for i, (t, l) in enumerate([('2× 3-D LiDAR', 'hemispherical, head'),
                            ('Central RGB', 'gripper workspace'),
                            ('2× fisheye RGB', 'global shutter')]):
    A.card(84 + i * 226, 262, 210, 52, t, [l], ts=17, fs=14)
A.card(84, 322, 436, 36, '', ['IMU · diff-drive base · lift · telescoping arm · wrist · gripper'],
       fs=14)
A.card(530, 322, 210, 36, '', ['compute: NUC 15 + Orin NX'], fs=14)

A.box(60, 390, 700, 112, 'loc', 'Localization', 'on the robot')
A.card(84, 430, 330, 60, 'LiDAR localization', ['on the ED305 map (2Fast-2Lamaa, testing)'],
       ts=17, fs=14)
A.card(430, 430, 310, 60, 'Map frame = twin frame', ['robot pose in the map, 10 Hz'],
       ts=17, fs=14)

A.box(60, 522, 700, 100, 'task', 'Task executive', 'stage machine')
A.card(84, 562, 656, 46, '', ['coarse → mid → fine → verify · publishes the stage · '
                              'triggers captures · retry or go'], fs=15)

A.box(60, 642, 700, 150, 'per', 'Object fusion', 'on the robot · LiDAR + RGB in')
A.card(84, 682, 210, 96, 'Mask + LiDAR', ['LiDAR points inside', 'the SAM 3 mask', '1 s accumulated'],
       ts=17, fs=14)
A.card(308, 682, 210, 96, 'Shape fit', ['cylinder / box RANSAC', '→ object pose + axis',
                                         'in the base frame'], ts=17, fs=14)
A.card(532, 682, 208, 96, 'Local tracking', ['camera, every frame', 'known size → distance',
                                              'fine stage only'], ts=17, fs=14)

A.box(60, 812, 700, 112, 'man', 'Motion', 'on the robot')
A.card(84, 852, 210, 58, 'Nav2', ['base · coarse'], ts=17, fs=14)
A.card(308, 852, 210, 58, 'Base + lift + arm', ['rotate–translate–rotate · mid'], ts=17, fs=14)
A.card(532, 852, 208, 58, 'Arm + wrist + gripper', ['servo · contact by effort'],
       ts=16, fs=14)

A.box(60, 944, 700, 180, 'com', 'Comms bridge · robot', 'ROS 2 nodes')
A.card(84, 984, 210, 124, 'offload_client', ['capture on trigger', 'crop + JPEG',
                                              'ID · capture time · pose',
                                              'timeout · retry · drop stale'],
       border=TEAL, dash=True, ts=17, fs=14)
A.card(308, 984, 210, 124, 'link_monitor', ['heartbeat: RTT, queue', 'RSSI + link map',
                                             '→ expected answer time'],
       border=TEAL, dash=True, ts=17, fs=14)
A.card(532, 984, 208, 124, 'policy', ['send when stopped', 'wait / retry / fallback',
                                       'speed limit → Nav2'],
       border=TEAL, dash=True, ts=17, fs=14)

# server column
A.box(820, 220, 360, 200, 'twin', 'ED305 twin', 'offline')
A.card(842, 262, 316, 70, 'Reference map', ['OS1 + GLIM · keyframes, objects'], ts=17, fs=14)
A.card(842, 342, 316, 62, 'Link map', ['RSSI · RTT · goodput per cell'],
       border=TEAL, dash=True, ts=17, fs=14)
A.box(1200, 220, 340, 200, 'hw', 'Camera network', 'wired · 360°')
A.card(1222, 262, 296, 96, 'Infrastructure cameras', ['calibrated in the map frame',
                                                       'live object + person detections'],
       dash=True, ts=17, fs=14)
A.box(1560, 220, 380, 200, 'task', 'Task')
A.card(1582, 262, 336, 96, 'Request → object query', ['“give me the Pringles”',
                                                       '→ class, instance, deliver to'],
       dash=True, ts=17, fs=14)

A.box(820, 445, 1120, 190, 'per', 'Object retrieval', 'coarse: where is it?')
for i, (t, l) in enumerate([('Keyframe search', ['Perception Encoder', 'twin keyframes + cameras']),
                            ('SAM 3', ['text prompt → masks', 'room views + robot crops']),
                            ('Object cache', ['pose, class, ID', 'fused across views']),
                            ('Target + view pose', ['object pose in map', '→ base pose that sees it'])]):
    x = 842 + i * 272
    A.card(x, 488, 254, 92, t, l, dash=True, ts=17, fs=14)
for i in range(3):
    x = 842 + i * 272 + 254
    A.line([(x, 534), (x + 16, 534)])

A.box(820, 660, 1120, 160, 'com', 'Comms bridge · server', 'ROS 2 nodes')
A.card(842, 702, 540, 102, 'offload_server', ['decode JPEG → SAM 3 input · match answer to request ID',
                                               'add server times · drop requests past deadline',
                                               'report queue length'],
       border=TEAL, dash=True, ts=17, fs=14)
A.card(1398, 702, 520, 102, 'logger', ['per request: capture, send, receive, answer times',
                                        'size, stage, RSSI, RTT', '→ link map'],
       border=TEAL, dash=True, ts=17, fs=14)

A.box(820, 845, 1120, 205, 'hw', 'Link robot ↔ server', 'Wi-Fi 5 GHz · ROS 2 topics')
rows = [('↓ Map', 'once · per map version'),
        ('↓ Target + view pose', 'per task · ID + capture time'),
        ('↑ JPEG crop', 'per request · ~50 KB · RGB only'),
        ('↓ SAM 3 mask', 'per request · small')]
rows2 = [('↑ Robot pose', '10 Hz · ~1 KB/s'),
         ('↕ Heartbeat', '5–10 Hz · RTT, queue'),
         ('✕ never sent', 'raw images, depth, LiDAR scans')]
for i, (a, b) in enumerate(rows):
    A.text(842, 890 + i * 30, a, 'lk')
    A.text(1045, 892 + i * 30, b, 'lv')
for i, (a, b) in enumerate(rows2):
    A.text(1400, 890 + i * 30, a, 'lk')
    A.text(1560, 892 + i * 30, b, 'lv')
A.text(842, 1020, 'QoS: images best effort, keep last 1 · answers reliable · '
       'DSCP priority for requests · DDS tuned for large messages', 'lv', style=f'color:{TEAL};font-weight:600')

legend(A, 820, 1078, [('hw', 'hardware, link'), ('loc', 'localization'), ('task', 'task'),
                      ('per', 'perception'), ('man', 'motion'), ('twin', 'twin'),
                      ('com', 'comms bridge')])
A.svg.append(f'<line x1="820" y1="1132" x2="850" y2="1132" stroke="{RED}" stroke-width="3" '
             f'stroke-dasharray="9,6"/>')
A.text(860, 1121, 'network boundary', 'lg')
A.out.append('<div class="sw" style="left:1060px;top:1121px;background:#fff;'
             'border:2px dashed #999"></div>')
A.text(1100, 1121, 'dashed: not built yet', 'lg')

# arrows, robot column
A.line([(250, 370), (250, 428)])
A.line([(560, 370), (560, 380), (775, 380), (775, 730), (742, 730)])
A.line([(410, 502), (410, 520)])
A.line([(410, 622), (410, 640)])
A.line([(410, 792), (410, 810)])
A.line([(150, 942), (150, 926)])
# map to robot localization
# bridges <-> link
A.line([(1380, 820), (1380, 843)], color=TEAL, both=True)
A.line([(820, 960), (765, 960), (765, 1046), (742, 1046)], color=TEAL, both=True)
# divider
A.svg.append(f'<line x1="{DIV}" y1="175" x2="{DIV}" y2="1190" stroke="{RED}" stroke-width="3" '
             f'stroke-dasharray="14,9"/>')

# =====================================================================
# Figure 2: pipeline with the hierarchy of movement and comms
# =====================================================================
P = Fig(2000, 1140, 'Stretch 4 Pick: Pipeline',
        'coarse to fine: what moves, how it is perceived, what crosses the link')
cols = [('0 · Task', 'request', '#efeafb'),
        ('1 · Coarse', 'find + drive · room', '#e4eefb'),
        ('2 · Mid', 'locate + align · object', '#fdf3e4'),
        ('3 · Fine', 'grasp · gripper', '#fde3d6'),
        ('4 · Verify', 'lift + check', '#eef0f3')]
X0, CW, G = 250, 330, 14
for i, (t, s, bg) in enumerate(cols):
    x = X0 + i * (CW + G)
    P.rect(x, 185, CW, 70, bg, rx=10)
    P.text(x, 192, t, 'col', w=CW)
    P.text(x, 226, s, 'cols', w=CW)
    if i < 4:
        P.line([(x + CW + 1, 220), (x + CW + G - 1, 220)])

# movement hierarchy (staircase)
P.text(40, 285, 'Movement', 'lane')
P.text(40, 312, 'hierarchy', 'lane')
P.text(40, 342, 'scope shrinks,', 'lanes')
P.text(40, 360, 'precision rises', 'lanes')
steps = [(1, 'Room', 'base · Nav2', '~10 cm · 0.3–0.5 m/s', 275, 130),
         (2, 'Object', 'base + lift + arm', '~1–2 cm · slow, stop to look', 300, 105),
         (3, 'Gripper', 'arm + wrist + gripper', '~5 mm · servo, contact by effort', 325, 80),
         (4, 'Gripper', 'lift + retract', 'check, then return', 300, 105)]
P.rect(X0, 275, CW, 130, '#f6f3fd', rx=8)
P.text(X0, 312, 'no motion', 'cols', w=CW)
P.text(X0, 334, 'server parses the request', 'cols', w=CW)
for i, title, what, prec, y, h in steps:
    x = X0 + i * (CW + G)
    bg = cols[i][2]
    P.rect(x, y, CW, h, bg, rx=8, stroke='#ccc')
    P.text(x, y + 10, f'<b>{title}-level</b>', 'cols', w=CW, style='font-size:18px;color:#222')
    P.text(x, y + 36, what, 'cols', w=CW)
    P.text(x, y + 56, prec, 'cols', w=CW, style='font-weight:600;color:#333')

lanes = [
    ('Perception', 'what is seen, by which model', 425, 112, [
        ['SAM 3 query → object class', 'and instance'],
        ['room cameras + twin keyframes', '→ keyframe search + SAM 3', '→ object pose + view pose',
         'robot: LiDAR localization'],
        ['central RGB crop → SAM 3 mask', 'LiDAR points in mask (1 s)', '→ cylinder fit → pose + axis'],
        ['SAM 3 mask once, then local', 'tracking on the camera', 'known size → distance',
         'LiDAR as a check'],
        ['gripper aperture + effort', 'camera check']]),
    ('Runs on', 'where the loop closes', 557, 46, [
        ['server'], ['server finds, robot drives'], ['server segments, robot fuses'],
        ['robot only'], ['robot (server optional)']]),
    ('Comms', 'what crosses the link', 623, 92, [
        ['↑ query text', '↓ object class'],
        ['↓ target + view pose', '(bytes, with capture time)', '↑ robot pose 10 Hz'],
        ['robot stops at a good-link spot', '↑ 1 JPEG crop (~50 KB)', '↓ 1 mask'],
        ['↑ 1 JPEG crop at the start', '↓ 1 mask', 'then nothing'],
        ['↑ 1 crop, optional', '↓ ok / retry']]),
    ('Answer due', 'timeout set from the link', 735, 46, [
        ['seconds'], ['seconds (robot drives meanwhile)'], ['~0.5–1 s (robot stopped)'],
        ['~0.5 s once · local loop 30 Hz'], ['~1 s']]),
    ('If late', 'fallback', 801, 70, [
        ['retry'], ['keep last valid target', 'slow down'], ['retry once', 'then onboard detector'],
        ['keep tracking locally', 'new crop only if lost'], ['local check only']]),
]
for name, sub, y, h, cells in lanes:
    P.text(40, y + 8, name, 'lane')
    P.text(40, y + 34, sub, 'lanes', w=200)
    for i, lines in enumerate(cells):
        x = X0 + i * (CW + G)
        com = name in ('Comms', 'Answer due', 'If late')
        P.card(x, y, CW, h, '', lines, border=TEAL if com else '#c8c8c8', dash=com,
               fs=15, bg='#f2fbfa' if com else '#fff')
    P.svg.append(f'<line x1="30" y1="{y - 8}" x2="1960" y2="{y - 8}" stroke="#e3e3e3" stroke-width="1"/>')

# link dependence bar
y = 893
P.text(40, y + 4, 'Link', 'lane')
P.text(40, y + 30, 'dependence', 'lanes', style='font-size:15px;font-weight:700;color:#333')
dep = [('high', '#0b7f74'), ('high: targets come from the server', '#2a9d8f'),
       ('medium: one request, then local', '#76c7bd'), ('low: one request, then none', '#bfe5e0'),
       ('none–low', '#e3f4f2')]
for i, (s, c) in enumerate(dep):
    x = X0 + i * (CW + G)
    P.rect(x, y, CW, 44, c, rx=6)
    P.text(x, y + 11, s, 'cols', w=CW,
           style=f'color:{"#fff" if i < 2 else "#123"};font-weight:700;font-size:16px')

P.text(40, 959, 'Precision', 'lane')
P.text(40, 985, 'needed', 'lanes', style='font-size:15px;font-weight:700;color:#333')
prec = ['—', '~10 cm', '~1–2 cm', '~5 mm', 'grasp held']
for i, s in enumerate(prec):
    x = X0 + i * (CW + G)
    P.rect(x, 955, CW, 44, ['#f6f3fd', '#dbe8fb', '#fbe6c4', '#f8cdb6', '#eef0f3'][i], rx=6)
    P.text(x, 966, s, 'cols', w=CW, style='color:#222;font-weight:700;font-size:16px')

P.line([(X0 + 345, 1030), (X0 + 4 * (CW + G) + CW - 10, 1030)], color='#555', width=3)
P.text(X0 + 345, 1040, 'finer motion → smaller region → fewer, smaller messages → '
       'loop moves from the server to the robot', 'lbl', w=1300,
       style='font-size:17px;font-weight:600;color:#333')
P.text(40, 1085, 'Raw images, depth and LiDAR scans never cross the link; '
       'every request carries an ID and capture time, every answer echoes them.',
       'lbl', w=1900, style='font-size:16px;color:#555;font-style:italic')

# =====================================================================
# Figure 3: general architecture (platform-agnostic)
# =====================================================================
G = Fig(2000, 1230, 'Coarse-to-Fine Pick: General Architecture',
        'any mobile manipulator and site: robot, edge server, and what crosses the link')
DIV = 790
G.text(60, 185, 'ROBOT · any mobile manipulator · onboard CPU + GPU · ROS 2', 'colh')
G.text(820, 185, 'EDGE SERVER + SITE INFRASTRUCTURE · GPU server', 'colh')

# robot column
G.box(60, 220, 700, 150, 'hw', 'Sensors + actuators')
for i, (t, l) in enumerate([('Depth source', 'RGB-D camera or 3-D LiDAR'),
                            ('RGB camera(s)', 'head and / or wrist'),
                            ('Odometry + IMU', 'wheels, IMU')]):
    G.card(84 + i * 226, 262, 210, 52, t, [l], ts=17, fs=14)
G.card(84, 322, 436, 36, '', ['mobile base · arm · wrist · gripper'],
       fs=14)
G.card(530, 322, 210, 36, '', ['compute: CPU + GPU'], fs=14)

G.box(60, 390, 700, 112, 'loc', 'Localization', 'on the robot')
G.card(84, 430, 330, 60, 'Map-based localization', ['2-D scan (AMCL), 3-D scan or visual'],
       ts=17, fs=14)
G.card(430, 430, 310, 60, 'Map frame = twin frame', ['robot pose in the map, 10 Hz'],
       ts=17, fs=14)

G.box(60, 522, 700, 100, 'task', 'Task executive', 'stage machine')
G.card(84, 562, 656, 46, '', ['coarse → mid → fine → verify · publishes the stage · '
                              'triggers captures · retry or go'], fs=15)

G.box(60, 642, 700, 150, 'per', 'Object fusion', 'on the robot · RGB + depth in')
G.card(84, 682, 210, 96, 'Mask + depth', ['depth points inside', 'the segmentation mask', '(camera or LiDAR)'],
       ts=17, fs=14)
G.card(308, 682, 210, 96, 'Shape fit', ['cylinder / box RANSAC', '→ object pose + axis',
                                         'in the base frame'], ts=17, fs=14)
G.card(532, 682, 208, 96, 'Local tracking', ['camera, every frame', 'depth or known size', 'fine stage only'], ts=17, fs=14)

G.box(60, 812, 700, 112, 'man', 'Motion', 'on the robot')
G.card(84, 852, 210, 58, 'Navigation', ['base · coarse'], ts=17, fs=14)
G.card(308, 852, 210, 58, 'Base + arm', ['approach and align · mid'], ts=17, fs=14)
G.card(532, 852, 208, 58, 'Arm + wrist + gripper', ['servo · contact by force'],
       ts=16, fs=14)

G.box(60, 944, 700, 180, 'com', 'Comms bridge · robot', 'ROS 2 nodes')
G.card(84, 984, 210, 124, 'offload_client', ['capture on trigger', 'crop + JPEG',
                                              'ID · capture time · pose',
                                              'timeout · retry · drop stale'],
       border=TEAL, dash=True, ts=17, fs=14)
G.card(308, 984, 210, 124, 'link_monitor', ['heartbeat: RTT, queue', 'RSSI + link map',
                                             '→ expected answer time'],
       border=TEAL, dash=True, ts=17, fs=14)
G.card(532, 984, 208, 124, 'policy', ['send when stopped', 'wait / retry / fallback',
                                       'speed limit → navigation'],
       border=TEAL, dash=True, ts=17, fs=14)

# server column
G.box(820, 220, 360, 200, 'twin', 'Site twin', 'offline')
G.card(842, 262, 316, 70, 'Reference map', ['any SLAM · keyframes, objects'], ts=17, fs=14)
G.card(842, 342, 316, 62, 'Link map', ['RSSI · RTT · goodput per cell'],
       border=TEAL, dash=True, ts=17, fs=14)
G.box(1200, 220, 340, 200, 'hw', 'Site cameras', 'optional · wired')
G.card(1222, 262, 296, 96, 'Fixed cameras', ['calibrated in the map frame',
                                                       'live object detections'],
       dash=True, ts=17, fs=14)
G.box(1560, 220, 380, 200, 'task', 'Task')
G.card(1582, 262, 336, 96, 'Request → object query', ['natural-language request',
                                                       '→ class, instance, deliver to'],
       dash=True, ts=17, fs=14)

G.box(820, 445, 1120, 190, 'per', 'Object retrieval', 'coarse: where is it?')
for i, (t, l) in enumerate([('Place search', ['image–text embeddings', 'twin keyframes + live views']),
                            ('Segmentation', ['open-vocabulary (e.g. SAM 3)', 'site views + robot crops']),
                            ('Object cache', ['pose, class, ID', 'fused across views']),
                            ('Target + view pose', ['object pose in map', '→ base pose that sees it'])]):
    x = 842 + i * 272
    G.card(x, 488, 254, 92, t, l, dash=True, ts=17, fs=14)
for i in range(3):
    x = 842 + i * 272 + 254
    G.line([(x, 534), (x + 16, 534)])

G.box(820, 660, 1120, 160, 'com', 'Comms bridge · server', 'ROS 2 nodes')
G.card(842, 702, 540, 102, 'offload_server', ['decode JPEG → model input · match answer to request ID',
                                               'add server times · drop requests past deadline',
                                               'report queue length'],
       border=TEAL, dash=True, ts=17, fs=14)
G.card(1398, 702, 520, 102, 'logger', ['per request: capture, send, receive, answer times',
                                        'size, stage, RSSI, RTT', '→ link map'],
       border=TEAL, dash=True, ts=17, fs=14)

G.box(820, 845, 1120, 205, 'hw', 'Link robot ↔ server', 'Wi-Fi / 5G · ROS 2 topics')
rows = [('↓ Map', 'once · per map version'),
        ('↓ Target + view pose', 'per task · ID + capture time'),
        ('↑ JPEG crop', 'per request · ~50 KB · RGB only'),
        ('↓ Mask', 'per request · small')]
rows2 = [('↑ Robot pose', '10 Hz · ~1 KB/s'),
         ('↕ Heartbeat', '5–10 Hz · RTT, queue'),
         ('✕ never sent', 'raw images, depth, point clouds')]
for i, (a, b) in enumerate(rows):
    G.text(842, 890 + i * 30, a, 'lk')
    G.text(1045, 892 + i * 30, b, 'lv')
for i, (a, b) in enumerate(rows2):
    G.text(1400, 890 + i * 30, a, 'lk')
    G.text(1560, 892 + i * 30, b, 'lv')
G.text(842, 1020, 'QoS: images best effort, keep last 1 · answers reliable · '
       'DSCP priority for requests · DDS tuned for large messages', 'lv', style=f'color:{TEAL};font-weight:600')

legend(G, 820, 1078, [('hw', 'hardware, link'), ('loc', 'localization'), ('task', 'task'),
                      ('per', 'perception'), ('man', 'motion'), ('twin', 'twin'),
                      ('com', 'comms bridge')])
G.svg.append(f'<line x1="820" y1="1132" x2="850" y2="1132" stroke="{RED}" stroke-width="3" '
             f'stroke-dasharray="9,6"/>')
G.text(860, 1121, 'network boundary', 'lg')
G.out.append('<div class="sw" style="left:1060px;top:1121px;background:#fff;'
             'border:2px dashed #999"></div>')
G.text(1100, 1121, 'dashed: not built yet', 'lg')

# arrows, robot column
G.line([(250, 370), (250, 428)])
G.line([(560, 370), (560, 380), (775, 380), (775, 730), (742, 730)])
G.line([(410, 502), (410, 520)])
G.line([(410, 622), (410, 640)])
G.line([(410, 792), (410, 810)])
G.line([(150, 942), (150, 926)])
# map to robot localization
# bridges <-> link
G.line([(1380, 820), (1380, 843)], color=TEAL, both=True)
G.line([(820, 960), (765, 960), (765, 1046), (742, 1046)], color=TEAL, both=True)
# divider
G.svg.append(f'<line x1="{DIV}" y1="175" x2="{DIV}" y2="1190" stroke="{RED}" stroke-width="3" '
             f'stroke-dasharray="14,9"/>')


here = Path(__file__).parent
(here / 'fig_architecture.html').write_text(A.html('Stretch 4 Architecture'), encoding='utf-8')
(here / 'fig_pipeline.html').write_text(P.html('Stretch 4 Pipeline'), encoding='utf-8')
print('ok')

(here / 'fig_architecture_general.html').write_text(G.html('General Architecture'), encoding='utf-8')
