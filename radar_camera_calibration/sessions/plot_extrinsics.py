#!/usr/bin/env python3
"""Draw the rig extrinsics (os_lidar / ZED left / radar1) as three orthographic
views plus the numeric translation + rotation tables. Values are the solved
transforms from 2026-08-19_ouster_radar1_lidar.md; edit q_lr/t_lr after a
re-solve. Writes extrinsics.svg."""
import numpy as np, math
from scipy.spatial.transform import Rotation as Rot
q_lc=[-0.497829,-0.498035,0.501789,0.502329]; t_lc=np.array([-0.074928,-0.066971,-0.091627])
R_lc=Rot.from_quat(q_lc).as_matrix()
q_lr=[0.111610,0.026730,0.993300,-0.013300]; t_lr=np.array([0.027900,0.133400,-0.210000])
R_lr=Rot.from_quat(q_lr).as_matrix()
S, AL = 620.0, 0.085          # px per metre, axis arrow length (m)
PW, PH = 300, 250
SENS = [('os_lidar', np.zeros(3), np.eye(3), '#12161c'),
        ('zed left', t_lc, R_lc, '#0c6b7a'),
        ('radar1',   t_lr, R_lr, '#a8331f')]
AXC = {'X': '#0b7285', 'Y': '#9c3a06', 'Z': '#5f3dc4'}

def panel(ox, oy, title, sub, ha, va, hl, vl, hf, vf, oi, osign):
    o, cx, cy = [], ox+PW/2, oy+120
    o.append(f'<text x="{ox+PW/2}" y="{oy+22}" class="pt">{title}</text>')
    o.append(f'<text x="{ox+PW/2}" y="{oy+38}" class="ps">{sub}</text>')
    o.append(f'<rect x="{ox+8}" y="{oy+48}" width="{PW-16}" height="{PH-48}" class="fr"/>')
    for g in range(-4, 5):
        if g == 0: continue
        gx, gy = cx+g*0.05*S*hf, cy+g*0.05*S*vf
        if ox+9 < gx < ox+PW-9: o.append(f'<line x1="{gx:.1f}" y1="{oy+48}" x2="{gx:.1f}" y2="{oy+PH}" class="gr"/>')
        if oy+49 < gy < oy+PH-1: o.append(f'<line x1="{ox+8}" y1="{gy:.1f}" x2="{ox+PW-8}" y2="{gy:.1f}" class="gr"/>')
    o.append(f'<line x1="{ox+8}" y1="{cy}" x2="{ox+PW-8}" y2="{cy}" class="ax0"/>')
    o.append(f'<line x1="{cx}" y1="{oy+48}" x2="{cx}" y2="{oy+PH}" class="ax0"/>')
    o.append(f'<text x="{ox+PW-12}" y="{cy-7}" class="al" text-anchor="end">{hl}</text>')
    o.append(f'<text x="{cx+6}" y="{oy+60}" class="al">{vl}</text>')
    # translation links
    for i in range(len(SENS)):
        for j in range(i+1, len(SENS)):
            a, b = SENS[i][1], SENS[j][1]
            o.append(f'<line x1="{cx+a[ha]*S*hf:.1f}" y1="{cy+a[va]*S*vf:.1f}" '
                     f'x2="{cx+b[ha]*S*hf:.1f}" y2="{cy+b[va]*S*vf:.1f}" class="lnk"/>')
    for nm, p, M, col in SENS:
        px, py = cx+p[ha]*S*hf, cy+p[va]*S*vf
        for k, a in enumerate('XYZ'):
            v = M[:, k]
            if abs(v[ha]) < 0.16 and abs(v[va]) < 0.16:
                toward = (v[oi]*osign) < 0
                o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="9" fill="none" '
                         f'stroke="{AXC[a]}" stroke-width="1.6" stroke-dasharray="2.5 2.5"/>')
                o.append(f'<text x="{px+12:.1f}" y="{py-9:.1f}" class="axl" fill="{AXC[a]}">'
                         f'{a}{"&#8857;" if toward else "&#8855;"}</text>')
            else:
                ex, ey = px+v[ha]*AL*S*hf, py+v[va]*AL*S*vf
                o.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                         f'stroke="{AXC[a]}" stroke-width="2.4" marker-end="url(#a{a})"/>')
                dx = 8 if ex >= px else -8
                o.append(f'<text x="{ex+dx:.1f}" y="{ey+(13 if ey>py else -6):.1f}" class="axl" '
                         f'fill="{AXC[a]}" text-anchor="{"start" if dx>0 else "end"}">{a}</text>')
        o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="{col}"/>')
        o.append(f'<text x="{px:.1f}" y="{py+20:.1f}" class="sn" fill="{col}" '
                 f'text-anchor="middle">{nm}</text>')
    return '\n'.join(o)

W, H = 980, 700
s = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"
 font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif">
<defs>''' + ''.join(
 f'<marker id="a{a}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" '
 f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for a, c in AXC.items()) + '''</defs>
<style>
 .pt{font-size:12.5px;font-weight:650;fill:#1a1a18;text-anchor:middle}
 .ps{font-size:10.5px;fill:#6d6d68;text-anchor:middle}
 .fr{fill:#ffffff;stroke:#d8d8d4;stroke-width:1}
 .gr{stroke:#f0f0ed;stroke-width:1} .ax0{stroke:#cdcdc8;stroke-width:1;stroke-dasharray:3 3}
 .al{font-size:9.5px;fill:#9a9a94} .sn{font-size:10.5px;font-weight:600}
 .axl{font-size:10.5px;font-weight:700}
 .lnk{stroke:#b9b9b3;stroke-width:1.4;stroke-dasharray:4 3}
 .h1{font-size:16px;font-weight:680;fill:#1a1a18} .h2{font-size:11px;fill:#6d6d68}
 .sec{font-size:12px;font-weight:680;fill:#1a1a18}
 .k{font-size:11.5px;fill:#3a3a36} .kb{font-size:11.5px;font-weight:650;fill:#1a1a18}
 .mn{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11.5px;fill:#3a3a36}
</style>
<rect x="0" y="0" width="%d" height="%d" fill="#fbfbfa"/>
<text x="26" y="32" class="h1">Rig extrinsics &#8212; os_lidar, ZED left, radar1</text>
<text x="26" y="50" class="h2">All drawn in the LIDAR frame (X forward, Y left, Z up). Grid 5 cm; axis arrows 8.5 cm.
 &#8857; out of page, &#8855; into page.</text>''' % (W, H)]
s.append(panel(20, 74, 'FRONT &#8212; looking along lidar +X', 'lidar +Y left, +Z up', 1, 2, 'lidar +Y (left)', 'lidar +Z (up)', -1, -1, 0, -1))
s.append(panel(340, 74, 'TOP &#8212; looking down', 'lidar +X forward, +Y left', 1, 0, 'lidar +Y (left)', 'lidar +X (fwd)', -1, -1, 2, -1))
s.append(panel(660, 74, 'SIDE &#8212; from lidar +Y', 'lidar +X forward, +Z up', 0, 2, 'lidar +X (fwd)', 'lidar +Z (up)', 1, -1, 1, -1))
y = 380
s.append(f'<text x="26" y="{y}" class="sec">TRANSLATION &#8212; positions in the lidar frame</text>')
for nm, p, _, col in SENS:
    y += 24
    s.append(f'<circle cx="34" cy="{y-4}" r="5" fill="{col}"/>')
    s.append(f'<text x="48" y="{y}" class="kb">{nm}</text>')
    s.append(f'<text x="150" y="{y}" class="mn">[{p[0]:+.4f}  {p[1]:+.4f}  {p[2]:+.4f}] m</text>')
    if nm != 'os_lidar':
        s.append(f'<text x="360" y="{y}" class="k">|t| = {np.linalg.norm(p)*100:.1f} cm from lidar</text>')
y += 26
s.append(f'<text x="150" y="{y}" class="k">camera &#8594; radar separation = '
         f'{np.linalg.norm(t_lr-t_lc)*100:.1f} cm</text>')
y += 40
s.append(f'<text x="26" y="{y}" class="sec">ROTATION &#8212; where each sensor&#8217;s own axes point, in the lidar frame</text>')
for nm, _, M, col in SENS[1:]:
    y += 26
    s.append(f'<text x="26" y="{y}" class="kb" fill="{col}">{nm}</text>')
    for k, a in enumerate('XYZ'):
        v = M[:, k]
        s.append(f'<text x="{150+k*228}" y="{y}" class="mn" fill="{AXC[a]}">'
                 f'{a}&#8594;[{v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}]</text>')
y += 34
tilt = math.degrees(math.asin(abs(R_lr[2, 0])))
s.append(f'<text x="26" y="{y}" class="sec">Reading</text>')
s.append(f'<text x="150" y="{y}" class="k">Radar and camera both look along lidar '
         f'<tspan class="mn">&#8722;X</tspan>; radar is UPRIGHT (its +Z within '
         f'{math.degrees(math.acos(R_lr[2,2])):.0f}&#176; of lidar +Z) and pitched '
         f'{tilt:.0f}&#176; down.</text>')
y += 22
s.append(f'<text x="150" y="{y}" class="k">Solved from 33 captures, 27 inliers, residual 1.33&#963; '
         f'&#183; rot 1&#963; 3.8/6.6/1.2&#176; &#183; t 1&#963; 11/31/60 mm</text>')
s.append('</svg>')
open('extrinsics.svg', 'w').write('\n'.join(s))
print('wrote extrinsics.svg')
