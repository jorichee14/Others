#!/usr/bin/env python3
"""
Draw T_cam_radar as three orthographic views, in the CAMERA optical frame.

    python3 plot_cam_radar.py --quat x,y,z,w --xyz x,y,z [-o cam_radar.svg]

A quaternion says nothing to a human standing at the rig. This renders the same
transform as "the radar is HERE relative to the lens, pointing THAT way", which
is the form you can check against the hardware with a tape measure and an eye.

Camera optical frame: +X right, +Y DOWN, +Z forward along the optical axis.
The Y-is-down convention is the usual source of sign errors here, so every view
is labelled with which way is physically up rather than which axis is drawn up.
"""
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot

AL = 0.075                     # axis arrow length, metres
PW, PH = 300, 244              # panel size
MARGIN = 26                    # px kept clear inside a panel for labels
COL = {'cam': '#2f6fd0', 'radar': '#d04a9c', 'grid': '#eeeeea', 'ax': '#cdcdc8'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quat', required=True)
    ap.add_argument('--xyz', required=True)
    ap.add_argument('-o', '--out', default='cam_radar.svg')
    a = ap.parse_args()
    R = Rot.from_quat([float(v) for v in a.quat.split(',')]).as_matrix()
    t = np.array([float(v) for v in a.xyz.split(',')])

    # Each view is (name, subtitle, u_axis, v_axis, right_label, up_label) where
    # u/v map a camera-frame vector to (screen right, screen up).
    views = [
        ('FRONT', 'looking the way the camera looks', [1, 0, 0], [0, -1, 0], 'X right', 'up'),
        ('TOP', 'looking down on the rig', [1, 0, 0], [0, 0, 1], 'X right', 'Z forward'),
        ('SIDE', "from the camera's left", [0, 0, 1], [0, -1, 0], 'Z forward', 'up'),
    ]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="650" '
         f'viewBox="0 0 980 650" font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif">',
         '<style>.t{font-size:12px;font-weight:650;fill:#1a1a18;text-anchor:middle}'
         '.s{font-size:10px;fill:#6d6d68;text-anchor:middle}'
         '.l{font-size:9.5px;fill:#9a9a94}.n{font-size:11px;fill:#3a3a36}'
         '.nb{font-size:11px;font-weight:650;fill:#1a1a18}'
         '.h{font-size:15px;font-weight:680;fill:#1a1a18}'
         '.m{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;fill:#3a3a36}'
         '.ax{font-size:10px;font-weight:700}</style>',
         '<rect width="980" height="650" fill="#ffffff"/>',
         '<text x="26" y="30" class="h">Where the radar is, seen from the ZED left lens</text>',
         '<text x="26" y="48" class="s" text-anchor="start" style="text-anchor:start">'
         'Camera optical frame: +X right, '
         '+Y DOWN, +Z forward. Axis arrows 7.5 cm; each view is scaled to fit.</text>']

    for k, (nm, sub, uax, vax, rl, ul) in enumerate(views):
        ox, oy = 30 + k * 315, 86
        u, v = np.array(uax, float), np.array(vax, float)
        # Auto-fit rather than a fixed scale: the radar can sit anywhere relative
        # to the lens, and a hard-coded zoom silently pushes it off the panel.
        pts = [np.zeros(3)] + [np.eye(3)[i] * AL for i in range(3)] \
            + [t] + [t + R @ np.eye(3)[i] * AL for i in range(3)]
        uu = np.array([p @ u for p in pts]); vv = np.array([p @ v for p in pts])
        span_u, span_v = max(np.ptp(uu), 1e-3), max(np.ptp(vv), 1e-3)
        S = min((PW - 2 * MARGIN) / span_u, (PH - 2 * MARGIN) / span_v)
        cx = ox + MARGIN + (-uu.min()) * S
        cy = oy + PH - MARGIN - (-vv.min()) * S
        px = lambda p: (cx + float(p @ u) * S, cy - float(p @ v) * S)
        gstep = 0.05 if S * 0.05 > 14 else 0.10
        s.append(f'<rect x="{ox}" y="{oy}" width="{PW}" height="{PH}" fill="#fff" '
                 f'stroke="#d8d8d4"/>')
        for g in range(-12, 13):                      # grid
            gx, gy = cx + g * gstep * S, cy - g * gstep * S
            if ox < gx < ox + PW:
                s.append(f'<line x1="{gx:.1f}" y1="{oy}" x2="{gx:.1f}" y2="{oy+PH}" '
                         f'stroke="{COL["grid"]}"/>')
            if oy < gy < oy + PH:
                s.append(f'<line x1="{ox}" y1="{gy:.1f}" x2="{ox+PW}" y2="{gy:.1f}" '
                         f'stroke="{COL["grid"]}"/>')
        s.append(f'<text x="{ox+PW/2}" y="{oy-16}" class="t">{nm}</text>')
        s.append(f'<text x="{ox+PW/2}" y="{oy-4}" class="s">{sub}</text>')
        s.append(f'<text x="{ox+PW-6}" y="{oy+PH-6}" class="l" text-anchor="end">{rl} &#8594;</text>')
        s.append(f'<text x="{ox+6}" y="{oy+12}" class="l">&#8593; {ul}</text>')

        def sensor(origin, Rm, colour, label, names):
            x0, y0 = px(origin)
            for vec, nlab in zip(np.eye(3), names):
                w = Rm @ vec * AL
                x1, y1 = px(origin + w)
                # an axis pointing at the viewer projects to nothing — mark it
                if abs(x1 - x0) < 2 and abs(y1 - y0) < 2:
                    toward = float((Rm @ vec) @ np.cross(u, v))
                    s.append(f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="5" fill="none" '
                             f'stroke="{colour}" stroke-width="1.4"/>')
                    s.append(f'<text x="{x0+9:.1f}" y="{y0-6:.1f}" class="ax" fill="{colour}">'
                             f'{nlab}{"&#8857;" if toward > 0 else "&#8855;"}</text>')
                    continue
                s.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                         f'stroke="{colour}" stroke-width="1.8"/>')
                lx = min(max(x1, ox + 12), ox + PW - 12)
                ly = min(max(y1 - 4, oy + 12), oy + PH - 4)
                s.append(f'<text x="{lx:.1f}" y="{ly:.1f}" class="ax" fill="{colour}" '
                         f'text-anchor="middle">{nlab}</text>')
            s.append(f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="3.4" fill="{colour}"/>')
            lx = min(max(x0, ox + 28), ox + PW - 28)
            s.append(f'<text x="{lx:.1f}" y="{min(y0+17, oy+PH-4):.1f}" class="t" '
                     f'fill="{colour}">{label}</text>')

        sensor(np.zeros(3), np.eye(3), COL['cam'], 'ZED left', ['X', 'Y', 'Z'])
        sensor(t, R, COL['radar'], 'radar1', ['fwd', 'left', 'up'])
        x0, y0 = px(np.zeros(3)); x1, y1 = px(t)
        s.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                 f'stroke="#b9b9b3" stroke-width="1.2" stroke-dasharray="4 3"/>')

    b = R @ [1, 0, 0]
    y = 392
    s.append(f'<text x="30" y="{y}" class="h">Reading it</text>')
    rows = [('Position of the radar, from the lens centre', ''),
            (f'&#160;&#160;{abs(t[0])*100:.1f} cm to the '
             f'{"RIGHT" if t[0] > 0 else "LEFT"}', 'camera +X'),
            (f'&#160;&#160;{abs(t[1])*100:.1f} cm {"DOWN" if t[1] > 0 else "UP"}',
             'camera +Y is down'),
            (f'&#160;&#160;{abs(t[2])*100:.1f} cm '
             f'{"BEHIND" if t[2] < 0 else "AHEAD OF"} the lens', 'camera +Z is forward'),
            (f'&#160;&#160;straight-line separation {np.linalg.norm(t)*100:.1f} cm', ''),
            ('', ''),
            ('Orientation', ''),
            (f'&#160;&#160;boresight is {np.degrees(np.arccos(b[2])):.1f}&#176; off the optical axis',
             f'{np.degrees(np.arcsin(-b[1])):+.1f}&#176; up, '
             f'{np.degrees(np.arcsin(b[0])):+.1f}&#176; right'),
            (f'&#160;&#160;roll about the boresight '
             f'{np.degrees(np.arcsin(abs((R@[0,1,0])[2]))):.1f}&#176;', 'upright, not inverted')]
    for i, (a_, b_) in enumerate(rows):
        yy = y + 22 + i * 19
        cls = 'nb' if a_ and not a_.startswith('&#160;') else 'n'
        s.append(f'<text x="30" y="{yy}" class="{cls}">{a_}</text>')
        if b_:
            s.append(f'<text x="380" y="{yy}" class="l">{b_}</text>')
    s.append(f'<text x="560" y="{y}" class="h">Numbers</text>')
    q = Rot.from_matrix(R).as_quat()
    for i, ln in enumerate([f't (m)&#160;&#160;&#160;&#160; {t[0]:+.6f} {t[1]:+.6f} {t[2]:+.6f}',
                            f'quat xyzw {q[0]:+.6f} {q[1]:+.6f} {q[2]:+.6f} {q[3]:+.6f}',
                            '',
                            'p_cam = R &#183; p_radar + t']):
        s.append(f'<text x="560" y="{y+22+i*19}" class="m">{ln}</text>')
    s.append('</svg>')
    open(a.out, 'w').write('\n'.join(s))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
