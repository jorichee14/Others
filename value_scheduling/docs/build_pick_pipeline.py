"""Full task pipeline (acknowledge -> hand over) with the room cameras and the Wi-Fi,
in the style of the cabinet-door slide. Writes fig_task_pipeline.html; render with
render_task_pipeline.js. Photos in pick_img/ are frames from the group's videos.
"""
from pathlib import Path

TEAL = '#0b7f74'
G = {  # group: (panel bg, title colour)
    'task': ('#efeafb', '#5b3fb0'),
    'nav': ('#e8f0fb', '#1a5fd0'),
    'man': ('#fbe9e7', '#a3171d'),
    'hand': ('#e6f4ea', '#1d7a3a'),
}
RUN = {'robot': ('Robot', '#1f6f43', '#dff3e6'), 'server': ('Server', '#5b3fb0', '#ece6fb')}
LOAD = {'none': ('Wi-Fi: none', '#e9ecef', '#555'),
        'tiny': ('Wi-Fi: tiny (bytes)', '#cfeee9', '#0b5f57'),
        'photo': ('Wi-Fi: one photo (~50 KB)', '#0b7f74', '#fff')}

groups = [
    ('task', 'Acknowledge', [
        dict(img=None, ph='room-camera view', title='Acknowledge', run=['server'], load='tiny',
             does='Turns “give me the Pringles from the cabinet” into two targets.',
             inp='request text · room-camera images',
             out='cabinet view pose and person position, each with the time it was seen',
             cam='find the cabinet and the person',
             up='request text', down='the two targets with their times',
             late='ask again')]),
    ('nav', 'Navigate to the cabinet', [
        dict(img='global_nav.png', title='Global navigation', run=['robot'], load='tiny',
             does='Nav2 drives to the cabinet view pose. AMCL and the robot’s lidar avoid obstacles.',
             inp='goal · 2D lidar scan',
             out='robot near the cabinet',
             cam='check the cabinet is still where the map says',
             up='robot pose (status only)', down='a new goal, only if the target changed',
             late='keep driving to the current goal'),
        dict(img='local_nav.png', title='Local navigation', run=['robot', 'server'], load='photo',
             does='One photo → handle point (server). Door plane from depth (robot). '
                  'Rotate – translate – rotate to 0.7 m.',
             inp='one RGB-D image, robot stopped',
             out='robot in position to open the door',
             cam='—',
             up='one JPEG photo', down='the handle’s pixel position',
             late='take a new photo, ask again')]),
    ('man', 'Open the cabinet', [
        dict(img=['grasp_handle.png', 'open.png', 'retract.png'], title='Grasp · open · retract',
             run=['robot'], load='none',
             does='Grasp the handle, swing the door open along its arc, retract the arm.',
             inp='handle position · door plane',
             out='cabinet open, arm clear of the door',
             cam='—',
             up=None, down=None, late='—')]),
    ('man', 'Pick up the Pringles', [
        dict(img='retract.png', title='Detect', run=['robot', 'server'], load='photo',
             does='One photo → Pringles mask (YOLOE). Mask × robot depth → 3-D centre.',
             inp='one RGB-D image · “pringles”',
             out='Pringles centre in the robot frame',
             cam='hint where the Pringles are, if a camera sees inside',
             up='one JPEG photo', down='the best mask',
             late='retry once, then use YOLOE on the robot'),
        dict(img='pick.png', title='Grasp + lift', run=['robot'], load='none',
             does='IK: pre-grasp → approach → close the gripper → lift 10 cm.',
             inp='Pringles centre · joint states',
             out='Pringles in the gripper',
             cam='—',
             up=None, down=None, late='—')]),
    ('nav', 'Return to the person', [
        dict(img=None, ph='room-camera view of the person', title='Return', run=['robot'],
             load='tiny',
             does='Nav2 drives to the person. The goal moves when the person moves.',
             inp='person position · 2D lidar scan',
             out='robot in front of the person',
             cam='track the person',
             up='robot pose (status only)', down='person position when it changes, with its time',
             late='slow down near the person, ask for an update')]),
    ('hand', 'Hand over', [
        dict(img='handover.png', title='Hand over', run=['robot'], load='none',
             does='Hold the Pringles at hand height; open the gripper when the person pulls.',
             inp='gripper effort',
             out='the person holds the Pringles',
             cam='—',
             up=None, down=None, late='—')]),
]


def chips(run):
    return ''.join(f'<span class="chip" style="color:{RUN[r][1]};background:{RUN[r][2]}">'
                   f'{RUN[r][0]}</span>' for r in run)


def card(c, n):
    if isinstance(c['img'], list):
        pic = '<div class="pics3">' + ''.join(
            f'<img src="pick_img/{f}">' for f in c['img']) + '</div>'
    elif c['img']:
        pic = f'<img class="pic" src="pick_img/{c["img"]}">'
    else:
        pic = f'<div class="ph">{c["ph"]}<br><span>(image to add)</span></div>'
    lbl, bg, fg = LOAD[c['load']]
    if c['up'] or c['down']:
        wifi = ''
        if c['up']:
            wifi += f'<div class="wl"><b>Robot → server:</b> {c["up"]}</div>'
        if c['down']:
            wifi += f'<div class="wl"><b>Server → robot:</b> {c["down"]}</div>'
        wifi += f'<div class="wl late"><b>If late:</b> {c["late"]}</div>'
    else:
        wifi = '<div class="wl">nothing crosses the Wi-Fi</div>'
    return f'''<div class="card">{pic}
<div class="t"><span class="n">{n}</span>{c["title"]}</div>
<div class="runs">Runs on {chips(c["run"])}</div>
<div class="d">{c["does"]}</div>
<div class="io"><b class="in">Input:</b> {c["inp"]}</div>
<div class="io"><b class="in">Output:</b> {c["out"]}</div>
<div class="cam"><b>Room cameras:</b> {c["cam"]}</div>
<div class="wifi"><div class="load" style="background:{bg};color:{fg}">{lbl}</div>{wifi}</div></div>'''


panels, n = [], 1
for kind, name, cards in groups:
    bg, fg = G[kind]
    body = ''
    for c in cards:
        body += card(c, n)
        n += 1
    panels.append(f'<div class="panel" style="background:{bg}"><div class="pt" style="color:{fg}">'
                  f'{name}</div><div class="row">{body}</div></div>')
flow = '<div class="arr">➜</div>'.join(panels)
task_line = ' → '.join(f'<span style="color:{G[k][1]}">{name}</span>' for k, name, _ in groups)
legend = (chips(['robot']) + ' runs on the robot &nbsp; ' + chips(['server']) +
          ' runs on the GPU server &nbsp;&nbsp;|&nbsp;&nbsp; ' +
          ''.join(f'<span class="load inl" style="background:{b};color:{f}">{lab}</span> '
                  for lab, b, f in LOAD.values()))

html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Pringles Task Pipeline</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ font-family:'Inter','DejaVu Sans',sans-serif; color:#1d1d1f; background:#fff;
        width:max-content; padding:28px 30px 30px }}
h1 {{ font-size:46px; font-weight:800 }}
.sub {{ font-size:21px; color:#555; margin:6px 0 14px }}
.task {{ font-size:25px; font-weight:800; margin-bottom:12px }}
.legend {{ font-size:17px; color:#444; margin-bottom:18px }}
.flow {{ display:flex; align-items:stretch }}
.panel {{ border-radius:18px; padding:14px 16px 16px }}
.pt {{ font-size:24px; font-weight:800; margin:0 0 12px 4px }}
.row {{ display:flex; gap:18px }}
.arr {{ align-self:center; font-size:30px; color:#555; padding:0 8px }}
.card {{ width:300px; background:#fff; border:2px solid #d6d6d6; border-radius:14px;
         padding:12px; display:flex; flex-direction:column; gap:8px; min-height:700px }}
.pic {{ width:274px; height:190px; object-fit:cover; border-radius:8px }}
.pics3 {{ display:flex; gap:4px }}
.pics3 img {{ width:89px; height:190px; object-fit:cover; border-radius:6px }}
.ph {{ width:274px; height:190px; border:2px dashed #aaa; border-radius:8px; display:flex;
       align-items:center; justify-content:center; text-align:center; color:#777; font-size:16px;
       flex-direction:column; background:#fafafa }}
.ph span {{ font-size:13px; color:#999 }}
.t {{ font-size:23px; font-weight:800; margin-top:2px }}
.n {{ display:inline-block; width:28px; height:28px; border-radius:14px; background:#333; color:#fff;
      font-size:16px; text-align:center; line-height:28px; margin-right:8px; vertical-align:3px }}
.runs {{ font-size:15px; color:#555 }}
.chip {{ display:inline-block; font-size:14px; font-weight:700; border-radius:10px;
         padding:2px 10px; margin-left:4px }}
.d {{ font-size:16px; color:#333; line-height:22px; min-height:88px }}
.io {{ font-size:15px; line-height:20px; color:#444 }}
.in {{ color:#1a5fd0 }}
.cam {{ background:#f3effc; border-radius:6px; padding:5px 8px; color:#3b2a7a; font-size:15px;
        line-height:20px }}
.wifi {{ border:2px solid #b9e3dc; border-radius:8px; padding:7px 8px; margin-top:auto;
         display:flex; flex-direction:column; gap:4px }}
.load {{ font-size:14px; font-weight:800; border-radius:6px; padding:3px 8px; align-self:flex-start }}
.load.inl {{ display:inline-block }}
.wl {{ font-size:15px; line-height:20px; color:#174f49 }}
.wl b {{ color:{TEAL} }}
.late b {{ color:#a3171d }}
.foot {{ font-size:18px; color:#444; margin-top:18px }}
.foot b {{ color:{TEAL} }}
</style></head><body>
<h1>Pringles from the cabinet: task pipeline</h1>
<div class="sub">Stretch robot + wired room cameras + GPU server · for each step: what runs where, what the room cameras add, and what crosses the robot’s Wi-Fi</div>
<div class="task">Robot: {task_line}</div>
<div class="legend">{legend}</div>
<div class="flow">{flow}</div>
<div class="foot">Nav2, AMCL, obstacle avoidance, depth and the arm all run on the robot. The room cameras are wired to the server, so their images never use the robot’s Wi-Fi. <b>Over Wi-Fi: only targets with their times, one photo per look, and the short answers.</b></div>
</body></html>'''

Path(__file__).with_name('fig_task_pipeline.html').write_text(html, encoding='utf-8')
print('ok')
