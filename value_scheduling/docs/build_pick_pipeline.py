"""Full task pipeline (acknowledge -> hand over) with the room cameras and the link,
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

groups = [
    ('task', 'Acknowledge', [
        dict(img=None, ph='request + room-camera view', title='Acknowledge',
             desc='Person: “Can you give me the Pringles from the cabinet?” The server turns it '
                  'into targets: the cabinet, the Pringles, and the person to deliver to.',
             inp='text request; room camera images',
             out='cabinet view pose and person position in the map, each with its age',
             cam='find the cabinet and the person (SAM 3 on the room views, depth from the map)',
             link='↑ request text · ↓ targets + age')]),
    ('nav', 'Navigate to the cabinet', [
        dict(img='global_nav.png', title='Global navigation',
             desc='Nav2 on the robot drives to the cabinet’s view pose: AMCL on the 2D lidar, and the robot’s own costmap avoids obstacles.',
             inp='2D lidar scan; cabinet view pose',
             out='the robot is near the cabinet',
             cam='confirm the cabinet’s position; send a new goal only if it changed',
             link='↓ new goal only if the target changed · ↑ robot pose (status) · late → keep the current goal'),
        dict(img='local_nav.png', title='Local navigation',
             desc='Gripper camera: handle keypoint + RANSAC door plane. Rotate – translate – rotate '
                  'to 0.7 m off the door.',
             inp='one RGB-D image (robot stopped)',
             out='the robot is in position to open the cabinet',
             cam='—',
             link='↑ one JPEG · ↓ handle keypoint (depth stays on the robot) · late → retry')]),
    ('man', 'Open the cabinet', [
        dict(img=['grasp_handle.png', 'open.png', 'retract.png'], title='Grasp · open · retract',
             desc='Grasp the handle, open along the door arc (door radius R, wrist offset r, θ), '
                  'then retract the arm and lower the wrist.',
             inp='handle pose; door plane',
             out='the cabinet is open and the arm is clear of the door',
             cam='optional: confirm the door is open',
             link='none · runs on the robot')]),
    ('man', 'Pick up the Pringles', [
        dict(img='retract.png', title='Detect',
             desc='Gripper camera snapshot → YOLOE-seg (“pringles”) → mask × depth → 3-D centre. '
                  'Small base moves if out of reach; detect again after each.',
             inp='RGB-D snapshot; prompt',
             out='Pringles centre in the base frame',
             cam='prior: where the Pringles are in the cabinet, if a room camera sees inside',
             link='↑ one JPEG · ↓ best mask (or YOLOE on the robot: nothing) · late → retry once'),
        dict(img='pick.png', title='Grasp + lift',
             desc='IK: pre-grasp → approach → close the gripper → lift 10 cm.',
             inp='object centre; joint states',
             out='the Pringles are in the gripper',
             cam='—',
             link='none · runs on the robot')]),
    ('nav', 'Return to the person', [
        dict(img=None, ph='room-camera view of the person', title='Return',
             desc='Nav2 on the robot drives to the person; the goal is updated when the room cameras see the person move.',
             inp='person position + age; 2D lidar scan',
             out='the robot is in front of the person',
             cam='live person position; re-sent when the person moves',
             link='↓ person position + age · ↑ robot pose · stale → slow down, ask for an update')]),
    ('hand', 'Hand over', [
        dict(img='handover.png', title='Hand over',
             desc='Raise the Pringles to hand height and hold; open the gripper when the person '
                  'pulls (gripper effort changes).',
             inp='gripper effort; person position',
             out='the person holds the Pringles',
             cam='optional: person’s hand position',
             link='none · runs on the robot')]),
]


def card(c, n):
    if isinstance(c['img'], list):
        pic = '<div class="pics3">' + ''.join(
            f'<img src="pick_img/{f}">' for f in c['img']) + '</div>'
    elif c['img']:
        pic = f'<img class="pic" src="pick_img/{c["img"]}">'
    else:
        pic = f'<div class="ph">{c["ph"]}<br><span>(image to add)</span></div>'
    return f'''<div class="card">{pic}
<div class="t"><span class="n">{n}</span>{c["title"]}</div>
<div class="d">{c["desc"]}</div>
<div class="io"><b class="in">Input:</b> {c["inp"]}</div>
<div class="io"><b class="in">Output:</b> {c["out"]}</div>
<div class="io cam"><b>Room cameras:</b> {c["cam"]}</div>
<div class="io link"><b>Link:</b> {c["link"]}</div></div>'''


panels, n = [], 1
for kind, name, cards in groups:
    bg, fg = G[kind]
    body = ''
    for c in cards:
        body += card(c, n)
        n += 1
    panels.append(f'<div class="panel" style="background:{bg}"><div class="pt" style="color:{fg}">'
                  f'{name}</div><div class="row">{body}</div></div>')
arrow = '<div class="arr">➜</div>'
flow = arrow.join(panels)

task_line = ' → '.join(
    f'<span style="color:{G[k][1]}">{name}</span>' for k, name, _ in groups)

html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Pringles Task Pipeline</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ font-family:'Inter','DejaVu Sans',sans-serif; color:#1d1d1f; background:#fff;
        width:max-content; padding:28px 30px 30px }}
h1 {{ font-size:46px; font-weight:800 }}
.sub {{ font-size:21px; color:#555; margin:6px 0 16px }}
.task {{ font-size:25px; font-weight:800; margin-bottom:20px }}
.task u {{ color:#111 }}
.flow {{ display:flex; align-items:stretch }}
.panel {{ border-radius:18px; padding:14px 16px 16px }}
.pt {{ font-size:24px; font-weight:800; margin:0 0 12px 4px }}
.row {{ display:flex; gap:18px }}
.arr {{ align-self:center; font-size:30px; color:#555; padding:0 8px }}
.card {{ width:300px; background:#fff; border:2px solid #d6d6d6; border-radius:14px;
         padding:12px; display:flex; flex-direction:column; gap:7px; min-height:690px }}
.pic {{ width:274px; height:204px; object-fit:cover; border-radius:8px }}
.pics3 {{ display:flex; gap:4px }}
.pics3 img {{ width:89px; height:204px; object-fit:cover; border-radius:6px }}
.ph {{ width:274px; height:204px; border:2px dashed #aaa; border-radius:8px; display:flex;
       align-items:center; justify-content:center; text-align:center; color:#777; font-size:16px;
       flex-direction:column; background:#fafafa }}
.ph span {{ font-size:13px; color:#999 }}
.t {{ font-size:23px; font-weight:800; margin-top:4px }}
.n {{ display:inline-block; width:28px; height:28px; border-radius:14px; background:#333; color:#fff;
      font-size:16px; text-align:center; line-height:28px; margin-right:8px; vertical-align:3px }}
.d {{ font-size:15.5px; color:#555; line-height:21px; min-height:105px }}
.io {{ font-size:15.5px; line-height:21px }}
.in {{ color:#1a5fd0 }}
.cam {{ background:#f3effc; border-radius:6px; padding:5px 7px; color:#3b2a7a }}
.link {{ background:#e6f6f3; border-radius:6px; padding:5px 7px; color:{TEAL}; font-weight:500 }}
.foot {{ font-size:18px; color:#444; margin-top:18px }}
.foot b {{ color:{TEAL} }}
</style></head><body>
<h1>Pringles from the cabinet: task pipeline</h1>
<div class="sub">Stretch + wired room cameras + GPU server · what each step does, what the room cameras add, and what crosses the robot’s Wi-Fi</div>
<div class="task"><u>Robot:</u> {task_line}</div>
<div class="flow">{flow}</div>
<div class="foot">Room cameras are wired to the server, so they never load the robot’s Wi-Fi. <b>Only targets with their age, one JPEG per look and the answers cross the link</b>; depth, lidar scans and video stay on the robot. Every request carries an ID and capture time.</div>
</body></html>'''

Path(__file__).with_name('fig_task_pipeline.html').write_text(html, encoding='utf-8')
print('ok')
