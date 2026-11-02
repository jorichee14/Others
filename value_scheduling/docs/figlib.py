"""Shared drawing helpers for the pick figures (HTML + SVG, rendered with Playwright)."""
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


