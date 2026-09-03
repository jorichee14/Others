"""Figures, tables and text snippets, written the same way by every stage.

    from mircpipe import report
    fig, ax = report.figure(1, 2, figsize=(12, 5))
    report.save_fig(fig, ctx.result("ntp", "fig_ntp"))     # .pdf and .png
    report.write_markdown(ctx.result("ntp", "ntp_summary.md"), "NTP", rows)
    report.write_latex_table(ctx.result("ntp", "ntp_subsection.tex"), df, caption=...)

Keeping this in one place is what makes a dozen analyses look like one
document instead of a dozen scripts.
"""
import os

STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
}


def use_agg():
    import matplotlib
    matplotlib.use("Agg")


def figure(nrows=1, ncols=1, **kw):
    use_agg()
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    return plt.subplots(nrows, ncols, **kw)


def save_fig(fig, stem, formats=("pdf", "png"), printer=print, close=True):
    """Save one figure under `stem` in each format. `stem` has no suffix."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
    fig.tight_layout()
    out = []
    for ext in formats:
        p = "%s.%s" % (stem, ext)
        fig.savefig(p)
        out.append(p)
    if close:
        plt.close(fig)
    if printer:
        printer("  wrote %s" % ", ".join(out))
    return out


def write_markdown(path, title, sections, printer=print):
    """sections: list of (heading, body) or plain strings; a DataFrame body is
    rendered as a markdown table."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    lines = ["# %s" % title, ""]
    for s in sections:
        head, body = s if isinstance(s, (tuple, list)) else (None, s)
        if head:
            lines += ["## %s" % head, ""]
        if hasattr(body, "to_markdown"):
            lines += [body.to_markdown(index=False), ""]
        else:
            lines += [str(body), ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    if printer:
        printer("  wrote %s" % path)
    return path


def write_latex_table(path, df, caption="", label="", printer=print, **kw):
    """A DataFrame as a standalone LaTeX table snippet, ready to \\input."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    body = df.to_latex(index=False, escape=True, **kw)
    with open(path, "w") as f:
        if caption or label:
            f.write("\\begin{table}[t]\n\\centering\n")
            f.write(body)
            if caption:
                f.write("\\caption{%s}\n" % caption)
            if label:
                f.write("\\label{%s}\n" % label)
            f.write("\\end{table}\n")
        else:
            f.write(body)
    if printer:
        printer("  wrote %s" % path)
    return path


def write_csv(path, df, printer=print):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    if printer:
        printer("  wrote %s" % path)
    return path
