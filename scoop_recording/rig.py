#!/usr/bin/env python3
"""Open the rig as native terminal tabs: one window per machine, one tab each.

    ./rig.py survey_1                 # pick machines, open their windows
    ./rig.py survey_1 --all
    ./rig.py survey_1 --only infra_1
    ./rig.py survey_1 --print         # show what each tab will do, open nothing
    ./rig.py survey_1 --run           # also press Enter on each tab's command

Every tab ends on an interactive shell that has already hopped over ssh, entered
the container if there is one, cd'd, and sourced -- with its command sitting in
bash history. Press Up, then Enter.

Before the windows open, the recording files (record.yaml, record.sh, svo.sh,
rosbag_qos_override.yaml, csi_params.sh) are copied from this folder to each
machine's Dataset/scripts, over ssh, so every machine records from the same
record.yaml. `--no-sync` skips that; `--print` only lists what would be copied.

Typed-not-run is the default on purpose: you read the command before it
executes, and a wrong pass name costs nothing.
"""

import argparse
import base64
import datetime
import os
import shlex
import shutil
import subprocess
import time
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

BANNER = r"printf '\n\033[1;33m  ↑ then Enter:\033[0m  %s\n\n' "


def subst(text, subs):
    """Replace only our placeholders.

    str.format() would choke on shell expansions like ${ISAAC_ROS_WS} and
    $HOME, which are meant for the shell, not for Python.
    """
    for k, v in subs.items():
        text = text.replace("{%s}" % k, str(v))
    return text


def choose(targets):
    """Interactive picker. Returns chosen names in file order."""
    names = list(targets)
    print("platforms:\n")
    for i, n in enumerate(names, 1):
        sh = targets[n].get("shell") or {}
        where = ("ssh %s" % sh["ssh"]) if (sh.get("ssh") or "").strip() else "(local)"
        if sh.get("container"):
            where += " + container"
        print("  %d) %-18s %2d tabs   %s" % (i, n, len(targets[n].get("tabs") or []), where))
    print("\n  a) all\n  q) quit\n")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)
        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit"):
            sys.exit(0)
        if raw.lower() in ("a", "all"):
            return names
        picked, bad = [], []
        for tok in raw.replace(",", " ").split():
            if tok.isdigit() and 1 <= int(tok) <= len(names):
                picked.append(names[int(tok) - 1])
            elif tok in names:
                picked.append(tok)
            else:
                bad.append(tok)
        if bad:
            print("  ? %s" % ", ".join(bad))
            continue
        return list(dict.fromkeys(picked))       # de-dup, keep file order


def inner_rc(title, cd, setup, cmd, run, also=()):
    """The rcfile the tab's final interactive bash starts from."""
    lines = [
        "[ -f ~/.bashrc ] && . ~/.bashrc",
        r"printf '\033]0;%s\007'" % title,        # sets the tab label
    ]
    if cd:
        lines.append("cd %s 2>/dev/null" % cd)
    lines += list(setup)
    # Oldest first, so the main command is one Up-arrow away and the extras
    # are two, three, ... -- a tab can carry a stop/teardown command next to
    # the start one without needing a tab of its own.
    for extra in also:
        lines.append("history -s %s" % shlex.quote(extra))
    if cmd:
        # history -s puts it one Up-arrow away without executing it.
        lines.append("history -s %s" % shlex.quote(cmd))
        lines.append(BANNER + shlex.quote(cmd))
        if run:
            lines.append(cmd)
    return "\n".join(lines) + "\n"


def wrap_bash(rc, tag):
    """Run an interactive bash from `rc`, shipping rc through base64.

    Base64 carries no quotes, so the same string survives being nested inside
    ssh "..." and docker exec bash -c '...' without any escaping games.
    """
    b64 = base64.b64encode(rc.encode()).decode()
        # $(id -u) keeps the path per-user: the same tab run as root and then as
    # admin would otherwise collide on a file the second user cannot write.
    # The single quotes belong to the outer shell, so the inner `bash -c`
    # expands this at runtime.
    path = "/tmp/rig_$(id -u)_%s.rc" % tag
    return ("bash -c 'echo %s | base64 -d > %s; exec bash --rcfile %s -i'"
            % (b64, path, path))


def tab_command(shell, tab_index, title, cmd, subs, run, also=(), extra=(), host=False):
    """The single shell command this tab runs, start to finish."""
    ssh = (shell.get("ssh") or "").strip()
    container = "" if host else (shell.get("container") or "").strip()
    cd = subst((shell.get("host_cd") if host else shell.get("cd")) or "", subs)
    setup = (([] if host else [subst(s, subs) for s in (shell.get("setup") or [])])
             + [subst(s, subs) for s in extra])
    tag = "%s_%d" % (title.replace("/", "_"), tab_index)

    # Tab 0 of a containerised machine creates and holds the container.
    # run_dev.sh starts it with --rm, so this tab must stay open.
    if container and tab_index == 0:
        create = subst(shell.get("create") or "", subs)
        inner = (r"printf '\033]0;%s\007'; " % title
                 + "cd %s && %s" % (subst(shell.get("host_cd") or "${ISAAC_ROS_WS}", subs), create))
        return "ssh -t %s %s" % (ssh, shlex.quote(inner)) if ssh else inner

    payload = wrap_bash(inner_rc(title, cd, setup, cmd, run, also), tag)
    if container:
        user = (shell.get("container_user") or "").strip()
        payload = "docker exec -it %s%s %s" % (
            ("-u %s " % user) if user else "", container, payload)
    if ssh:
        payload = "ssh -t %s %s" % (ssh, shlex.quote(payload))
    return payload


SYNC_FILES = ["record.yaml", "record.sh", "svo.sh", "rosbag_qos_override.yaml",
              "csi_params.sh"]


def scripts_dir(shell):
    """Where a machine keeps Dataset/scripts, on the HOST (not in its container).

    `scripts_dir:` in panes.yaml wins; otherwise <host_cd or cd>/Dataset/scripts,
    which is where the record tab does `cd Dataset`.
    """
    if shell.get("scripts_dir"):
        return shell["scripts_dir"]
    base = shell.get("host_cd") or shell.get("cd") or "~"
    return base.rstrip("/") + "/Dataset/scripts"


def sync_scripts(targets, just_print):
    """Copy SYNC_FILES from this folder to every machine in `targets`.

    Over ssh with rsync (scp if there is no rsync). A machine with no ssh host
    is this one: its files are copied only if its scripts folder is not this one.
    A machine that cannot be reached is reported and skipped, not fatal -- the
    windows still open.
    """
    files = [os.path.join(HERE, f) for f in SYNC_FILES
             if os.path.isfile(os.path.join(HERE, f))]
    print("sync: %s" % ", ".join(os.path.basename(f) for f in files))
    for name, cfg in targets.items():
        shell = cfg.get("shell") or {}
        ssh = (shell.get("ssh") or "").strip()
        dest = scripts_dir(shell)
        if not ssh:
            local = os.path.expanduser(dest)
            if os.path.realpath(local) == os.path.realpath(HERE):
                print("  %-18s this folder, nothing to copy" % name)
                continue
            cmds = [["mkdir", "-p", local], ["cp", "-p"] + files + [local + "/"]]
        elif shutil.which("rsync"):
            cmds = [["ssh", "-o", "ConnectTimeout=8", ssh, "mkdir -p %s" % dest],
                    ["rsync", "-az", "-e", "ssh -o ConnectTimeout=8"] + files
                    + ["%s:%s/" % (ssh, dest)]]
        else:
            cmds = [["ssh", "-o", "ConnectTimeout=8", ssh, "mkdir -p %s" % dest],
                    ["scp", "-o", "ConnectTimeout=8", "-p"] + files
                    + ["%s:%s/" % (ssh, dest)]]
        label = "%s:%s" % (ssh, dest) if ssh else dest
        if just_print:
            print("  %-18s -> %s" % (name, label))
            continue
        try:
            for c in cmds:
                subprocess.run(c, check=True, stdout=subprocess.DEVNULL, timeout=60)
            print("  %-18s -> %s  ok" % (name, label))
        except (subprocess.SubprocessError, OSError) as e:
            print("  %-18s -> %s  FAILED (%s) -- this machine keeps its old files"
                  % (name, label, e))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pass_name", metavar="pass")
    ap.add_argument("--yaml", default=os.path.join(HERE, "panes.yaml"))
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="execute each tab's command instead of just typing it")
    ap.add_argument("--print", dest="just_print", action="store_true")
    ap.add_argument("--no-sync", action="store_true",
                    help="do not copy record.yaml and the scripts to the machines")
    ap.add_argument("--terminal", default=None,
                    help="terminal emulator (default: from panes.yaml)")
    a = ap.parse_args()

    doc = yaml.safe_load(open(a.yaml)) or {}
    defaults = doc.get("defaults") or {}
    targets = doc.get("targets") or {}
    term = a.terminal or defaults.get("terminal", "gnome-terminal")
    domain = defaults.get("domain_id", 0)

    if a.only:
        missing = [t for t in a.only if t not in targets]
        if missing:
            sys.exit("no such target(s): %s (have: %s)"
                     % (", ".join(missing), ", ".join(sorted(targets))))
        targets = {t: targets[t] for t in a.only}
    elif not a.all:
        if not sys.stdin.isatty():
            sys.exit("not a terminal: pass --only <target> or --all")
        print("pass: %s\n" % a.pass_name)
        targets = {t: targets[t] for t in choose(targets)}
        print()

    plan = {}
    for name, cfg in targets.items():
        # {date} matches the stamp record.sh puts in the bag name, so the ZED
        # .svo and the rosbag land in the same folder under the same base name.
        subs = {"machine": name, "pass": a.pass_name, "domain": domain,
                "date": datetime.date.today().strftime("%Y%m%d")}
        shell = cfg.get("shell") or {}
        tabs = []
        for i, t in enumerate(cfg.get("tabs") or []):
            if not t.get("enabled", True):
                continue
            cmd = subst(t.get("cmd") or "", subs).strip()
            also = [subst(x, subs).strip() for x in (t.get("also") or [])]
            tabs.append((t["name"], cmd,
                         tab_command(shell, i, t["name"], cmd, subs, a.run,
                                     also, t.get("setup") or [],
                                     t.get("host", False))))
        plan[name] = tabs

    if not a.no_sync:
        sync_scripts(targets, a.just_print)

    for name, tabs in plan.items():
        print("== %s ==" % name)
        for tname, cmd, full in tabs:
            print("  [%-10s] %s" % (tname, cmd or "(plain shell)"))
            if a.just_print:
                print("               %s" % full)
        print()

    if a.just_print:
        return
    if not shutil.which(term):
        sys.exit("%s not found. install it, or pass --terminal <emulator>" % term)

    # One invocation per machine, with a --window/-e group per tab.
    #
    # Two forms do NOT work here:
    #   `--window -- cmd --tab -- cmd2`  -- everything after the first `--` is
    #       that tab's command, so the later --tab flags go to bash and vanish.
    #   separate calls with `--tab`      -- --tab attaches to whichever window
    #       has focus, which is the one rig.py was launched from, not the new
    #       one.
    # `-e` takes a single string, so groups parse cleanly and all the tabs
    # belong to the window opened by the first group. It is deprecated but
    # still works; each tab's command lives in its own script so gnome-terminal
    # never has to re-split our quoting.
    scriptdir = os.path.join("/tmp", "rig_%s" % a.pass_name)
    os.makedirs(scriptdir, exist_ok=True)

    for name, tabs in plan.items():
        argv = [term]
        for i, (tname, _, full) in enumerate(tabs):
            sh = os.path.join(scriptdir, "%s_%d_%s.sh"
                              % (name.replace("/", "-"), i, tname))
            with open(sh, "w") as f:
                f.write("#!/bin/bash\n%s\n" % full)
            os.chmod(sh, 0o755)
            argv += ["--window" if i == 0 else "--tab", "-e", "bash %s" % sh]
        try:
            subprocess.Popen(argv, start_new_session=True,
                             stderr=subprocess.DEVNULL)   # -e deprecation notice
        except OSError as e:
            sys.exit("could not launch %s: %s" % (term, e))
        print("opened %s (%d tabs)  scripts in %s" % (name, len(tabs), scriptdir))
        time.sleep(1.5)

    print("\nTab 0 of a containerised machine holds the container open -- "
          "closing it stops every other tab on that machine.")


if __name__ == "__main__":
    main()
