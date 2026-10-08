# Project tracker

`Project_Tracker_v3.xlsx`: Tasks sheet → per-person tabs, per-goal tabs, Gantt, Dashboard
(all automatic). Works in Excel and Google Sheets.

## One Google Sheet per person, plus the shared overview

Each person edits their own file; the master collects everyone's tasks and is what you share
with the whole team. Set up once:

1. Upload `Project_Tracker_v3.xlsx` to Google Drive, open it, **File > Save as Google Sheets**.
   This is the master.
2. In the master's **People** sheet: real names in column A (and in Tasks, Ctrl+H each
   placeholder such as `GS-SLAM owner` to the real name), emails in column D (optional: the
   script shares each person's file with that address).
3. **Extensions > Apps Script**, delete the sample code, paste `split_per_person.gs`, Save,
   choose `setupPersonFiles`, **Run**, approve the permissions (create spreadsheets, share files).
4. Back in the master, People column F: click each `#REF!` cell → **Allow access** (once per
   person). When it shows the person's name, that file is connected.
5. Share the master with the team (Viewer is enough; they edit their own files).

After that:
- People add and edit tasks in **their own file** (`Tasks - <name>`); the master updates within a
  minute or so.
- You own every person file, so you can add a task for anyone in their file.
- New person: add the name (and email) in People, run **Tracker > Create / update person files**
  again (the menu appears after reloading the master). Then Allow access for them in column F.
  Their tab in the master: duplicate a person tab and point its C2 at their People row.

The script stops without changing anything if a task is assigned to a name that is not in
People. Version history (File > Version history) has the master as it was before the split.

Not tested in Google here (no Google account in this environment): the script is
syntax-checked only. If a step fails, the error message from Apps Script says where.
