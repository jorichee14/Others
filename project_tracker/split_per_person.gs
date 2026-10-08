/**
 * Project tracker: one Google Sheet per person, collected into the master.
 *
 * Run this from the MASTER spreadsheet (Project_Tracker_v3 saved as Google Sheets):
 *   Extensions > Apps Script > paste this file > Save > run setupPersonFiles (or use the Tracker menu).
 *
 * What it does:
 *   1. For every name in People (column A) without a file yet, creates "Tasks - <name>" with a
 *      "My Tasks" sheet, copies that person's current rows from Tasks into it, and shares it with
 *      the email in People column D (if given). The file's link goes in People column E.
 *   2. Renames each person's tab in the master to their name (People column C).
 *   3. Replaces the master's Tasks rows with one formula that imports every person's "My Tasks".
 *      Everything else in the master (person tabs, goal tabs, Gantt, Dashboard) reads Tasks, so it
 *      keeps working.
 *
 * Safe to run again: it only creates files for people without one, and copies rows only on the
 * first run (while Tasks still holds typed rows). It stops before changing anything if a task is
 * assigned to a name that is not in People.
 */

const TASKS_SHEET = 'Tasks';
const PEOPLE_SHEET = 'People';
const FIRST_ROW = 4;              // first task row in the master's Tasks
const LAST_ROW = 403;             // last row the master's formulas read
const PEOPLE_FIRST = 4;           // first person row in People
const PEOPLE_SLOTS = 30;          // People rows 4..33
const PERSON_ROWS = 200;          // task rows in each person's file (rows 4..203)
const COLS = 9;                   // ID, Goal, Task, Assigned to, Status, Start, Weeks, End, Notes
const HEADERS = ['ID', 'Goal', 'Task', 'Assigned to', 'Status', 'Start', 'Weeks', 'End', 'Notes'];
const GOALS = ['G1 ED305 live digital twin', 'G2 SCooP dataset & paper', 'Shared SCooP testbed',
               'Dual-Link Communication', 'Semantic Communication'];
const STATUSES = ['Not Started', 'In Progress', 'Blocked', 'Complete', 'Partially Completed'];
const STATUS_FILL = {'Not Started': '#D9D9D9', 'In Progress': '#FFE699', 'Blocked': '#F4B6B6',
                     'Complete': '#C6E0B4', 'Partially Completed': '#F8CBAD'};

function onOpen() {
  SpreadsheetApp.getUi().createMenu('Tracker')
      .addItem('Create / update person files', 'setupPersonFiles')
      .addToUi();
}

function setupPersonFiles() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const ui = SpreadsheetApp.getUi();
  const people = ss.getSheetByName(PEOPLE_SHEET);
  const tasks = ss.getSheetByName(TASKS_SHEET);
  if (!people || !tasks) {
    ui.alert('Run this from the master tracker (it needs the Tasks and People sheets).');
    return;
  }

  people.getRange('D3:F3').setValues([['Email (optional)', 'Their file', 'Access (name = connected)']])
      .setFontWeight('bold').setFontColor('#FFFFFF').setBackground('#1F4E78');
  people.setColumnWidth(4, 220); people.setColumnWidth(5, 260); people.setColumnWidth(6, 200);

  const plist = people.getRange(PEOPLE_FIRST, 1, PEOPLE_SLOTS, 5).getValues();   // name, area, tab, email, link
  const names = plist.map(p => String(p[0]).trim()).filter(n => n);

  // First run: Tasks still holds typed rows -> they get copied into the person files.
  const firstRun = tasks.getRange(FIRST_ROW, 1).getFormula().indexOf('IMPORTRANGE') < 0;
  let rows = [];
  if (firstRun) {
    rows = tasks.getRange(FIRST_ROW, 1, LAST_ROW - FIRST_ROW + 1, COLS).getValues()
        .filter(r => String(r[2]).trim() !== '');
    const orphans = rows.filter(r => names.indexOf(String(r[3]).trim()) < 0);
    if (orphans.length) {
      ui.alert('Nothing changed. These tasks are assigned to a name that is not in People (or to nobody):\n\n' +
               orphans.slice(0, 15).map(r => '- ' + r[2] + '  ->  "' + r[3] + '"').join('\n') +
               (orphans.length > 15 ? '\n... and ' + (orphans.length - 15) + ' more' : '') +
               '\n\nFix Assigned to (or add the name to People), then run again.');
      return;
    }
  }

  const masterUrl = ss.getUrl();
  const created = [], shared = [];
  plist.forEach((p, i) => {
    const name = String(p[0]).trim();
    if (!name) return;
    const tab = String(p[2]).trim(), email = String(p[3]).trim();
    let link = String(p[4]).trim();
    const row = PEOPLE_FIRST + i;

    let file;
    if (link) {
      file = SpreadsheetApp.openByUrl(link);
    } else {
      file = createPersonFile_(name, masterUrl);
      link = file.getUrl();
      people.getRange(row, 5).setValue(link);
      created.push(name);
      const mine = rows.filter(r => String(r[3]).trim() === name);
      if (mine.length) {
        const sh = file.getSheetByName('My Tasks');
        sh.getRange(4, 1, mine.length, 3).setValues(mine.map(r => [r[0], r[1], r[2]]));      // ID, Goal, Task
        sh.getRange(4, 5, mine.length, 3).setValues(mine.map(r => [r[4], r[5], r[6]]));      // Status, Start, Weeks
        sh.getRange(4, 9, mine.length, 1).setValues(mine.map(r => [r[8]]));                  // Notes
      }
    }
    if (email) {
      try { DriveApp.getFileById(file.getId()).addEditor(email); shared.push(name); }
      catch (e) { Logger.log('Could not share with ' + email + ': ' + e); }
    }
    people.getRange(row, 6).setFormula('=IMPORTRANGE(E' + row + ',"\'My Tasks\'!B1")');

    // the person's tab in the master takes their name
    if (tab && tab !== name && ss.getSheetByName(tab) && !ss.getSheetByName(name)) {
      ss.getSheetByName(tab).setName(name);
      people.getRange(row, 3).setValue(name);
    }
  });

  // Master Tasks: rows now come from every person's file.
  const block = tasks.getRange(FIRST_ROW, 1, LAST_ROW - FIRST_ROW + 1, COLS);
  block.clearContent();
  block.clearDataValidations();
  tasks.getRange(FIRST_ROW, 1).setFormula(stackFormula_());
  tasks.getRange('A2').setValue('Rows come from each person\'s own file (People > Their file). ' +
                                'Add or edit tasks there; this sheet updates by itself.');

  ui.alert('Done.\n\nNew files: ' + (created.join(', ') || 'none') +
           '\nShared by email: ' + (shared.join(', ') || 'none') +
           '\n\nLast step: in People, column F, click each #REF! cell and press "Allow access" ' +
           '(once per person). A name there means that file is connected.');
}

/** One import per People slot, stacked; empty slots and empty rows drop out. */
function stackFormula_() {
  const blank = '{' + new Array(COLS).fill('""').join(',') + '}';
  const range = "'My Tasks'!A4:I" + (3 + PERSON_ROWS);   // quoted: the sheet name has a space
  const parts = [];
  for (let r = PEOPLE_FIRST; r < PEOPLE_FIRST + PEOPLE_SLOTS; r++) {
    parts.push('IFERROR(IMPORTRANGE(' + PEOPLE_SHEET + '!$E$' + r + ',"' + range + '"),' + blank + ')');
  }
  return '=IFERROR(QUERY({' + parts.join(';') + '},"select * where Col3 is not null and Col3 <> \'\'",0),"")';
}

function createPersonFile_(name, masterUrl) {
  const file = SpreadsheetApp.create('Tasks - ' + name);
  const sh = file.getSheets()[0].setName('My Tasks');
  const n = PERSON_ROWS, last = 3 + n;

  sh.getRange('A1:B1').setValues([['Tasks of', name]]).setFontWeight('bold').setFontSize(14).setFontColor('#1F4E78');
  sh.getRange('A2').setValue('Add or edit your tasks below (Goal, Task, Status, Start = a Monday, Weeks). ' +
                             'They show up in the shared tracker by themselves.');
  sh.getRange('A2').setFontStyle('italic').setFontColor('#595959');
  sh.getRange('F1').setFormula('=HYPERLINK("' + masterUrl + '","Open the shared tracker")');

  sh.getRange(3, 1, 1, COLS).setValues([HEADERS])
      .setFontWeight('bold').setFontColor('#FFFFFF').setBackground('#1F4E78');
  [10, 26, 52, 20, 17, 11, 7, 11, 40].forEach((w, j) => sh.setColumnWidth(j + 1, w * 7));
  sh.getRange(1, 1, last, COLS).setFontFamily('Arial').setFontSize(10);
  sh.getRange('B1').setFontSize(14);
  sh.setFrozenRows(3);

  // computed columns: Assigned to = this person, End = Friday of the last week
  sh.getRange(4, 4, n, 1).setFormula('=IF($C4="","",$B$1)').setFontColor('#595959').setFontStyle('italic');
  sh.getRange(4, 8, n, 1).setFormula('=IF(OR($F4="",$G4=""),"",$F4+7*$G4-3)').setFontColor('#595959').setFontStyle('italic');
  sh.getRange(4, 6, n, 1).setNumberFormat('d mmm yy');
  sh.getRange(4, 8, n, 1).setNumberFormat('d mmm yy');
  sh.getRange(4, 3, n, 1).setWrap(true);
  sh.getRange(4, 9, n, 1).setWrap(true);
  [sh.getRange(4, 4, n, 1), sh.getRange(4, 8, n, 1)].forEach(r =>
      r.protect().setDescription('Filled automatically').setWarningOnly(true));

  // dropdowns and checks
  sh.getRange(4, 2, n, 1).setDataValidation(SpreadsheetApp.newDataValidation().requireValueInList(GOALS, true).build());
  sh.getRange(4, 5, n, 1).setDataValidation(SpreadsheetApp.newDataValidation().requireValueInList(STATUSES, true).build());
  sh.getRange(4, 6, n, 1).setDataValidation(SpreadsheetApp.newDataValidation().requireDate()
      .setHelpText('Start date (a Monday)').build());
  sh.getRange(4, 7, n, 1).setDataValidation(SpreadsheetApp.newDataValidation().requireNumberBetween(1, 104)
      .setHelpText('Length in weeks').build());

  // colours: status, overdue end date
  const statusRange = sh.getRange(4, 5, n, 1);
  const rules = Object.keys(STATUS_FILL).map(s => SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo(s).setBackground(STATUS_FILL[s]).setRanges([statusRange]).build());
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=AND(ISNUMBER($H4),$H4<TODAY(),$E4<>"Complete")')
      .setFontColor('#C00000').setBold(true).setRanges([sh.getRange(4, 8, n, 1)]).build());
  sh.setConditionalFormatRules(rules);
  return file;
}
