# SCOOPS — IEEE Conference Paper

IEEE conference paper for **SCOOPS** (Sensing, Communication, cOOperative Perception, and Scene reconstruction), a holistic multi-agent dataset for cooperative
perception, sensing, and communication. Based on the official IEEEtran
template (version 6/27/2024), structured to follow `paper_outline.docx`.

Placeholders are marked in red with `\todo{...}` in the PDF — search for
`\todo` in `main.tex`; none may remain at submission. Bibliography entries
were filled in from memory as starting points — **verify every entry**
against the actual publication.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | The paper — edit this |
| `references.bib` | BibTeX bibliography — add your references here |
| `IEEEtran.cls` | IEEE conference class file (do not edit) |
| `figures/` | Put all figures here (`\graphicspath` already points to it) |
| `template/IEEEconferencetemplate062824.tex` | Original template, kept for reference only |

## Building

With `latexmk` (recommended):

```bash
latexmk -pdf main.tex
```

Or manually:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

On Overleaf: upload this folder (or link the repo) and set `main.tex` as the root document.

## Template rules to remember

- No symbols, special characters, footnotes, or math in the title or abstract.
- Define abbreviations at first use, even if already defined in the abstract.
- Refer to equations as `\eqref{...}` and figures as `Fig.~\ref{...}` (even at sentence start).
- Figures/tables go at the top or bottom of columns, after their first mention; captions below figures, table titles above tables.
- Axis labels use words with units in parentheses, e.g. "Magnetization (kA/m)".
- Remove all remaining placeholder text before submission.
