# IEEE Conference Paper

IEEE conference paper based on the official IEEEtran template (version 6/27/2024).

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
