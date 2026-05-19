# Thesis LaTeX Template (v2)

LaTeX template for HSE FCS DSBA bachelor's thesis on
**"Community Detection with Graph Neural Networks via QUBO Formulation"**.

## Structure

```
thesis_v2/
├── main.tex                    # main file, compile this
├── title_page.tex              # HSE title page
├── references.bib              # bibliography (~30 entries pre-filled)
├── chapters/
│   ├── 00_abstract.tex         # ~300 words, DRAFT FILLED
│   ├── 01_introduction.tex     # 1.1 Motivation FILLED, 1.2-1.5 TODO
│   ├── 02_related_work.tex     # all subsections TODO
│   ├── 03_methodology.tex      # 3.1.1 FILLED, others TODO with hints
│   ├── 04_experiments.tex      # tables filled with real numbers,
│   │                            #   text TODO with bullet hints
│   ├── 05_conclusion.tex       # findings list FILLED, others TODO
│   └── A_annex.tex             # placeholder
└── figures/                    # put PNG files here
```

## Conventions

### En-dash style

This template uses **en-dash with spaces** (`~--~`) for parenthetical
breaks within sentences, not em-dash (`---`). Example:

```
This approach~--~unlike Louvain~--~yields...
```

The non-breaking spaces (`~`) prevent line breaks before/after the
dash. Use this consistently throughout.

### Math notation shortcuts

Defined in main.tex:

```latex
\B   --> bold B (modularity matrix)
\Q   --> bold Q (QUBO matrix)
\xb  --> bold x (binary vector)
\pb  --> bold p (probability vector)
\Pb  --> bold P (probability matrix)
\Lcal --> calligraphic L (loss)
\Ncal --> calligraphic N (neighborhood)
```

### TODO markers

Look for `\textit{[TODO: ...]}` placeholders. They are formatted in
italics so they stand out in the compiled PDF. Each has a short hint
about what to write there.

### Tables and figures

- All tables and figures must be referenced in text by `\ref{...}`.
- Captions: figures below image, tables above table.
- Already provided: `tab:datasets`, `tab:hypers`, `tab:k2_main`,
  `tab:formula_5_vs_6`.

## How to compile

In Overleaf: upload all files, set main.tex as compile target.

Locally:
```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Workflow recommendations

1. **Already drafted:** Abstract, 1.1 Motivation, 3.1.1 Newman's modularity.
2. **Methodology first** (chapter 3): math is partially done.
3. **Then Experiments** (chapter 4): numbers are in tables, write
   commentary around them.
4. **Then Related Work** (chapter 2).
5. **Then finish Introduction** (1.2--1.5).
6. **Then Discussion / Conclusion** (chapter 5).
7. **Abstract last**: re-check the draft once everything else is done.

## Word count targets

| Chapter              | Target pages |
|----------------------|--------------|
| Abstract             | 1            |
| Introduction         | 3-4          |
| Related Work         | 4-5          |
| Methodology          | 8-10         |
| Experiments          | 12-14        |
| Conclusion           | 3-4          |
| **Total**            | **34-41**    |
