# Audiometry analysis pipeline

Python code for processing and plotting serial pure-tone audiometry data. Developed
for a natural history study of hearing loss in Norrie disease and an accompanying
review of audiograms published in the literature.

## Contents

| File | Purpose |
|---|---|
| `audiometry.py` | Core pipeline: workbook ingestion, structural validation, and figure generation for serial audiometric data |
| `make_published_waterfall.py` | Pooled waterfall plot of audiograms digitised from published case reports |
| `make_published_grid2.py` | Per-case small-multiple grid (right/left ear paired) for published cases |
| `make_published_genotype_waterfall.py` | Published cases stratified by variant class (truncating / non-truncating / unknown) |
| `groups.example.json` | Format example for participant group membership |

## Figures produced by `audiometry.py`

- Per-participant audiogram waterfalls, with opacity encoding age at test
- Per-participant threshold-vs-age trajectories, coloured by frequency
- Small-multiple grids across the cohort (by ear, and with ears paired)
- Pooled cohort waterfall coloured by age at test
- Optional group-stratified pooled waterfalls (e.g. by genotype class)

Encodings are consistent across figures: marker shape distinguishes conventional
pure-tone audiometry from behavioural/play methods; a coloured marker edge flags
tests whose comments suggest possible conductive involvement (a keyword-derived
screening aid requiring manual adjudication, not an automated classification);
missing thresholds break the connecting line rather than being interpolated; and
no-response results are plotted with a distinct symbol rather than as a numeric
threshold.

## Note on case identifiers

Case labels appearing in the `make_published_*.py` scripts (for example `ND31`,
`ND44`) are registry identifiers from Smith et al. 2012 (*Am J Med Genet A*
158A:1909–1917) and refer to previously published cases. **They are unrelated to
participants in the natural history study**, which happens to use a similar
labelling convention.

## Data

No patient data is included in this repository.

`audiometry.py` expects an Excel workbook with one sheet per participant and a
two-row header: a merged ear row (`Right Ear` / `Left Ear`) above a detail row
giving `Date`, `Age at test`, `Type`, thresholds at 125 Hz–8 kHz for each ear, a
four-frequency pure-tone average, and `Comment`. The workbook is validated on load
and the script fails loudly if any sheet deviates from this layout.

Thresholds are numeric (dB HL). The string `No response` is recognised and plotted
at a fixed level with a distinct symbol; any other non-numeric value is treated as
missing.

Group membership for stratified plots is read from a local `groups.json`, which is
excluded from version control. See `groups.example.json` for the expected format.

## Requirements

```
pip install -r requirements.txt
```

Requires Python 3.9 or later with pandas, numpy, matplotlib, and openpyxl.

## Usage

Run the full pipeline from the command line:

```
python audiometry.py path/to/workbook.xlsx
```

This writes all figures to a `figs/` directory, exports the parsed data in long
format, and prints a per-participant summary. If a `groups.json` file is present,
group-stratified figures and a group composition summary are also produced.

Alternatively, import the module and call functions selectively:

```python
import audiometry as A

df = A.load_workbook("workbook.xlsx")
fig = A.figure_waterfall(df, "P01")
fig.savefig("waterfall_P01.png", dpi=150, bbox_inches="tight")
```

## Interpretation

Pooled and group-stratified figures are descriptive and exploratory. Repeated
measures within a participant are not independent, participants contribute unequal
numbers of tests, and groups may differ in age structure as well as in the variable
of interest. Apparent between-group differences in these figures are confounded with
those factors and should not be read as evidence of a group effect. Function
docstrings restate these caveats where relevant.

## Acknowledgement

Code was produced with AI assistance (Claude Opus 4.8, Anthropic). All analytical
decisions, data preparation, and interpretation were the author's own; generated
code was reviewed and verified by the author.

## Licence

MIT
