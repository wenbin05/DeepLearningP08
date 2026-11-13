# Submitted survey results

The current response set is in [responses/](responses/): `p01.csv` through `p10.csv`.
Each file contains 20 ratings, identified by participant ID and summary ID (200 rows total).

## Contents

- `responses/`: latest submitted CSVs, preserved byte-for-byte.
- [manifest.json](manifest.json): upload provenance, original Git blob IDs and format-check results.
- [REVIEW_NOTES.md](REVIEW_NOTES.md): interpretation limitations retained with the results.
- [ANALYSIS.md](ANALYSIS.md) and [analysis.json](analysis.json): descriptive results and agreement.
- [summary_key.json](summary_key.json): summary-to-configuration mapping, released after rating collection.
- [analyze.py](analyze.py): standalone, standard-library reproduction script.

## Columns

| Column | Meaning |
|---|---|
| `participant_id` | Assigned anonymous ID, P01–P10. |
| `summary_id` | Blinded summary ID, S01–S20. Join by this ID, not row position. |
| `judgment` | `consistent`, `inconsistent`, or `uncertain`. |
| `evidence` | Submitted explanation; required for an `inconsistent` judgment. |

The packet design uses the same 20 summaries in different orders across response files.
These represent five source articles and four configurations. Do not count the 200
ratings as 200 independent summaries.

## Version and validation

This set was uploaded at commit `fd90ad3c6dc3e109f8dccea845f0b4c44abbb01f`.
The earlier files under `participant_csv_outputs/` are superseded. The root-level
CSV copies have been moved here. Previous versions remain in Git history.

Checks confirmed the expected columns, 10 files with 20 rows each, matching
participant IDs, one occurrence of every expected summary ID per file, valid
labels and nonempty evidence for inconsistent judgments. No ratings or explanation
wording were edited during organization.

The project team has confirmed that people independently read the articles and
compared them with the summaries to complete these ratings. This collection
statement is team-reported; file checks establish completeness, not independent
verification of the collection process. Results are analyzed as submitted, without
correction of judgments. Low agreement and rubric-interpretation issues are
reported in ANALYSIS.md and REVIEW_NOTES.md.

Keep these raw files unchanged. Save any participant clarifications or adjudicated
interpretations separately, with a clear record of what changed and why.

## Reproduce the descriptive analysis

```bash
python survey_results/analyze.py
```

Python 3 and its standard library are sufficient. The command reads the response
CSVs and summary key and regenerates analysis.json without modifying raw ratings.
