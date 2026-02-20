# ALAN Citation Graph Analysis

Fetches paper metadata for a list of DOIs from the [Inciteful API](https://api.inciteful.xyz), builds a citation graph, and identifies the most connected external papers (papers not in the original list but heavily cited by or citing papers in the list).

## Requirements

- Python 3.10+ (uses only the standard library, no `pip install` needed)
- `ALAN_DB.csv` in the same directory, with a single `DOI` column

## Usage

```bash
python3 alan_citation_analysis.py
```

Runtime is roughly 1-2 minutes (5,200+ DOIs across 53 API batches with a 0.5s delay between each).

## What it does

1. **Reads DOIs** from `ALAN_DB.csv`
2. **Fetches paper metadata** from the Inciteful API in batches of 100, collecting citation links (`citing` and `cited_by` arrays)
3. **Builds a citation graph** where nodes are paper IDs and edges are citation relationships. External papers (referenced but not in the CSV) get stub nodes.
4. **Analyzes external papers** by three rankings:
   - **(a) Most cited by in-list papers** — papers your list references the most (influential foundational works)
   - **(b) Citing the most in-list papers** — papers that reference many of your papers (review articles, meta-analyses)
   - **(c) Combined** — sum of (a) and (b)
5. **Fetches metadata** (DOI, title, year) for the top external papers via a second round of API calls
6. **Outputs results** to the console and saves them to `alan_citation_results.json`

## Output

`alan_citation_results.json` contains:

- `summary` — graph stats (node/edge counts, papers fetched vs missing)
- `top_cited_by_inlist` — top 50 external papers most cited by your list
- `top_citing_inlist` — top 50 external papers that cite the most papers in your list
- `top_combined` — top 50 by combined score

Each entry includes `id`, `doi`, `doi_url`, `openalex_url`, `title`, `year`, `journal`, and the relevant counts.

## Configuration

All tunable constants are at the top of `alan_citation_analysis.py`:

| Constant        | Default                      | Description                       |
| --------------- | ---------------------------- | --------------------------------- |
| `BATCH_SIZE`    | `100`                        | Number of DOIs per API request    |
| `REQUEST_DELAY` | `0.5`                        | Seconds to wait between API calls |
| `CSV_PATH`      | `ALAN_DB.csv`                | Path to the input CSV file        |
| `OUTPUT_PATH`   | `alan_citation_results.json` | Path to the output JSON file      |

### Common modifications

**Change the input file:** Update `CSV_PATH` to point to a different CSV. The file must have a header row and DOIs in the first column.

**Adjust the top-N ranking size:** The `most_common(50)` calls control how many results appear. Search for `most_common(50)` and change `50` to your desired number. The `n` parameter on `print_top` controls the console output separately.

**Change batch size:** If you hit URL length limits with very long DOIs, reduce `BATCH_SIZE`. If you want fewer API calls, increase it (up to ~200 is safe for typical DOI lengths).

**Add rate limiting:** Increase `REQUEST_DELAY` if you encounter 429 rate-limit errors.
