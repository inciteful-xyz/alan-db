# Citation Graph Analysis

Fetches paper metadata for a list of DOIs from the Inciteful API, builds a citation graph, and identifies the most connected external papers — papers not in your original list but heavily cited by or citing papers in it.

## Requirements

- Python 3.10+ (uses only the standard library, no `pip install` needed)
- A CSV file with a `DOI` column (one DOI per row)

## Quick start

```bash
# Run with defaults (reads dois.csv, writes results.json)
python3 citation_analysis.py

# Customize inputs and output
python3 citation_analysis.py -i my_papers.csv -o my_results.json -n 100
```

Run `python3 citation_analysis.py --help` for all options.

## What it does

1. **Reads DOIs** from the input CSV
2. **Loads the ignore list** — DOIs or OpenAlex IDs of papers to exclude (previously reviewed and rejected)
3. **Fetches paper metadata** from the Inciteful API in batches, collecting citation links (`citing` and `cited_by` arrays)
4. **Builds a citation graph** where nodes are paper IDs and edges are citation relationships. External papers (referenced but not in the CSV) get stub nodes. Ignored papers are excluded entirely.
5. **Analyzes external papers** by six rankings:
   - **(a) Most cited by in-list papers** — papers your list references the most (influential foundational works)
   - **(b) Citing the most in-list papers** — papers that reference many of your papers (review articles, meta-analyses)
   - **(c) Combined** — min of (a) and (b), rewarding bidirectional connections
   - **(d) Similarity** — fraction of an external paper's outgoing citations that point to in-list papers; highlights papers focused on your topic
   - **(e) Adamic/Adar** — like (b) but weighted: citing a niche in-list paper (few total citations) contributes more than citing a popular one; surfaces papers that share a specific sub-topic
   - **(f) Salton Index** — co-citation metric: when an in-list paper's reference list includes both an external paper and other in-list papers, those are co-cited; normalized by `sqrt(num_cited_by)` of both papers to handle citation count disparities
6. **Fetches metadata** (DOI, title, year, journal) for the top external papers via a second round of API calls
7. **Outputs results** to the console and saves them to a JSON file

## Output

The JSON results file contains:

- `summary` — graph stats (node/edge counts, papers fetched vs. missing, ignored count)
- `top_cited_by_inlist` — top N external papers most cited by your list
- `top_citing_inlist` — top N external papers that cite the most papers in your list
- `top_combined` — top N by combined (min) score
- `top_similarity` — top N by citation fraction
- `top_adamic_adar` — top N by Adamic/Adar score
- `top_salton` — top N by Salton Index

Each entry includes `id`, `doi`, `doi_url`, `openalex_url`, `title`, `year`, `journal`, and the relevant scores/counts.

## CLI options

| Flag           | Short | Default           | Description                                           |
| -------------- | ----- | ----------------- | ----------------------------------------------------- |
| `--input`      | `-i`  | `dois.csv`        | Path to the input DOIs CSV file                       |
| `--output`     | `-o`  | `results.json`    | Path for the JSON results file                        |
| `--ignore`     |       | `ignore_list.csv` | Path to the ignore-list CSV (DOIs or OpenAlex IDs)    |
| `--top-n`      | `-n`  | `50`              | Number of top results to display and save per ranking |
| `--batch-size` |       | `100`             | Number of DOIs per API request                        |
| `--delay`      |       | `0.5`             | Seconds to wait between API requests                  |
| `--pool-size`  |       | `1000`            | Candidate pool size for similarity/Salton rankings    |

## Input files

**DOIs CSV** — must have a header row with DOIs in the first column. Example:

```
DOI
10.1001/jama.288.7.841
10.1038/s41586-020-2649-2
```

**Ignore list CSV** — same format, but entries can be either DOIs or OpenAlex IDs (e.g. `W4285719527`). Papers on this list are excluded from the graph and all rankings. Use this to skip papers you've already reviewed and determined are not relevant.

## Tips

- **First run:** Start with defaults to see what the data looks like, then adjust `--top-n` or `--pool-size` as needed.
- **Rate limits:** If you encounter 429 errors, increase `--delay`.
- **URL length limits:** If you have very long DOIs, reduce `--batch-size`.
- **Re-running monthly:** Add rejected papers to the ignore list so they don't reappear in future results.
