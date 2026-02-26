#!/usr/bin/env python3
"""
Large Citation Graph Analysis

Reads DOIs from a CSV, fetches paper metadata from the Inciteful API,
builds a citation graph, and identifies the most connected external papers.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Required, TypedDict

logger = logging.getLogger(__name__)

BASE_URL = "https://graph.incitefulmed.com/openalex/paper"
MAX_URL_LENGTH = 4000  # conservative; avoids 414 URI Too Long

# Defaults (overridable via CLI)
DEFAULT_BATCH_SIZE = 100
DEFAULT_DELAY = 0.5
DEFAULT_TOP_N = 50
DEFAULT_INPUT = Path("dois.csv")
DEFAULT_IGNORE = Path("ignore_list.csv")
DEFAULT_POOL_SIZE = 1000
DEFAULT_OUTPUT = Path("results.json")

# HTTP status codes that are safe to retry
_RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------


class PaperDict(TypedDict, total=False):
    """Paper record as returned by the Inciteful API."""

    id: Required[str]  # noqa: A003  (shadows builtin, but matches API)
    doi: str | None
    title: str | None
    published_year: int | None
    journal: str | None
    num_citing: int
    num_cited_by: int
    citing: list[str]
    cited_by: list[str]


class MetaEntry(TypedDict):
    """Metadata stored for an external paper after the second API fetch.

    Note: ``year`` corresponds to ``published_year`` in the raw API response.
    """

    doi: str | None
    title: str | None
    year: int | None
    journal: str | None
    num_citing: int
    num_cited_by: int


MetaDict = dict[str, MetaEntry]

_EMPTY_META: MetaEntry = {
    "doi": None,
    "title": None,
    "year": None,
    "journal": None,
    "num_citing": 0,
    "num_cited_by": 0,
}


@dataclass
class GraphStats:
    """Summary statistics from building the citation graph."""

    in_list_ids: set[str] = field(default_factory=set)
    node_count: int = 0
    in_list_count: int = 0
    external_count: int = 0
    edge_count: int = 0


@dataclass
class AnalysisResult:
    """Results from analyzing external papers across all ranking modes."""

    cited_by_inlist: Counter[str] = field(default_factory=Counter)
    citing_inlist: Counter[str] = field(default_factory=Counter)
    combined: Counter[str] = field(default_factory=Counter)
    adamic_adar: dict[str, float] = field(default_factory=dict)
    salton_partial: dict[str, float] = field(default_factory=dict)


@dataclass
class RankedResults:
    """Final output of the analysis pipeline.

    Contains only what is needed for printing and saving — raw paper data
    is not retained.
    """

    graph: GraphStats = field(default_factory=GraphStats)
    analysis: AnalysisResult = field(default_factory=AnalysisResult)
    ext_meta: MetaDict = field(default_factory=dict)
    similarity: list[tuple[str, float]] = field(default_factory=list)
    adamic_adar_ranked: list[tuple[str, float]] = field(default_factory=list)
    salton_ranked: list[tuple[str, float]] = field(default_factory=list)
    top_n: int = DEFAULT_TOP_N
    total_dois: int = 0
    papers_fetched: int = 0
    ignored_count: int = 0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_dois(csv_path: Path) -> list[str]:
    """Read DOIs from the CSV file.

    The CSV must have a header row; DOIs are read from the first column.
    Returns an empty list if the file has only a header or is empty.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        if next(reader, None) is None:
            return []
        return [row[0].strip() for row in reader if row and row[0].strip()]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def fetch_batch(ids: list[str], *, max_retries: int = 3) -> list[PaperDict]:
    """Fetch paper data for a batch of DOIs/IDs from the Inciteful API.

    Automatically splits the request if the URL would exceed *MAX_URL_LENGTH*,
    and retries with exponential back-off on transient errors.  Non-retryable
    HTTP errors (4xx other than 429) are raised immediately.
    """
    params = "&".join(
        f"ids[]={urllib.parse.quote(paper_id, safe='')}" for paper_id in ids
    )
    url = f"{BASE_URL}?{params}"

    if len(url) > MAX_URL_LENGTH and len(ids) > 1:
        mid = len(ids) // 2
        left = fetch_batch(ids[:mid], max_retries=max_retries)
        right = fetch_batch(ids[mid:], max_retries=max_retries)
        return left + right

    for attempt in range(max_retries):
        req = urllib.request.Request(
            url, headers={"User-Agent": "LargeCitationGraphAnalysis/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result: list[PaperDict] = json.loads(resp.read())
                return result
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE_HTTP_CODES:
                raise  # 400/404/422 etc. — won't succeed on retry
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt
            logger.warning(
                "    Retry %d/%d after %ds: HTTP %d",
                attempt + 1,
                max_retries,
                wait,
                exc.code,
            )
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt
            logger.warning(
                "    Retry %d/%d after %ds: %s",
                attempt + 1,
                max_retries,
                wait,
                exc,
            )
            time.sleep(wait)

    return []  # unreachable, but keeps type-checkers happy


def fetch_all_papers(
    ids: list[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay: float = DEFAULT_DELAY,
) -> list[PaperDict]:
    """Fetch papers in batches, returning a list of paper dicts.

    Failed batches are logged and skipped; a summary warning is emitted at the
    end so callers know data may be incomplete.
    """
    all_papers: list[PaperDict] = []
    total_batches = max(1, (len(ids) + batch_size - 1) // batch_size)
    failed_ids = 0

    for i in range(0, len(ids), batch_size):
        batch_num = i // batch_size + 1
        batch = ids[i : i + batch_size]
        try:
            papers = fetch_batch(batch)
            all_papers.extend(papers)
            logger.info(
                "  Batch %d/%d: got %d papers (%d total)",
                batch_num,
                total_batches,
                len(papers),
                len(all_papers),
            )
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning(
                "  Batch %d/%d: FAILED - %s",
                batch_num,
                total_batches,
                exc,
            )
            failed_ids += len(batch)

        if batch_num < total_batches:
            time.sleep(delay)

    if failed_ids:
        logger.warning(
            "  WARNING: %d ID(s) from failed batches were skipped",
            failed_ids,
        )

    return all_papers


def fetch_external_metadata(
    paper_ids: set[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay: float = DEFAULT_DELAY,
) -> MetaDict:
    """Fetch metadata for external papers from the Inciteful API."""
    raw = fetch_all_papers(list(paper_ids), batch_size=batch_size, delay=delay)
    metadata: MetaDict = {}
    for p in raw:
        metadata[p["id"]] = {
            "doi": p.get("doi"),
            "title": p.get("title"),
            "year": p.get("published_year"),
            "journal": p.get("journal"),
            "num_citing": p.get("num_citing", 0),
            "num_cited_by": p.get("num_cited_by", 0),
        }
    return metadata


# ---------------------------------------------------------------------------
# Graph building & analysis
# ---------------------------------------------------------------------------


def build_graph(
    papers: Sequence[PaperDict],
    ignore_ids: set[str] | None = None,
) -> GraphStats:
    """Build a citation graph from fetched papers and return summary stats.

    Papers whose ID is in *ignore_ids* are excluded entirely — even if they
    appear in the input DOI list.
    """
    if ignore_ids is None:
        ignore_ids = set()

    in_list_ids: set[str] = set()
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()

    for paper in papers:
        pid: str = paper["id"]
        if pid in ignore_ids:
            continue
        in_list_ids.add(pid)
        seen_nodes.add(pid)

        for cited_id in paper.get("citing", []):
            if cited_id in ignore_ids:
                continue
            seen_edges.add((pid, cited_id))
            seen_nodes.add(cited_id)

        for citing_id in paper.get("cited_by", []):
            if citing_id in ignore_ids:
                continue
            seen_edges.add((citing_id, pid))
            seen_nodes.add(citing_id)

    return GraphStats(
        in_list_ids=in_list_ids,
        node_count=len(seen_nodes),
        in_list_count=len(in_list_ids),
        external_count=len(seen_nodes) - len(in_list_ids),
        edge_count=len(seen_edges),
    )


def analyze_external_papers(
    in_list_ids: set[str],
    papers: Sequence[PaperDict],
    ignore_ids: set[str] | None = None,
) -> AnalysisResult:
    """Analyze external papers by their connections to in-list papers.

    Computes five ranking metrics (mode (d) similarity requires external
    metadata and is computed separately via :func:`compute_similarity`):

    (a) **cited_by_inlist** — how many in-list papers cite this external paper.
    (b) **citing_inlist** — how many in-list papers this external paper cites.
    (c) **combined** — ``min(a, b)``; rewards bidirectional connections.
    (e) **adamic_adar** — like (b) but weighted by ``1 / ln(num_cited_by)`` of
        the in-list paper, so citing a niche paper contributes more than citing
        a highly-cited one.  Uses the *natural* logarithm (not base-2) with a
        floor of 2 to avoid division by zero.  Note: this is a *global* degree
        proxy, not a shared-neighbour Adamic/Adar — it measures how "niche"
        each in-list paper is in the broader literature.
    (f) **salton_partial** — weighted co-citation partial score.  For each
        in-list paper P that cites both an external paper E and other in-list
        papers M, E receives ``sum(1/sqrt(ncb(m)) for m in M)``.  The final
        score is ``partial / sqrt(ncb(E))``, computed after the metadata fetch
        via :func:`compute_salton`.  NOTE: this is an approximation of the
        Salton co-citation index, not the canonical symmetric pairwise formula.
        It measures how strongly an external paper is co-cited alongside niche
        in-list papers.
    """
    if ignore_ids is None:
        ignore_ids = set()
    exclude = in_list_ids | ignore_ids

    cited_by_inlist: Counter[str] = Counter()
    citing_inlist: Counter[str] = Counter()
    adamic_adar: defaultdict[str, float] = defaultdict(float)

    for paper in papers:
        if paper["id"] in ignore_ids:
            continue

        for cited_id in paper.get("citing", []):
            if cited_id not in exclude:
                cited_by_inlist[cited_id] += 1

        num_cited_by = paper.get("num_cited_by", 0)
        for citing_id in paper.get("cited_by", []):
            if citing_id not in exclude:
                citing_inlist[citing_id] += 1
                weight = 1.0 / math.log(max(num_cited_by, 2))
                adamic_adar[citing_id] += weight

    # (c) Combined — min of both directions
    combined: Counter[str] = Counter()
    for ext_id in cited_by_inlist.keys() | citing_inlist.keys():
        val = min(cited_by_inlist[ext_id], citing_inlist[ext_id])
        if val > 0:
            combined[ext_id] = val

    # (f) Salton Index — partial scores (before dividing by sqrt(ncb(e))).
    inlist_ncb = {
        p["id"]: p.get("num_cited_by", 0) for p in papers if p["id"] in in_list_ids
    }
    salton_partial: defaultdict[str, float] = defaultdict(float)

    for paper in papers:
        pid = paper["id"]
        if pid in ignore_ids:
            continue
        refs = set(paper.get("citing", []))

        inlist_refs = [
            r for r in refs if r in in_list_ids and r not in ignore_ids and r != pid
        ]
        if not inlist_refs:
            continue

        inlist_weight = sum(
            1.0 / math.sqrt(max(inlist_ncb.get(m_id, 1), 1)) for m_id in inlist_refs
        )

        for ref_id in refs:
            if ref_id not in exclude:
                salton_partial[ref_id] += inlist_weight

    return AnalysisResult(
        cited_by_inlist=cited_by_inlist,
        citing_inlist=citing_inlist,
        combined=combined,
        adamic_adar=dict(adamic_adar),
        salton_partial=dict(salton_partial),
    )


# ---------------------------------------------------------------------------
# Score finalization (pure functions, testable independently)
# ---------------------------------------------------------------------------


def compute_similarity(
    citing_inlist: Counter[str],
    ext_meta: MetaDict,
) -> list[tuple[str, float]]:
    """Mode (d): fraction of an external paper's citations that are in-list.

    ``similarity = citing_inlist[e] / num_citing(e)``, clamped to [0, 1].
    """
    result: list[tuple[str, float]] = []
    for ext_id, count in citing_inlist.items():
        info = ext_meta.get(ext_id, _EMPTY_META)
        num_citing = info.get("num_citing", 0)
        if num_citing > 0:
            result.append((ext_id, min(count / num_citing, 1.0)))
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def compute_salton(
    salton_partial: dict[str, float],
    ext_meta: MetaDict,
) -> list[tuple[str, float]]:
    """Mode (f): finalize Salton by dividing partial score by sqrt(num_cited_by).

    ``salton = partial / sqrt(ncb(e))``
    """
    result: list[tuple[str, float]] = []
    for ext_id, partial in salton_partial.items():
        info = ext_meta.get(ext_id, _EMPTY_META)
        ncb = info.get("num_cited_by", 0)
        if ncb > 0:
            result.append((ext_id, partial / math.sqrt(ncb)))
    result.sort(key=lambda x: x[1], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Ignore-list resolution
# ---------------------------------------------------------------------------


def _is_openalex_id(value: str) -> bool:
    """Return True if *value* looks like an OpenAlex work ID (e.g. W4285719527)."""
    return len(value) > 1 and value[0] == "W" and value[1:].isdigit()


def resolve_ignore_ids(
    ignore_path: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay: float = DEFAULT_DELAY,
) -> set[str]:
    """Read entries from the ignore-list CSV and resolve them to OpenAlex IDs.

    Each entry can be either an OpenAlex ID (e.g. W4285719527) or a DOI.
    OpenAlex IDs are used directly; DOIs are resolved via the API.
    """
    if not ignore_path.exists():
        return set()

    entries = read_dois(ignore_path)
    if not entries:
        return set()

    ignore_ids: set[str] = set()
    dois_to_resolve: list[str] = []
    for entry in entries:
        if _is_openalex_id(entry):
            ignore_ids.add(entry)
        else:
            dois_to_resolve.append(entry)

    logger.info("  Read %d entries from %s", len(entries), ignore_path)
    logger.info(
        "  %d OpenAlex IDs, %d DOIs to resolve",
        len(ignore_ids),
        len(dois_to_resolve),
    )

    if dois_to_resolve:
        papers = fetch_all_papers(dois_to_resolve, batch_size=batch_size, delay=delay)
        for p in papers:
            ignore_ids.add(p["id"])

    logger.info("  Total ignore-list IDs: %d", len(ignore_ids))
    return ignore_ids


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _doi_url(doi: str | None) -> str | None:
    """Format a DOI as a clickable URL, or return None."""
    return f"https://doi.org/{doi}" if doi else None


def _openalex_url(paper_id: str) -> str:
    """Format an OpenAlex work ID as an API URL."""
    return f"https://api.openalex.org/works/{paper_id}"


def _enrich(pid: str, ext_meta: MetaDict) -> dict[str, Any]:
    """Build the common metadata fields for a result entry."""
    info = ext_meta.get(pid, _EMPTY_META)
    return {
        "doi": info.get("doi"),
        "doi_url": _doi_url(info.get("doi")),
        "openalex_url": _openalex_url(pid),
        "title": info.get("title"),
        "year": info.get("year"),
        "journal": info.get("journal"),
    }


def print_top(
    label: str,
    ranked: Sequence[tuple[str, int | float]],
    meta: MetaDict,
    n: int = 50,
    *,
    count_source: Counter[str] | None = None,
    count_label: str = "Citing",
    total_label: str = "TotCit",
    total_field: str = "num_citing",
) -> None:
    """Print the top N entries from a ranking with DOI and title.

    When *count_source* is None, prints a simple Rank/Count/DOI/Title table
    (used for integer-count rankings like modes a, b, c).

    When *count_source* is provided, prints an extended table with
    Score/CountLabel/TotalLabel columns (used for float-score rankings like
    modes d, e, f).
    """
    logger.info("\n%s", "=" * 130)
    logger.info("  %s", label)
    logger.info("%s", "=" * 130)

    if count_source is None:
        # Simple integer-count format
        logger.info("  %-6s %-8s %-60s %s", "Rank", "Count", "DOI URL", "Title")
        logger.info("  %s %s %s %s", "-" * 6, "-" * 8, "-" * 60, "-" * 50)
        for rank, (paper_id, count) in enumerate(ranked[:n], 1):
            info = meta.get(paper_id, _EMPTY_META)
            doi = _doi_url(info.get("doi")) or "\u2014"
            title: str = info.get("title") or "\u2014"
            if len(title) > 55:
                title = title[:52] + "..."
            logger.info("  %-6d %-8d %-60s %s", rank, count, doi, title)
            logger.info("         %s", _openalex_url(paper_id))
    else:
        # Extended float-score format
        logger.info(
            "  %-6s %-8s %-8s %-8s %-52s %s",
            "Rank",
            "Score",
            count_label,
            total_label,
            "DOI URL",
            "Title",
        )
        logger.info(
            "  %s %s %s %s %s %s",
            "-" * 6,
            "-" * 8,
            "-" * 8,
            "-" * 8,
            "-" * 52,
            "-" * 42,
        )
        for rank, (paper_id, score) in enumerate(ranked[:n], 1):
            info = meta.get(paper_id, _EMPTY_META)
            doi = _doi_url(info.get("doi")) or "\u2014"
            title = info.get("title") or "\u2014"
            if len(title) > 47:
                title = title[:44] + "..."
            cit_count = count_source[paper_id]
            total_val = info.get(total_field, 0)
            logger.info(
                "  %-6d %-8.4f %-8d %-8d %-52s %s",
                rank,
                score,
                cit_count,
                total_val,
                doi,
                title,
            )
            logger.info("         %s", _openalex_url(paper_id))


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    """Argparse type for positive integers."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"{value} must be >= 1")
    return n


def _non_negative_float(value: str) -> float:
    """Argparse type for non-negative floats."""
    f = float(value)
    if f < 0:
        raise argparse.ArgumentTypeError(f"{value} must be >= 0")
    return f


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a citation graph from a list of DOIs and rank the most "
            "connected external papers."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the DOIs CSV file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for the JSON results file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--ignore",
        type=Path,
        default=DEFAULT_IGNORE,
        help=f"Path to an ignore-list CSV (default: {DEFAULT_IGNORE})",
    )
    parser.add_argument(
        "-n",
        "--top-n",
        type=_positive_int,
        default=DEFAULT_TOP_N,
        help=f"Number of top results to display/save (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of DOIs per API request (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--delay",
        type=_non_negative_float,
        default=DEFAULT_DELAY,
        help=f"Seconds to wait between API requests (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--pool-size",
        type=_positive_int,
        default=DEFAULT_POOL_SIZE,
        help=(
            f"Candidate pool size for similarity/Salton rankings "
            f"(default: {DEFAULT_POOL_SIZE})"
        ),
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> RankedResults:
    """Execute the full analysis pipeline and return ranked results."""
    csv_path: Path = args.input
    ignore_path: Path = args.ignore
    top_n: int = args.top_n
    batch_size: int = args.batch_size
    delay: float = args.delay
    pool_size: int = args.pool_size

    logger.info("=" * 70)
    logger.info("  Large Citation Graph Analysis")
    logger.info("=" * 70)

    # Step 1: Read DOIs
    logger.info("\n[1/6] Reading DOIs from %s...", csv_path)
    dois = read_dois(csv_path)
    logger.info("  Found %d DOIs", len(dois))

    # Step 2: Resolve ignore list
    logger.info("\n[2/6] Loading ignore list from %s...", ignore_path)
    ignore_ids = resolve_ignore_ids(ignore_path, batch_size=batch_size, delay=delay)

    # Step 3: Fetch papers
    logger.info(
        "\n[3/6] Fetching papers from Inciteful API (%d per batch)...",
        batch_size,
    )
    papers = fetch_all_papers(dois, batch_size=batch_size, delay=delay)
    logger.info(
        "\n  Successfully fetched %d papers out of %d DOIs",
        len(papers),
        len(dois),
    )
    missing = len(dois) - len(papers)
    if missing > 0:
        logger.info("  %d DOI(s) not returned by API", missing)

    # Step 4: Build graph
    logger.info("\n[4/6] Building citation graph...")
    graph = build_graph(papers, ignore_ids=ignore_ids)
    logger.info(
        "  Nodes: %s (%s in-list, %s external)",
        f"{graph.node_count:,}",
        f"{graph.in_list_count:,}",
        f"{graph.external_count:,}",
    )
    logger.info("  Edges: %s", f"{graph.edge_count:,}")
    if ignore_ids:
        logger.info("  Ignored: %d papers excluded from graph", len(ignore_ids))

    # Step 5: Analyze external papers
    logger.info("\n[5/6] Analyzing external papers...")
    analysis = analyze_external_papers(graph.in_list_ids, papers, ignore_ids=ignore_ids)
    logger.info(
        "  External papers cited by in-list papers: %s",
        f"{len(analysis.cited_by_inlist):,}",
    )
    logger.info(
        "  External papers citing in-list papers: %s",
        f"{len(analysis.citing_inlist):,}",
    )
    combined_nonzero = sum(1 for v in analysis.combined.values() if v > 0)
    logger.info(
        "  External papers with bidirectional links: %s",
        f"{combined_nonzero:,}",
    )

    # Pre-sort float rankings
    adamic_adar_ranked = sorted(
        analysis.adamic_adar.items(), key=lambda x: x[1], reverse=True
    )
    salton_partial_ranked = sorted(
        analysis.salton_partial.items(), key=lambda x: x[1], reverse=True
    )

    # Collect IDs for metadata fetch
    citing_top = analysis.citing_inlist.most_common(pool_size)
    top_ids = (
        {pid for pid, _ in analysis.cited_by_inlist.most_common(top_n)}
        | {pid for pid, _ in citing_top}
        | {pid for pid, _ in analysis.combined.most_common(top_n)}
        | {pid for pid, _ in adamic_adar_ranked[:top_n]}
        | {pid for pid, _ in salton_partial_ranked[:pool_size]}
    )

    # Step 6: Fetch metadata & compute final scores
    logger.info(
        "\n[6/6] Fetching metadata for %d external papers...",
        len(top_ids),
    )
    ext_meta = fetch_external_metadata(top_ids, batch_size=batch_size, delay=delay)
    resolved = sum(1 for v in ext_meta.values() if v.get("doi"))
    logger.info("  Resolved DOIs for %d/%d papers", resolved, len(top_ids))

    similarity = compute_similarity(analysis.citing_inlist, ext_meta)
    salton_ranked = compute_salton(analysis.salton_partial, ext_meta)

    return RankedResults(
        graph=graph,
        analysis=analysis,
        ext_meta=ext_meta,
        similarity=similarity,
        adamic_adar_ranked=adamic_adar_ranked,
        salton_ranked=salton_ranked,
        top_n=top_n,
        total_dois=len(dois),
        papers_fetched=len(papers),
        ignored_count=len(ignore_ids),
    )


def print_results(result: RankedResults) -> None:
    """Print all ranking tables to stdout."""
    top_n = result.top_n
    a = result.analysis
    meta = result.ext_meta

    print_top(
        f"(a) TOP {top_n}: Most Cited by Papers in the List",
        a.cited_by_inlist.most_common(top_n),
        meta,
        n=top_n,
    )
    print_top(
        f"(b) TOP {top_n}: Citing the Most Papers in the List",
        a.citing_inlist.most_common(top_n),
        meta,
        n=top_n,
    )
    print_top(
        f"(c) TOP {top_n}: Combined (min of Cited, Citing)",
        a.combined.most_common(top_n),
        meta,
        n=top_n,
    )
    print_top(
        f"(d) TOP {top_n}: Similarity (fraction of citations to in-list papers)",
        result.similarity,
        meta,
        n=top_n,
        count_source=a.citing_inlist,
    )
    print_top(
        f"(e) TOP {top_n}: Adamic/Adar (niche-weighted citations to in-list papers)",
        result.adamic_adar_ranked,
        meta,
        n=top_n,
        count_source=a.citing_inlist,
    )
    print_top(
        f"(f) TOP {top_n}: Salton Index (co-citation with in-list papers)",
        result.salton_ranked,
        meta,
        n=top_n,
        count_source=a.cited_by_inlist,
        count_label="CitedBy",
        total_label="NCitedBy",
        total_field="num_cited_by",
    )


def save_results(result: RankedResults, output_path: Path) -> None:
    """Serialize all rankings to a JSON file."""
    top_n = result.top_n
    a = result.analysis
    meta = result.ext_meta

    def _entry(pid: str, **extra: Any) -> dict[str, Any]:
        return {"id": pid, **extra, **_enrich(pid, meta)}

    def _meta_val(pid: str, field: str) -> int:
        return meta.get(pid, _EMPTY_META).get(field, 0)

    data = {
        "summary": {
            "total_dois_in_csv": result.total_dois,
            "papers_fetched": result.papers_fetched,
            "total_nodes": result.graph.node_count,
            "in_list_nodes": result.graph.in_list_count,
            "external_nodes": result.graph.external_count,
            "total_edges": result.graph.edge_count,
            "unique_external_cited_by_inlist": len(a.cited_by_inlist),
            "unique_external_citing_inlist": len(a.citing_inlist),
            "ignored_papers": result.ignored_count,
        },
        "top_cited_by_inlist": [
            _entry(pid, count=count)
            for pid, count in a.cited_by_inlist.most_common(top_n)
        ],
        "top_citing_inlist": [
            _entry(pid, count=count)
            for pid, count in a.citing_inlist.most_common(top_n)
        ],
        "top_combined": [
            _entry(
                pid,
                cited_by_inlist=a.cited_by_inlist[pid],
                citing_inlist=a.citing_inlist[pid],
                combined=count,
            )
            for pid, count in a.combined.most_common(top_n)
        ],
        "top_similarity": [
            _entry(
                pid,
                similarity=round(score, 6),
                citing_inlist=a.citing_inlist[pid],
                num_citing=_meta_val(pid, "num_citing"),
            )
            for pid, score in result.similarity[:top_n]
        ],
        "top_adamic_adar": [
            _entry(pid, adamic_adar=round(score, 6), citing_inlist=a.citing_inlist[pid])
            for pid, score in result.adamic_adar_ranked[:top_n]
        ],
        "top_salton": [
            _entry(
                pid,
                salton=round(score, 6),
                cited_by_inlist=a.cited_by_inlist[pid],
                num_cited_by=_meta_val(pid, "num_cited_by"),
            )
            for pid, score in result.salton_ranked[:top_n]
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("\n  Results saved to %s", output_path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    result = run_pipeline(args)
    print_results(result)
    save_results(result, args.output)
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
