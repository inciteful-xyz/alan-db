#!/usr/bin/env python3
"""
ALAN Citation Graph Analysis

Reads DOIs from ALAN_DB.csv, fetches paper metadata from the Inciteful API,
builds a citation graph, and identifies the most connected external papers.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

BASE_URL = "https://graph.incitefulmed.com/openalex/paper"
BATCH_SIZE = 100
REQUEST_DELAY = 0.5  # seconds between API calls
CSV_PATH = Path(__file__).parent / "ALAN_DB.csv"
OUTPUT_PATH = Path(__file__).parent / "alan_citation_results.json"

# Type aliases
PaperDict = dict[str, Any]
NodeDict = dict[str, Any]
MetaDict = dict[str, dict[str, Any]]


def read_dois(csv_path: Path) -> list[str]:
    """Read DOIs from the CSV file."""
    dois: list[str] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row and row[0].strip():
                dois.append(row[0].strip())
    return dois


def fetch_batch(ids: list[str]) -> list[PaperDict]:
    """Fetch paper data for a batch of DOIs/IDs from the Inciteful API."""
    params = "&".join(f"ids[]={urllib.parse.quote(id_, safe='')}" for id_ in ids)
    url = f"{BASE_URL}?{params}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "ALANCitationAnalysis/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result: list[PaperDict] = json.loads(resp.read())
        return result


def fetch_all_papers(dois: list[str]) -> list[PaperDict]:
    """Fetch all papers in batches, returning a list of paper dicts."""
    all_papers: list[PaperDict] = []
    total_batches = (len(dois) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(dois), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = dois[i : i + BATCH_SIZE]
        try:
            papers = fetch_batch(batch)
            all_papers.extend(papers)
            print(
                f"  Batch {batch_num}/{total_batches}: got {len(papers)} papers "
                f"({len(all_papers)} total)"
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"  Batch {batch_num}/{total_batches}: FAILED - {e}")

        if batch_num < total_batches:
            time.sleep(REQUEST_DELAY)

    return all_papers


def build_graph(
    papers: list[PaperDict],
) -> tuple[set[str], dict[str, NodeDict], set[tuple[str, str]]]:
    """
    Build a citation graph from the fetched papers.

    Returns:
        in_list_ids: set of IDs for papers that were in our DOI list
        nodes: dict of paper_id -> metadata
        edges: set of (source_id, target_id) tuples (source cites target)
    """
    in_list_ids: set[str] = set()
    nodes: dict[str, NodeDict] = {}
    edges: set[tuple[str, str]] = set()

    for paper in papers:
        pid: str = paper["id"]
        in_list_ids.add(pid)
        nodes[pid] = {
            "id": pid,
            "doi": paper.get("doi"),
            "title": paper.get("title"),
            "year": paper.get("published_year"),
            "journal": paper.get("journal"),
            "num_citing": paper.get("num_citing", 0),
            "num_cited_by": paper.get("num_cited_by", 0),
            "in_list": True,
        }

        # Papers this one cites (outgoing references)
        for cited_id in paper.get("citing", []):
            edges.add((pid, cited_id))
            if cited_id not in nodes:
                nodes[cited_id] = {"id": cited_id, "in_list": False}

        # Papers that cite this one (incoming citations)
        for citing_id in paper.get("cited_by", []):
            edges.add((citing_id, pid))
            if citing_id not in nodes:
                nodes[citing_id] = {"id": citing_id, "in_list": False}

    return in_list_ids, nodes, edges


def analyze_external_papers(
    in_list_ids: set[str], papers: list[PaperDict]
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    """
    Analyze external papers (not in our DOI list) by their connections.

    Returns three Counters:
        cited_by_inlist: external_id -> count of in-list papers that cite it
        citing_inlist: external_id -> count of in-list papers it cites
        combined: external_id -> sum of both counts
    """
    # (a) How many in-list papers cite this external paper
    cited_by_inlist: Counter[str] = Counter()
    # (b) How many in-list papers does this external paper cite
    citing_inlist: Counter[str] = Counter()

    for paper in papers:
        for cited_id in paper.get("citing", []):
            if cited_id not in in_list_ids:
                cited_by_inlist[cited_id] += 1

        for citing_id in paper.get("cited_by", []):
            if citing_id not in in_list_ids:
                citing_inlist[citing_id] += 1

    # (c) Combined
    all_external = set(cited_by_inlist.keys()) | set(citing_inlist.keys())
    combined: Counter[str] = Counter()
    for ext_id in all_external:
        combined[ext_id] = cited_by_inlist[ext_id] + citing_inlist[ext_id]

    return cited_by_inlist, citing_inlist, combined


def fetch_external_metadata(paper_ids: set[str]) -> MetaDict:
    """Fetch metadata (DOI, title) for external papers from the Inciteful API."""
    metadata: MetaDict = {}
    ids_list = list(paper_ids)
    total_batches = (len(ids_list) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(ids_list), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = ids_list[i : i + BATCH_SIZE]
        try:
            papers = fetch_batch(batch)
            for p in papers:
                metadata[p["id"]] = {
                    "doi": p.get("doi"),
                    "title": p.get("title"),
                    "year": p.get("published_year"),
                    "journal": p.get("journal"),
                }
            print(f"  Batch {batch_num}/{total_batches}: resolved {len(papers)} papers")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"  Batch {batch_num}/{total_batches}: FAILED - {e}")

        if batch_num < total_batches:
            time.sleep(REQUEST_DELAY)

    return metadata


def print_top(label: str, counter: Counter[str], meta: MetaDict, n: int = 50) -> None:
    """Print the top N entries from a counter with DOI and title."""
    print(f"\n{'=' * 130}")
    print(f"  {label}")
    print(f"{'=' * 130}")
    print(f"  {'Rank':<6} {'Count':<8} {'DOI URL':<60} {'Title'}")
    print(f"  {'-'*6} {'-'*8} {'-'*60} {'-'*50}")
    for rank, (paper_id, count) in enumerate(counter.most_common(n), 1):
        info = meta.get(paper_id, {})
        raw_doi: str | None = info.get("doi")
        doi_url = f"https://doi.org/{raw_doi}" if raw_doi else "—"
        openalex_url = f"https://api.openalex.org/works/{paper_id}"
        title: str = info.get("title") or "—"
        if len(title) > 55:
            title = title[:52] + "..."
        print(f"  {rank:<6} {count:<8} {doi_url:<60} {title}")
        print(f"         {openalex_url}")


def main() -> None:
    print("=" * 70)
    print("  ALAN Citation Graph Analysis")
    print("=" * 70)

    # Step 1: Read DOIs
    print(f"\n[1/5] Reading DOIs from {CSV_PATH.name}...")
    dois = read_dois(CSV_PATH)
    print(f"  Found {len(dois)} DOIs")

    # Step 2: Fetch papers
    print(f"\n[2/5] Fetching papers from Inciteful API ({BATCH_SIZE} per batch)...")
    papers = fetch_all_papers(dois)
    print(f"\n  Successfully fetched {len(papers)} papers out of {len(dois)} DOIs")
    missing = len(dois) - len(papers)
    if missing > 0:
        fetched_dois = {p["doi"] for p in papers if p.get("doi")}
        missing_dois = [d for d in dois if d not in fetched_dois]
        print(f"  {missing} DOIs not found in API")
        if missing_dois[:5]:
            print(f"  First few missing: {missing_dois[:5]}")

    # Step 3: Build graph
    print("\n[3/5] Building citation graph...")
    in_list_ids, nodes, edges = build_graph(papers)
    external_count = sum(1 for n in nodes.values() if not n.get("in_list"))
    print(
        f"  Nodes: {len(nodes):,} ({len(in_list_ids):,} in-list, {external_count:,} external)"
    )
    print(f"  Edges: {len(edges):,}")

    # Step 4: Analyze external papers
    print("\n[4/5] Analyzing external papers...")
    cited_by_inlist, citing_inlist, combined = analyze_external_papers(
        in_list_ids, papers
    )
    print(f"  External papers cited by in-list papers: {len(cited_by_inlist):,}")
    print(f"  External papers citing in-list papers: {len(citing_inlist):,}")
    print(f"  Total unique external papers with connections: {len(combined):,}")

    # Collect all unique external IDs from top 50 of each ranking
    top_ids: set[str] = set()
    for pid, _ in cited_by_inlist.most_common(50):
        top_ids.add(pid)
    for pid, _ in citing_inlist.most_common(50):
        top_ids.add(pid)
    for pid, _ in combined.most_common(50):
        top_ids.add(pid)

    # Fetch metadata for top external papers
    print(f"\n[5/5] Fetching metadata for {len(top_ids)} top external papers...")
    ext_meta = fetch_external_metadata(top_ids)
    resolved = sum(1 for v in ext_meta.values() if v.get("doi"))
    print(f"  Resolved DOIs for {resolved}/{len(top_ids)} papers")

    # Print rankings
    print_top("(a) TOP 50: Most Cited by Papers in the List", cited_by_inlist, ext_meta)
    print_top("(b) TOP 50: Citing the Most Papers in the List", citing_inlist, ext_meta)
    print_top("(c) TOP 50: Combined (Cited + Citing)", combined, ext_meta)

    # Save results
    def enrich(pid: str) -> dict[str, Any]:
        info = ext_meta.get(pid, {})
        raw_doi: str | None = info.get("doi")
        return {
            "doi": raw_doi,
            "doi_url": f"https://doi.org/{raw_doi}" if raw_doi else None,
            "openalex_url": f"https://api.openalex.org/works/{pid}",
            "title": info.get("title"),
            "year": info.get("year"),
            "journal": info.get("journal"),
        }

    results = {
        "summary": {
            "total_dois_in_csv": len(dois),
            "papers_fetched": len(papers),
            "total_nodes": len(nodes),
            "in_list_nodes": len(in_list_ids),
            "external_nodes": external_count,
            "total_edges": len(edges),
            "unique_external_cited_by_inlist": len(cited_by_inlist),
            "unique_external_citing_inlist": len(citing_inlist),
        },
        "top_cited_by_inlist": [
            {"id": pid, "count": count, **enrich(pid)}
            for pid, count in cited_by_inlist.most_common(50)
        ],
        "top_citing_inlist": [
            {"id": pid, "count": count, **enrich(pid)}
            for pid, count in citing_inlist.most_common(50)
        ],
        "top_combined": [
            {
                "id": pid,
                "cited_by_inlist": cited_by_inlist[pid],
                "citing_inlist": citing_inlist[pid],
                "combined": count,
                **enrich(pid),
            }
            for pid, count in combined.most_common(50)
        ],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_PATH.name}")
    print("\nDone!")


if __name__ == "__main__":
    main()
