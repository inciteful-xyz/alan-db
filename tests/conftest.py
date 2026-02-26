"""Shared fixtures for citation_analysis tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from citation_analysis import MetaEntry, PaperDict


# ---------------------------------------------------------------------------
# A small, self-contained citation graph used by the unit tests.
#
#   In-list papers: P1, P2, P3
#   External papers: E1, E2, E3, E4
#   Ignored paper:   IGN
#
#   P1 cites:  E1, E2, P2, IGN
#   P2 cites:  E1, E3
#   P3 cites:  E2
#
#   E1 cites:  P1, P2       (cited_by of P1 and P2)
#   E2 cites:  P1, P3       (cited_by of P1 and P3)
#   E3 cites:  P2           (cited_by of P2)
#   E4 cites:  P1           (cited_by of P1)
#   IGN cites: P1           (cited_by of P1, but should be ignored)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_papers() -> list[PaperDict]:
    """Three in-list papers with known citation links."""
    return [
        {
            "id": "P1",
            "doi": "10.1000/p1",
            "title": "Paper One",
            "published_year": 2020,
            "journal": "Journal A",
            "num_citing": 4,
            "num_cited_by": 50,
            "citing": ["E1", "E2", "P2", "IGN"],
            "cited_by": ["E1", "E2", "E4", "IGN"],
        },
        {
            "id": "P2",
            "doi": "10.1000/p2",
            "title": "Paper Two",
            "published_year": 2021,
            "journal": "Journal B",
            "num_citing": 2,
            "num_cited_by": 30,
            "citing": ["E1", "E3"],
            "cited_by": ["E1", "E3"],
        },
        {
            "id": "P3",
            "doi": "10.1000/p3",
            "title": "Paper Three",
            "published_year": 2022,
            "journal": "Journal C",
            "num_citing": 1,
            "num_cited_by": 10,
            "citing": ["E2"],
            "cited_by": ["E2"],
        },
    ]


@pytest.fixture()
def ignore_ids() -> set[str]:
    return {"IGN"}


# ---------------------------------------------------------------------------
# Well-known DOIs for integration tests.  These are highly-cited papers
# unlikely to vanish from OpenAlex / Inciteful.
# ---------------------------------------------------------------------------

KNOWN_DOIS = [
    "10.1016/s0140-6736(20)30183-5",  # Huang et al., Lancet 2020 (COVID)
    "10.1056/nejmoa2001017",           # Li et al., NEJM 2020
]

KNOWN_OPENALEX_ID = "W3005144120"  # one of the above, stable


@pytest.fixture()
def known_dois() -> list[str]:
    return list(KNOWN_DOIS)


@pytest.fixture()
def known_openalex_id() -> str:
    return KNOWN_OPENALEX_ID


def make_meta(
    *,
    doi: str | None = None,
    title: str | None = None,
    year: int | None = None,
    journal: str | None = None,
    num_citing: int = 0,
    num_cited_by: int = 0,
) -> MetaEntry:
    """Build a MetaEntry with sensible defaults for tests."""
    return {
        "doi": doi, "title": title, "year": year, "journal": journal,
        "num_citing": num_citing, "num_cited_by": num_cited_by,
    }
