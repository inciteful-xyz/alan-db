"""Comprehensive tests for citation_analysis.py.

Unit tests use synthetic data from conftest fixtures.
Integration tests (marked with @pytest.mark.integration) call the live
Inciteful API — run them with:  pytest -m integration
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
from pathlib import Path
from unittest.mock import patch

import pytest

import citation_analysis as ca
from conftest import make_meta


# ===================================================================
#  Unit tests — pure functions, no network
# ===================================================================


class TestReadDois:
    """Tests for read_dois()."""

    def test_reads_dois_from_csv(self, tmp_path: Path) -> None:
        csv = tmp_path / "dois.csv"
        csv.write_text("DOI\n10.1000/a\n10.1000/b\n10.1000/c\n")
        result = ca.read_dois(csv)
        assert result == ["10.1000/a", "10.1000/b", "10.1000/c"]

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        csv = tmp_path / "dois.csv"
        csv.write_text("DOI\n  10.1000/a  \n")
        assert ca.read_dois(csv) == ["10.1000/a"]

    def test_skips_blank_rows(self, tmp_path: Path) -> None:
        csv = tmp_path / "dois.csv"
        csv.write_text("DOI\n10.1000/a\n\n  \n10.1000/b\n")
        assert ca.read_dois(csv) == ["10.1000/a", "10.1000/b"]

    def test_empty_csv_returns_empty(self, tmp_path: Path) -> None:
        csv = tmp_path / "dois.csv"
        csv.write_text("DOI\n")
        assert ca.read_dois(csv) == []

    def test_completely_empty_file(self, tmp_path: Path) -> None:
        """A file with zero bytes should not crash."""
        csv = tmp_path / "dois.csv"
        csv.write_text("")
        assert ca.read_dois(csv) == []

    def test_bom_header(self, tmp_path: Path) -> None:
        """UTF-8 BOM should not corrupt the first DOI."""
        csv = tmp_path / "dois.csv"
        csv.write_bytes(b"\xef\xbb\xbfDOI\n10.1000/a\n")
        assert ca.read_dois(csv) == ["10.1000/a"]


class TestIsOpenalexId:
    """Tests for _is_openalex_id()."""

    def test_valid_ids(self) -> None:
        assert ca._is_openalex_id("W4285719527") is True
        assert ca._is_openalex_id("W1") is True

    def test_invalid_ids(self) -> None:
        assert ca._is_openalex_id("10.1000/abc") is False
        assert ca._is_openalex_id("W") is False
        assert ca._is_openalex_id("") is False
        assert ca._is_openalex_id("w123") is False  # lowercase
        assert ca._is_openalex_id("WABC") is False  # non-digit after W


class TestParseArgs:
    """Tests for parse_args()."""

    def test_defaults(self) -> None:
        args = ca.parse_args([])
        assert args.top_n == 50
        assert args.batch_size == 100
        assert args.delay == 0.5
        assert args.pool_size == 1000

    def test_custom_values(self) -> None:
        args = ca.parse_args([
            "-i", "my.csv",
            "-o", "out.json",
            "-n", "10",
            "--batch-size", "50",
            "--delay", "1.0",
            "--pool-size", "500",
            "--ignore", "ign.csv",
        ])
        assert args.input == Path("my.csv")
        assert args.output == Path("out.json")
        assert args.top_n == 10
        assert args.batch_size == 50
        assert args.delay == 1.0
        assert args.pool_size == 500
        assert args.ignore == Path("ign.csv")

    def test_rejects_invalid_batch_size(self) -> None:
        with pytest.raises(SystemExit):
            ca.parse_args(["--batch-size", "0"])

    def test_rejects_negative_delay(self) -> None:
        with pytest.raises(SystemExit):
            ca.parse_args(["--delay", "-1"])


# -------------------------------------------------------------------
#  build_graph
# -------------------------------------------------------------------


class TestBuildGraph:
    """Tests for build_graph()."""

    def test_in_list_ids(self, sample_papers: list) -> None:
        graph = ca.build_graph(sample_papers)
        assert graph.in_list_ids == {"P1", "P2", "P3"}

    def test_counts(self, sample_papers: list) -> None:
        graph = ca.build_graph(sample_papers)
        assert graph.in_list_count == 3
        # External: E1, E2, E3, E4, IGN = 5
        assert graph.external_count == 5
        assert graph.node_count == 8  # 3 + 5

    def test_edge_count(self, sample_papers: list) -> None:
        graph = ca.build_graph(sample_papers)
        # citing edges (paper -> cited):
        #   P1->E1, P1->E2, P1->P2, P1->IGN, P2->E1, P2->E3, P3->E2 = 7
        # cited_by edges (citing_paper -> paper):
        #   E1->P1, E1->P2, E2->P1, E2->P3, E3->P2, E4->P1, IGN->P1 = 7
        # All are unique directed tuples = 14
        assert graph.edge_count == 14

    def test_ignore_ids_excludes_from_graph(
        self, sample_papers: list, ignore_ids: set
    ) -> None:
        graph = ca.build_graph(sample_papers, ignore_ids=ignore_ids)
        assert "IGN" not in graph.in_list_ids
        # One fewer external node (IGN removed)
        graph_no_ign = ca.build_graph(sample_papers)
        assert graph.node_count < graph_no_ign.node_count

    def test_ignore_skips_in_list_papers(self) -> None:
        """An in-list paper in the ignore set should be excluded entirely."""
        papers = [
            {"id": "A", "num_cited_by": 10, "citing": ["X"], "cited_by": []},
            {"id": "B", "num_cited_by": 5, "citing": [], "cited_by": []},
        ]
        graph = ca.build_graph(papers, ignore_ids={"A"})
        assert "A" not in graph.in_list_ids
        assert graph.in_list_count == 1


# -------------------------------------------------------------------
#  analyze_external_papers
# -------------------------------------------------------------------


class TestAnalyzeExternalPapers:
    """Tests for analyze_external_papers() — modes a, b, c, e, f."""

    def _run(self, papers: list, ignore: set | None = None) -> ca.AnalysisResult:
        in_list = {p["id"] for p in papers}
        return ca.analyze_external_papers(in_list, papers, ignore_ids=ignore)

    # -- mode (a): cited_by_inlist --

    def test_cited_by_inlist_counts(self, sample_papers: list) -> None:
        result = self._run(sample_papers)
        # E1 appears in P1.citing and P2.citing -> 2
        assert result.cited_by_inlist["E1"] == 2
        # E2 appears in P1.citing and P3.citing -> 2
        assert result.cited_by_inlist["E2"] == 2
        # E3 appears only in P2.citing -> 1
        assert result.cited_by_inlist["E3"] == 1

    def test_cited_by_excludes_in_list(self, sample_papers: list) -> None:
        """P2 is in P1.citing but should not count as external."""
        result = self._run(sample_papers)
        assert "P2" not in result.cited_by_inlist

    # -- mode (b): citing_inlist --

    def test_citing_inlist_counts(self, sample_papers: list) -> None:
        result = self._run(sample_papers)
        # E1 in P1.cited_by and P2.cited_by -> cites 2 in-list papers
        assert result.citing_inlist["E1"] == 2
        # E2 in P1.cited_by and P3.cited_by -> 2
        assert result.citing_inlist["E2"] == 2
        # E4 in P1.cited_by only -> 1
        assert result.citing_inlist["E4"] == 1

    # -- mode (c): combined = min(a, b) --

    def test_combined_is_min(self, sample_papers: list) -> None:
        result = self._run(sample_papers)
        for ext_id in result.combined:
            assert result.combined[ext_id] == min(
                result.cited_by_inlist[ext_id],
                result.citing_inlist[ext_id],
            )

    def test_combined_excludes_zero(self, sample_papers: list) -> None:
        """E4 has citing_inlist=1 but cited_by_inlist=0 -> not in combined."""
        result = self._run(sample_papers)
        assert "E4" not in result.combined

    # -- mode (e): Adamic/Adar --

    def test_adamic_adar_uses_log_weighting(self, sample_papers: list) -> None:
        result = self._run(sample_papers)
        # E1 cites P1 (ncb=50) and P2 (ncb=30)
        # score = 1/log(50) + 1/log(30)
        expected = 1.0 / math.log(50) + 1.0 / math.log(30)
        assert result.adamic_adar["E1"] == pytest.approx(expected)

    def test_adamic_adar_floors_at_log2(self) -> None:
        """Papers with num_cited_by <= 1 use log(2) as the floor."""
        papers = [
            {
                "id": "X1",
                "num_cited_by": 0,
                "citing": [],
                "cited_by": ["EXT"],
            }
        ]
        result = ca.analyze_external_papers({"X1"}, papers)
        assert result.adamic_adar["EXT"] == pytest.approx(1.0 / math.log(2))

    def test_adamic_adar_niche_beats_popular(self) -> None:
        """External paper citing a niche in-list paper scores higher
        per-citation than one citing a popular in-list paper."""
        niche = [
            {
                "id": "NICHE",
                "num_cited_by": 5,
                "citing": [],
                "cited_by": ["EXT_A"],
            }
        ]
        popular = [
            {
                "id": "POP",
                "num_cited_by": 5000,
                "citing": [],
                "cited_by": ["EXT_B"],
            }
        ]
        r_n = ca.analyze_external_papers({"NICHE"}, niche)
        r_p = ca.analyze_external_papers({"POP"}, popular)
        assert r_n.adamic_adar["EXT_A"] > r_p.adamic_adar["EXT_B"]

    # -- mode (f): Salton partial --

    def test_salton_partial_basic(self, sample_papers: list) -> None:
        """E1 is co-cited with P2 by P1 (P1.citing contains E1 and P2)."""
        result = self._run(sample_papers)
        assert "E1" in result.salton_partial
        assert result.salton_partial["E1"] > 0

    def test_salton_partial_exact_value(self, sample_papers: list) -> None:
        """Verify the exact Salton partial score for E1.

        P1 cites E1, E2, P2, IGN.  In-list refs (excl P1 itself) = [P2].
        (IGN is not in in_list_ids so it doesn't count as an in-list ref.)
        inlist_weight for P1 = 1/sqrt(30) (P2 has ncb=30).
        P2 cites E1, E3.  In-list refs (excl P2 itself) = [] -> no contribution.
        So E1's partial = 1/sqrt(30).
        """
        result = self._run(sample_papers)
        expected = 1.0 / math.sqrt(30)
        assert result.salton_partial["E1"] == pytest.approx(expected)

    def test_salton_partial_excludes_self_cocitation(
        self, sample_papers: list
    ) -> None:
        """P1 should not be co-cited with itself."""
        result = self._run(sample_papers)
        assert "P1" not in result.salton_partial

    def test_salton_partial_zero_when_no_inlist_corefs(self) -> None:
        """If a paper's citing list has no other in-list papers,
        its external refs get no Salton score."""
        papers = [
            {
                "id": "SOLO",
                "num_cited_by": 10,
                "citing": ["EXT1"],
                "cited_by": [],
            }
        ]
        result = ca.analyze_external_papers({"SOLO"}, papers)
        assert result.salton_partial.get("EXT1", 0) == 0

    def test_salton_ignores_ignored_inlist_refs(self) -> None:
        """In-list papers on the ignore list should not contribute to Salton weights."""
        papers = [
            {
                "id": "A",
                "num_cited_by": 100,
                "citing": ["EXT", "B"],
                "cited_by": [],
            },
            {
                "id": "B",
                "num_cited_by": 10,
                "citing": [],
                "cited_by": [],
            },
        ]
        # Without ignore: A cites EXT and B (in-list), so EXT gets 1/sqrt(10)
        r1 = ca.analyze_external_papers({"A", "B"}, papers)
        assert r1.salton_partial["EXT"] == pytest.approx(1.0 / math.sqrt(10))

        # With B ignored: A cites EXT and B, but B is ignored => no in-list refs
        r2 = ca.analyze_external_papers({"A", "B"}, papers, ignore_ids={"B"})
        assert r2.salton_partial.get("EXT", 0) == 0

    # -- ignore list filtering --

    def test_ignore_excludes_from_all_counters(
        self, sample_papers: list, ignore_ids: set
    ) -> None:
        result = self._run(sample_papers, ignore=ignore_ids)
        assert "IGN" not in result.cited_by_inlist
        assert "IGN" not in result.citing_inlist
        assert "IGN" not in result.combined
        assert "IGN" not in result.adamic_adar
        assert "IGN" not in result.salton_partial

    def test_ignore_skips_in_list_paper_contributions(self) -> None:
        """An in-list paper on the ignore list should not contribute."""
        papers = [
            {
                "id": "A",
                "num_cited_by": 10,
                "citing": ["EXT"],
                "cited_by": ["EXT"],
            },
            {
                "id": "B",
                "num_cited_by": 5,
                "citing": ["EXT"],
                "cited_by": [],
            },
        ]
        result = ca.analyze_external_papers({"A", "B"}, papers, ignore_ids={"A"})
        # Only B contributes; A is ignored
        assert result.cited_by_inlist["EXT"] == 1
        assert result.citing_inlist.get("EXT", 0) == 0  # B has no cited_by with EXT


# -------------------------------------------------------------------
#  Score finalization
# -------------------------------------------------------------------


class TestComputeSimilarity:
    """Tests for compute_similarity()."""

    def test_basic(self) -> None:
        citing: ca.Counter[str] = ca.Counter({"E1": 3, "E2": 1})
        meta: ca.MetaDict = {
            "E1": make_meta(num_citing=10, num_cited_by=5),
            "E2": make_meta(num_citing=4, num_cited_by=2),
        }
        result = ca.compute_similarity(citing, meta)
        scores = dict(result)
        assert scores["E1"] == pytest.approx(3 / 10)
        assert scores["E2"] == pytest.approx(1 / 4)

    def test_clamps_to_one(self) -> None:
        citing: ca.Counter[str] = ca.Counter({"E1": 5})
        meta: ca.MetaDict = {"E1": make_meta(num_citing=2)}
        result = ca.compute_similarity(citing, meta)
        assert result[0][1] == 1.0

    def test_skips_zero_citing(self) -> None:
        citing: ca.Counter[str] = ca.Counter({"E1": 3})
        meta: ca.MetaDict = {"E1": make_meta()}
        result = ca.compute_similarity(citing, meta)
        assert len(result) == 0


class TestComputeSalton:
    """Tests for compute_salton()."""

    def test_basic(self) -> None:
        partial = {"E1": 2.0}
        meta: ca.MetaDict = {"E1": make_meta(num_citing=10, num_cited_by=16)}
        result = ca.compute_salton(partial, meta)
        assert result[0] == ("E1", pytest.approx(2.0 / math.sqrt(16)))

    def test_skips_zero_ncb(self) -> None:
        partial = {"E1": 2.0}
        meta: ca.MetaDict = {"E1": make_meta(num_citing=10)}
        result = ca.compute_salton(partial, meta)
        assert len(result) == 0


# -------------------------------------------------------------------
#  Dataclasses
# -------------------------------------------------------------------


class TestDataclasses:
    """Tests for GraphStats, AnalysisResult, RankedResults."""

    def test_graph_stats_defaults(self) -> None:
        gs = ca.GraphStats()
        assert gs.in_list_ids == set()
        assert gs.node_count == 0

    def test_analysis_result_defaults(self) -> None:
        ar = ca.AnalysisResult()
        assert len(ar.cited_by_inlist) == 0
        assert len(ar.adamic_adar) == 0

    def test_ranked_results_defaults(self) -> None:
        rr = ca.RankedResults()
        assert rr.total_dois == 0
        assert rr.top_n == ca.DEFAULT_TOP_N


# -------------------------------------------------------------------
#  Error handling
# -------------------------------------------------------------------


class TestFetchBatchRetry:
    """Tests for fetch_batch retry and URL-length logic."""

    def test_retry_on_transient_error(self) -> None:
        """fetch_batch retries on URLError and succeeds on second attempt."""
        call_count = 0

        def mock_urlopen(_req, timeout=60):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise urllib.error.URLError("transient")

            class FakeResp:
                def read(self):
                    return b'[{"id": "W1"}]'
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass

            return FakeResp()

        with patch("citation_analysis.urllib.request.urlopen", mock_urlopen):
            with patch("citation_analysis.time.sleep"):
                result = ca.fetch_batch(["10.1000/test"], max_retries=3)
        assert len(result) == 1
        assert call_count == 2

    def test_raises_after_max_retries(self) -> None:
        """fetch_batch raises after exhausting retries."""

        def mock_urlopen(_req, timeout=60):
            raise urllib.error.URLError("permanent")

        with patch("citation_analysis.urllib.request.urlopen", mock_urlopen):
            with patch("citation_analysis.time.sleep"):
                with pytest.raises(urllib.error.URLError):
                    ca.fetch_batch(["10.1000/test"], max_retries=2)

    def test_does_not_retry_client_error(self) -> None:
        """fetch_batch raises immediately on 404 without retrying."""
        call_count = 0

        def mock_urlopen(_req, timeout=60):
            nonlocal call_count
            call_count += 1
            raise urllib.error.HTTPError(
                "http://example.com", 404, "Not Found", {}, None
            )

        with patch("citation_analysis.urllib.request.urlopen", mock_urlopen):
            with patch("citation_analysis.time.sleep"):
                with pytest.raises(urllib.error.HTTPError):
                    ca.fetch_batch(["bad_doi"], max_retries=3)
        assert call_count == 1  # no retries

    def test_retries_on_429(self) -> None:
        """fetch_batch retries on 429 Too Many Requests."""
        call_count = 0

        def mock_urlopen(_req, timeout=60):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise urllib.error.HTTPError(
                    "http://example.com", 429, "Too Many Requests", {}, None
                )

            class FakeResp:
                def read(self):
                    return b'[{"id": "W1"}]'
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass

            return FakeResp()

        with patch("citation_analysis.urllib.request.urlopen", mock_urlopen):
            with patch("citation_analysis.time.sleep"):
                result = ca.fetch_batch(["10.1000/test"], max_retries=3)
        assert len(result) == 1
        assert call_count == 3

    def test_retries_on_json_decode_error(self) -> None:
        """fetch_batch retries on malformed JSON response."""
        call_count = 0

        def mock_urlopen(_req, timeout=60):
            nonlocal call_count
            call_count += 1

            class FakeResp:
                def read(self):
                    if call_count == 1:
                        return b"<html>Server Error</html>"
                    return b'[{"id": "W1"}]'
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass

            return FakeResp()

        with patch("citation_analysis.urllib.request.urlopen", mock_urlopen):
            with patch("citation_analysis.time.sleep"):
                result = ca.fetch_batch(["10.1000/test"], max_retries=3)
        assert len(result) == 1
        assert call_count == 2

    def test_url_length_guard_splits_batch(self) -> None:
        """Very long IDs cause the URL to exceed MAX_URL_LENGTH, triggering a split."""
        long_id = "10.1000/" + "x" * 500
        ids = [long_id] * 20

        # Verify the test setup: constructed URL exceeds the limit
        params = "&".join(
            f"ids[]={urllib.parse.quote(pid, safe='')}" for pid in ids
        )
        url = f"{ca.BASE_URL}?{params}"
        assert len(url) > ca.MAX_URL_LENGTH

        # Track calls — splits mean more urlopen calls
        call_count = 0

        def mock_urlopen(_req, timeout=60):
            nonlocal call_count
            call_count += 1

            class FakeResp:
                def read(self_inner):
                    return json.dumps([{"id": f"W{call_count}"}]).encode()
                def __enter__(self_inner):
                    return self_inner
                def __exit__(self_inner, *args):
                    pass

            return FakeResp()

        with patch("citation_analysis.urllib.request.urlopen", mock_urlopen):
            result = ca.fetch_batch(ids)

        assert call_count > 1
        # Each sub-batch returns one unique paper; verify all are present
        result_ids = {p["id"] for p in result}
        assert len(result_ids) == call_count


# -------------------------------------------------------------------
#  _enrich and save_results
# -------------------------------------------------------------------


class TestEnrich:
    """Tests for _enrich() helper."""

    def test_with_doi(self) -> None:
        meta: ca.MetaDict = {
            "W1": make_meta(doi="10.1000/x", title="T", year=2020,
                            journal="J", num_citing=5, num_cited_by=10),
        }
        result = ca._enrich("W1", meta)
        assert result["doi_url"] == "https://doi.org/10.1000/x"
        assert result["openalex_url"] == "https://api.openalex.org/works/W1"

    def test_without_doi(self) -> None:
        meta: ca.MetaDict = {"W1": make_meta()}
        result = ca._enrich("W1", meta)
        assert result["doi_url"] is None

    def test_missing_paper(self) -> None:
        result = ca._enrich("UNKNOWN", {})
        assert result["doi"] is None
        assert result["title"] is None


class TestSaveResults:
    """Tests for save_results()."""

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        rr = ca.RankedResults(total_dois=2, papers_fetched=2, top_n=5)
        output = tmp_path / "out.json"
        ca.save_results(rr, output)
        data = json.loads(output.read_text())
        assert "summary" in data
        assert data["summary"]["total_dois_in_csv"] == 2

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        rr = ca.RankedResults()
        output = tmp_path / "nested" / "dir" / "results.json"
        ca.save_results(rr, output)
        assert output.exists()


# ===================================================================
#  Integration tests — call the live Inciteful API
# ===================================================================


@pytest.mark.integration
class TestFetchBatch:
    """Tests for fetch_batch() with real API calls."""

    def test_returns_list_of_dicts(self, known_dois: list[str]) -> None:
        papers = ca.fetch_batch(known_dois[:1])
        assert isinstance(papers, list)
        assert len(papers) >= 1

    def test_response_has_expected_fields(self, known_dois: list[str]) -> None:
        papers = ca.fetch_batch(known_dois[:1])
        paper = papers[0]
        assert "id" in paper
        assert "doi" in paper
        assert "citing" in paper
        assert "cited_by" in paper
        assert "num_citing" in paper
        assert "num_cited_by" in paper

    def test_doi_roundtrips(self, known_dois: list[str]) -> None:
        """The returned paper's DOI should match what we asked for."""
        doi = known_dois[0]
        papers = ca.fetch_batch([doi])
        returned_dois = {p.get("doi") for p in papers}
        assert doi in returned_dois


@pytest.mark.integration
class TestFetchAllPapers:
    """Tests for fetch_all_papers() batching logic."""

    def test_single_batch(self, known_dois: list[str]) -> None:
        papers = ca.fetch_all_papers(known_dois, batch_size=100, delay=0)
        assert len(papers) == len(known_dois)

    def test_multiple_batches(self, known_dois: list[str]) -> None:
        """Force batch_size=1 so each DOI is a separate batch."""
        papers = ca.fetch_all_papers(known_dois, batch_size=1, delay=0.2)
        assert len(papers) == len(known_dois)
        ids = {p["id"] for p in papers}
        assert len(ids) == len(known_dois)


@pytest.mark.integration
class TestFetchExternalMetadata:
    """Tests for fetch_external_metadata() with real API calls."""

    def test_returns_metadata_dict(self, known_openalex_id: str) -> None:
        meta = ca.fetch_external_metadata(
            {known_openalex_id}, batch_size=100, delay=0
        )
        assert known_openalex_id in meta
        info = meta[known_openalex_id]
        assert "doi" in info
        assert "title" in info
        assert "num_citing" in info
        assert "num_cited_by" in info


@pytest.mark.integration
class TestResolveIgnoreIds:
    """Tests for resolve_ignore_ids() with real API calls."""

    def test_openalex_ids_pass_through(
        self, tmp_path: Path, known_openalex_id: str
    ) -> None:
        csv = tmp_path / "ignore.csv"
        csv.write_text(f"DOI\n{known_openalex_id}\n")
        result = ca.resolve_ignore_ids(csv, batch_size=100, delay=0)
        assert known_openalex_id in result

    def test_dois_resolved_to_ids(
        self, tmp_path: Path, known_dois: list[str]
    ) -> None:
        csv = tmp_path / "ignore.csv"
        csv.write_text(f"DOI\n{known_dois[0]}\n")
        result = ca.resolve_ignore_ids(csv, batch_size=100, delay=0)
        assert len(result) == 1
        resolved_id = next(iter(result))
        assert ca._is_openalex_id(resolved_id)

    def test_mixed_entries(
        self, tmp_path: Path, known_dois: list[str], known_openalex_id: str
    ) -> None:
        csv = tmp_path / "ignore.csv"
        csv.write_text(f"DOI\n{known_openalex_id}\n{known_dois[0]}\n")
        result = ca.resolve_ignore_ids(csv, batch_size=100, delay=0)
        assert known_openalex_id in result
        assert len(result) >= 2  # the DOI resolves to a different ID

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = ca.resolve_ignore_ids(tmp_path / "nonexistent.csv")
        assert result == set()

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        csv = tmp_path / "ignore.csv"
        csv.write_text("DOI\n")
        result = ca.resolve_ignore_ids(csv)
        assert result == set()


@pytest.mark.integration
class TestEndToEnd:
    """Full pipeline integration test with a small DOI set."""

    def test_main_produces_json(
        self, tmp_path: Path, known_dois: list[str]
    ) -> None:
        csv = tmp_path / "dois.csv"
        csv.write_text("DOI\n" + "\n".join(known_dois) + "\n")
        output = tmp_path / "results.json"
        ignore = tmp_path / "ignore.csv"
        ignore.write_text("DOI\n")

        ca.main([
            "-i", str(csv),
            "-o", str(output),
            "--ignore", str(ignore),
            "-n", "5",
            "--pool-size", "10",
            "--delay", "0.3",
        ])

        assert output.exists()
        results = json.loads(output.read_text())

        assert "summary" in results
        assert "top_cited_by_inlist" in results
        assert "top_citing_inlist" in results
        assert "top_combined" in results
        assert "top_similarity" in results
        assert "top_adamic_adar" in results
        assert "top_salton" in results

        s = results["summary"]
        assert s["total_dois_in_csv"] == len(known_dois)
        assert s["papers_fetched"] >= 1
        assert s["total_nodes"] > s["in_list_nodes"]
        assert s["total_edges"] > 0

    def test_main_with_ignore_list(
        self, tmp_path: Path, known_dois: list[str]
    ) -> None:
        """Ignored papers should not appear in any result list."""
        csv = tmp_path / "dois.csv"
        csv.write_text("DOI\n" + "\n".join(known_dois) + "\n")
        output = tmp_path / "results.json"

        ignore_empty = tmp_path / "ignore_empty.csv"
        ignore_empty.write_text("DOI\n")
        ca.main([
            "-i", str(csv),
            "-o", str(output),
            "--ignore", str(ignore_empty),
            "-n", "3",
            "--pool-size", "10",
            "--delay", "0.3",
        ])
        baseline = json.loads(output.read_text())
        top_id = baseline["top_cited_by_inlist"][0]["id"]

        ignore_csv = tmp_path / "ignore.csv"
        ignore_csv.write_text(f"DOI\n{top_id}\n")
        output2 = tmp_path / "results2.json"
        ca.main([
            "-i", str(csv),
            "-o", str(output2),
            "--ignore", str(ignore_csv),
            "-n", "3",
            "--pool-size", "10",
            "--delay", "0.3",
        ])
        filtered = json.loads(output2.read_text())

        for key in [
            "top_cited_by_inlist",
            "top_citing_inlist",
            "top_combined",
            "top_similarity",
            "top_adamic_adar",
            "top_salton",
        ]:
            ids_in_results = {entry["id"] for entry in filtered[key]}
            assert top_id not in ids_in_results, (
                f"{top_id} should be filtered from {key}"
            )

    def test_result_entries_have_expected_fields(
        self, tmp_path: Path, known_dois: list[str]
    ) -> None:
        csv = tmp_path / "dois.csv"
        csv.write_text("DOI\n" + "\n".join(known_dois) + "\n")
        output = tmp_path / "results.json"
        ignore = tmp_path / "ignore.csv"
        ignore.write_text("DOI\n")

        ca.main([
            "-i", str(csv),
            "-o", str(output),
            "--ignore", str(ignore),
            "-n", "3",
            "--pool-size", "10",
            "--delay", "0.3",
        ])
        results = json.loads(output.read_text())

        if results["top_cited_by_inlist"]:
            entry = results["top_cited_by_inlist"][0]
            assert "id" in entry
            assert "count" in entry
            assert "openalex_url" in entry

        if results["top_similarity"]:
            entry = results["top_similarity"][0]
            assert "similarity" in entry
            assert "citing_inlist" in entry
            assert "num_citing" in entry
            assert 0 < entry["similarity"] <= 1.0

        if results["top_adamic_adar"]:
            entry = results["top_adamic_adar"][0]
            assert "adamic_adar" in entry
            assert entry["adamic_adar"] > 0

        if results["top_salton"]:
            entry = results["top_salton"][0]
            assert "salton" in entry
            assert entry["salton"] > 0
