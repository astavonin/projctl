"""Tests for projctl.handlers.docs_search — local docs/planning corpus search.

Filesystem-touching tests hit the real filesystem under tmp_path (no
mocking), following tests/test_cli_activity.py's precedent, and construct
DocsSearchHandler(repo_root=...) so no `git rev-parse` subprocess runs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import yaml

from projctl.config import Config, ConfigurationError
from projctl.exceptions import PlatformError
from projctl.handlers import docs_search
from projctl.handlers.docs_search import (
    BM25_B,
    BM25_K1,
    CASE_EXACT_MAX_LEN,
    CASE_EXACT_MIN_LEN,
    DIRECTORY_DECAY,
    FILE_DECAY,
    HEADING_BOOST,
    RANKED_CANDIDATE_CAP,
    ROADMAP_SCOPE_WORD_CAP,
    TEMPLATE_HEADING_SUPPRESSIONS,
    WORD_BUDGET,
    Bm25Ranker,
    CorpusResolver,
    CorpusRoot,
    DocsDigest,
    DocsSearchHandler,
    Hit,
    RankedLine,
    RoadmapEntry,
    RoadmapIndex,
    RoadmapLine,
    ScoredHit,
    SectionExtractor,
    _assemble_digest,
    _assert_ranking_policy_shape,
    _assert_template_heading_shape,
    _budget_ranked,
    _budget_roadmap,
    _classify_kind,
    _compile_token_pattern,
    _dedupe_roots,
    _first_scope_line,
    _greedy_diversify,
    _neutralize_body,
    _read_bold_fields,
    _render_ranked_line,
    _render_roadmap_line,
    _split_sections,
    _walk_corpus_roots,
    tokenize,
)

_SYMLINKS_UNSUPPORTED = not hasattr(os, "symlink")
requires_symlinks = pytest.mark.skipif(
    _SYMLINKS_UNSUPPORTED, reason="the platform provides no symlink support"
)

# ---------------------------------------------------------------------------
# Tokenisation and matching
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_hyphenated_term_stays_one_token(self) -> None:
        assert tokenize("cross-compile") == ["cross-compile"]

    def test_dotted_term_stays_one_token(self) -> None:
        assert tokenize("Params.get") == ["Params.get"]

    def test_plain_words_split_on_whitespace(self) -> None:
        assert tokenize("cross compile") == ["cross", "compile"]

    def test_parenthesized_term_strips_to_bare_word(self) -> None:
        assert tokenize("(design)") == ["design"]

    def test_pure_symbol_term_with_no_alphanumeric_is_taken_verbatim(self) -> None:
        assert tokenize(".*") == [".*"]

    def test_pipe_alternation_survives_the_strip_branch_unchanged(self) -> None:
        assert tokenize("a|b") == ["a|b"]

    def test_plus_plus_survives_the_strip_branch_unchanged(self) -> None:
        assert tokenize("C++") == ["C++"]

    def test_ticket_sigil_survives_the_strip_branch_unchanged(self) -> None:
        assert tokenize("#44") == ["#44"]


class TestCompileTokenPattern:
    def test_plus_plus_matches_at_end_of_line(self) -> None:
        pattern = _compile_token_pattern("C++")
        assert pattern.search("built with C++") is not None

    def test_can_matches_can_and_not_lowercase_variants(self) -> None:
        pattern = _compile_token_pattern("CAN")
        assert pattern.search("the CAN bus") is not None
        assert pattern.search("you can do it") is None
        assert pattern.search("Can you") is None
        assert pattern.search("a scan result") is None

    def test_case_exact_window_lower_bound_two_chars_is_case_exact(self) -> None:
        pattern = _compile_token_pattern("CI")
        assert pattern.search("run CI now") is not None
        assert pattern.search("run ci now") is None

    def test_case_exact_window_upper_bound_five_chars_is_case_exact(self) -> None:
        pattern = _compile_token_pattern("NOTES")
        assert pattern.search("see NOTES here") is not None
        assert pattern.search("see notes here") is None

    def test_single_char_token_case_folds_despite_no_lowercase(self) -> None:
        pattern = _compile_token_pattern("C")
        assert pattern.search("a c compiler") is not None

    def test_six_char_all_caps_token_case_folds(self) -> None:
        pattern = _compile_token_pattern("README")
        assert pattern.search("see readme file") is not None

    def test_token_with_a_lowercase_letter_case_folds_at_any_length(self) -> None:
        pattern = _compile_token_pattern("Can")
        assert pattern.search("you can do it") is not None
        assert pattern.search("you CAN do it") is not None

    def test_rollback_matches_case_folded_but_not_inside_a_longer_word(self) -> None:
        pattern = _compile_token_pattern("rollback")
        assert pattern.search("a rollback happened") is not None
        assert pattern.search("Rollback happened") is not None
        assert pattern.search("rollbacks happened") is None


def test_case_exact_window_constants_match_the_measured_bounds() -> None:
    assert CASE_EXACT_MIN_LEN == 2
    assert CASE_EXACT_MAX_LEN == 5


def test_decay_constants_match_the_documented_factors() -> None:
    assert FILE_DECAY == 0.5
    assert DIRECTORY_DECAY == 0.75


def test_bounding_constants_match_the_documented_values() -> None:
    assert RANKED_CANDIDATE_CAP == 200
    assert ROADMAP_SCOPE_WORD_CAP == 25


class TestTokenizeEdges:
    def test_an_empty_query_yields_no_tokens(self) -> None:
        assert tokenize("") == []

    def test_a_whitespace_only_query_yields_no_tokens(self) -> None:
        assert tokenize("   \t\n ") == []


class TestRankingPolicyShapeGuard:
    """The import-time guard on the ranking-policy block."""

    def test_the_shipped_constants_pass_the_guard(self) -> None:
        _assert_ranking_policy_shape()

    @pytest.mark.parametrize(
        "name, bad_value",
        [
            ("BM25_K1", 0.0),
            ("BM25_B", 1.5),
            ("HEADING_BOOST", 1.0),
            ("FILE_DECAY", 1.0),
            ("DIRECTORY_DECAY", 0.0),
            ("WORD_BUDGET", 0),
            ("ROADMAP_SHARE", 0.5),
            ("CASE_EXACT_MIN_LEN", 9),
            ("ROADMAP_SCOPE_WORD_CAP", 0),
            ("RANKED_CANDIDATE_CAP", 0),
        ],
    )
    def test_a_constant_outside_its_range_raises_at_import_time(
        self, name: str, bad_value, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docs_search, name, bad_value)
        with pytest.raises(ValueError, match="misconfigured"):
            _assert_ranking_policy_shape()


class TestKindPolicyShapeGuard:
    def test_a_kind_left_out_of_every_policy_set_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docs_search, "_VALID_KINDS", docs_search._VALID_KINDS | {"newkind"})
        with pytest.raises(ValueError, match="undecided"):
            docs_search._assert_kind_policy_shape()

    def test_a_policy_set_naming_a_kind_outside_the_closed_set_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docs_search, "_PLAIN_KINDS", frozenset({"bogus"}))
        with pytest.raises(ValueError, match="outside the closed set"):
            docs_search._assert_kind_policy_shape()

    def test_the_classifier_validates_the_kind_it_returns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docs_search, "_VALID_KINDS", frozenset({"docs"}))
        with pytest.raises(ValueError, match="untyped"):
            _classify_kind("planning", "notes.md", [])


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassifyKind:
    def test_docs_root_section_is_docs_kind_whatever_its_filename(self) -> None:
        assert _classify_kind("docs", "architecture.md", ["Backpressure"]) == "docs"

    def test_observed_failures_file_is_failure_kind_whatever_its_headings(self) -> None:
        assert _classify_kind("planning", "observed-failures.md", ["Unrelated"]) == "failure"

    def test_overview_and_status_files_are_roadmap_kind_whatever_their_headings(self) -> None:
        assert _classify_kind("planning", "overview.md", ["Anything"]) == "roadmap"
        assert _classify_kind("planning", "status.md", ["Anything"]) == "roadmap"

    def test_design_section_under_numbered_trade_offs_heading_is_alternative(self) -> None:
        chain = ["7. Trade-offs and Alternatives", "Option A"]
        assert _classify_kind("planning", "design.md", chain) == "alternative"

    def test_design_section_under_bare_goals_heading_is_alternative(self) -> None:
        assert _classify_kind("planning", "design.md", ["Goals"]) == "alternative"

    def test_analysis_section_under_ticket_constraints_heading_is_constraint(self) -> None:
        assert _classify_kind("planning", "analysis.md", ["Ticket Constraints"]) == "constraint"

    def test_heading_predicate_requires_an_exact_match_not_a_superstring(self) -> None:
        chain = ["Goals and Non-Goals of the Rewrite"]
        assert _classify_kind("planning", "design.md", chain) == "untyped"

    def test_nested_heading_under_trade_offs_still_matches_via_ancestry(self) -> None:
        chain = ["7. Trade-offs and Alternatives", "Rejected"]
        assert _classify_kind("planning", "design.md", chain) == "alternative"

    def test_design_section_under_unnamed_heading_is_untyped(self) -> None:
        chain = ["5. Detailed Design"]
        assert _classify_kind("planning", "design.md", chain) == "untyped"

    def test_unrecognized_filename_is_untyped(self) -> None:
        assert _classify_kind("planning", "notes.md", ["Anything"]) == "untyped"


# ---------------------------------------------------------------------------
# Section splitting (fence-aware) and bold-field reading
# ---------------------------------------------------------------------------


class TestSplitSections:
    def test_heading_chain_reflects_full_ancestry(self) -> None:
        text = "# A\ntext a\n## B\ntext b\n### C\ntext c\n"
        sections, _open = _split_sections(text)
        assert sections[-1][0] == ["A", "B", "C"]

    def test_preamble_yields_one_unit_with_empty_heading_chain(self) -> None:
        text = "Some preamble text.\n\n# Title\nbody\n"
        sections, _open = _split_sections(text)
        assert sections[0] == ([], "Some preamble text.")
        assert sections[1][0] == ["Title"]

    def test_empty_preamble_yields_no_preamble_unit(self) -> None:
        text = "# Title\nbody\n"
        sections, _open = _split_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == ["Title"]

    def test_headingless_file_is_wholly_one_unit(self) -> None:
        text = "Just some prose.\nMore prose.\n"
        sections, _open = _split_sections(text)
        assert len(sections) == 1
        assert sections[0] == ([], "Just some prose.\nMore prose.")

    def test_heading_like_line_inside_fenced_block_is_body_text(self) -> None:
        text = "# Real Heading\n```\n# not a heading\n```\nmore body\n"
        sections, _open = _split_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == ["Real Heading"]
        assert "# not a heading" in sections[0][1]

    def test_empty_file_yields_no_units(self) -> None:
        assert _split_sections("") == ([], False)

    def test_heading_at_end_of_file_with_no_body_still_yields_a_unit(self) -> None:
        text = "# A\nbody a\n## B\n"
        sections, _open = _split_sections(text)
        assert sections[-1] == (["A", "B"], "")

    def test_a_fence_closes_only_on_its_own_marker_character(self) -> None:
        text = "# A\n```\ncode\n~~~\nstill code\n```\n## B\nbody b\n"
        sections, fence_left_open = _split_sections(text)
        assert [chain for chain, _body in sections] == [["A"], ["A", "B"]]
        assert fence_left_open is False

    def test_a_closing_marker_shorter_than_its_opener_does_not_close_the_fence(self) -> None:
        text = "# A\n````\ncode\n```\nstill code\n````\n## B\nbody b\n"
        sections, _open = _split_sections(text)
        assert [chain for chain, _body in sections] == [["A"], ["A", "B"]]

    def test_an_unclosed_fence_is_reported_so_the_absorbed_headings_are_visible(self) -> None:
        text = "# A\n```\ncode\n## B\nbody b\n"
        sections, fence_left_open = _split_sections(text)
        assert [chain for chain, _body in sections] == [["A"]]
        assert fence_left_open is True

    def test_a_marker_carrying_an_info_string_does_not_close_a_fence(self) -> None:
        # Treating it as a close would invent "Injected" as a heading out of
        # code-block content and render it into provenance.
        text = "# A\n```\nexample:\n```js\n## Injected\n"
        sections, fence_left_open = _split_sections(text)
        assert [chain for chain, _body in sections] == [["A"]]
        assert fence_left_open is True

    def test_a_fence_indented_four_spaces_is_an_indented_code_block_not_a_fence(self) -> None:
        text = "# A\n    ```\nbody a\n## B\nbody b\n"
        sections, fence_left_open = _split_sections(text)
        assert [chain for chain, _body in sections] == [["A"], ["A", "B"]]
        assert fence_left_open is False

    def test_a_fence_indented_three_spaces_still_opens(self) -> None:
        text = "# A\n   ```\n## Not a heading\n   ```\n## B\nbody b\n"
        sections, fence_left_open = _split_sections(text)
        assert [chain for chain, _body in sections] == [["A"], ["A", "B"]]
        assert fence_left_open is False


class TestReadBoldFields:
    def test_bold_status_line_populates_fields(self) -> None:
        fields = _read_bold_fields("**Status:** covered\nmore text")
        assert fields == {"Status": "covered"}

    def test_status_colon_colon_in_prose_does_not_populate_fields(self) -> None:
        fields = _read_bold_fields("The handler catches Status::sigterm in the loop.")
        assert fields == {}

    def test_a_bold_field_appearing_mid_line_in_prose_is_not_read_as_a_field(self) -> None:
        # The line-start anchor is what separates a record's own field lines
        # from the many prose sentences that quote one.
        assert _read_bold_fields("the gate does not close while **Status:** open") == {}

    def test_a_quoted_field_in_prose_cannot_overwrite_the_records_own_field(self) -> None:
        body = "**Status:** covered\nprose noting that **Status:** open elsewhere"
        assert _read_bold_fields(body) == {"Status": "covered"}


# ---------------------------------------------------------------------------
# SectionExtractor
# ---------------------------------------------------------------------------


def _extract(extractor: SectionExtractor, path: Path, tmp_path: Path, corpus: str = "planning"):
    return extractor.extract(
        path, corpus=corpus, repo="myrepo", platform="gitlab", project_root=tmp_path
    )


class TestSectionExtractor:
    def test_extract_classifies_each_section_and_computes_word_cost(self, tmp_path: Path) -> None:
        path = tmp_path / "design.md"
        path.write_text("# Title\n## Goals\nfirst goal statement here\n")

        hits = _extract(SectionExtractor(), path, tmp_path)

        kinds = {tuple(h.heading_path): h.kind for h in hits}
        assert kinds[("Title",)] == "untyped"
        assert kinds[("Title", "Goals")] == "alternative"
        goals_hit = next(h for h in hits if h.heading_path == ["Title", "Goals"])
        assert goals_hit.cost_words == 4
        assert goals_hit.repo == "myrepo"
        assert goals_hit.platform == "gitlab"
        assert goals_hit.project_root == tmp_path

    def test_field_map_populated_only_for_failure_and_roadmap_kind(self, tmp_path: Path) -> None:
        path = tmp_path / "observed-failures.md"
        path.write_text("## 2026-08-01 Some Failure\n**Status:** covered\ndetails\n")
        hits = _extract(SectionExtractor(), path, tmp_path)
        assert hits[0].kind == "failure"
        assert hits[0].fields == {"Status": "covered"}

    def test_docs_or_untyped_section_carries_no_field_map(self, tmp_path: Path) -> None:
        path = tmp_path / "design.md"
        path.write_text("## Detailed Design\n**Status:** covered\n")
        hits = _extract(SectionExtractor(), path, tmp_path)
        assert hits[0].kind == "untyped"
        assert hits[0].fields == {}

    def test_unreadable_file_is_skipped_and_recorded(self, tmp_path: Path) -> None:
        path = tmp_path / "gone.md"
        extractor = SectionExtractor()
        hits = _extract(extractor, path, tmp_path)
        assert hits == []
        assert len(extractor.skips()) == 1
        assert "gone.md" in extractor.skips()[0].locator

    def test_invalid_utf8_file_is_skipped_and_recorded(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_bytes(b"\xff\xfe not valid utf-8")
        extractor = SectionExtractor()
        hits = _extract(extractor, path, tmp_path)
        assert hits == []
        assert len(extractor.skips()) == 1

    def test_skip_locator_is_relative_to_the_project_root(self, tmp_path: Path) -> None:
        path = tmp_path / "planning" / "bad.md"
        path.parent.mkdir()
        path.write_bytes(b"\xff\xfe not valid utf-8")
        extractor = SectionExtractor()
        _extract(extractor, path, tmp_path)
        assert extractor.skips()[0].locator == "planning/bad.md"

    def test_unclosed_fence_is_recorded_as_a_skip_naming_the_lost_headings(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "notes.md"
        path.write_text("# A\n```\ncode\n## B\nbody\n")
        extractor = SectionExtractor()
        hits = _extract(extractor, path, tmp_path)
        assert [h.heading_path for h in hits] == [["A"]]
        assert any("unclosed code fence" in skip.reason for skip in extractor.skips())


# ---------------------------------------------------------------------------
# Bm25Ranker — scoring and ordering
# ---------------------------------------------------------------------------


def _hit(
    path: str,
    body: str,
    *,
    heading_path=None,
    kind="untyped",
    fields=None,
    repo="repo",
    platform="gitlab",
    project_root=Path("."),
) -> Hit:
    return Hit(
        path=Path(path),
        kind=kind,
        heading_path=heading_path or [],
        body=body,
        cost_words=len(body.split()),
        fields=fields or {},
        repo=repo,
        platform=platform,
        project_root=project_root,
    )


class TestBm25RankerScoring:
    def test_short_section_with_two_occurrences_outranks_a_long_section_with_two(self) -> None:
        short = _hit("a.md", "sysroot appears here and sysroot again")
        long_body = "sysroot " + ("filler word " * 40) + "sysroot"
        long_ = _hit("b.md", long_body)
        ranker = Bm25Ranker([short, long_], "sysroot")
        ordered = ranker.rank()
        assert [sh.hit.path for sh in ordered] == [Path("a.md"), Path("b.md")]

    def test_token_present_in_every_document_contributes_a_non_negative_idf(self) -> None:
        hits = [_hit("dir_a/a.md", "toolchain here"), _hit("dir_b/b.md", "toolchain there")]
        ranked = Bm25Ranker(hits, "toolchain").rank()
        # Without the idf floor the term goes negative, every hit is filtered
        # out for scoring zero, and a loop-only assertion would pass by never
        # running its body.
        assert len(ranked) == 2
        for scored in ranked:
            assert scored.score > 0.0

    def test_a_token_repeated_in_the_query_is_scored_once(self) -> None:
        hits = [_hit("a.md", "the CAN bus carries CAN frames")]
        doubled = Bm25Ranker(hits, "CAN CAN").rank()[0].score
        single = Bm25Ranker(hits, "CAN").rank()[0].score
        assert doubled == pytest.approx(single)

    def test_bm25_constants_match_the_documented_values(self) -> None:
        assert BM25_K1 == 1.2
        assert BM25_B == 0.75

    def test_a_section_matching_only_in_its_own_heading_still_ranks(self) -> None:
        heading_only = _hit(
            "dir_a/a.md", "discussion text with no repeat", heading_path=["Backpressure"]
        )
        body_only = _hit("dir_b/b.md", "backpressure text with no repeat", heading_path=["Other"])
        ranked = Bm25Ranker([heading_only, body_only], "backpressure").rank()
        scores = {sh.hit.path: sh.score for sh in ranked}
        assert len(ranked) == 2
        # Both bodies are five words, so the two BM25 terms are equal and any
        # difference between the scores is boost alone. The heading match is
        # already the whole of the first unit's term frequency, so it earns no
        # boost on top of it.
        assert scores[Path("dir_a/a.md")] == pytest.approx(scores[Path("dir_b/b.md")])

    def test_a_zero_word_section_matching_on_its_heading_alone_scores_unboosted(self) -> None:
        # Both units are empty-bodied and score their token off the own-heading
        # floor alone, so their BM25 terms are identical by construction: equal
        # document frequency, equal length. Only the boost can separate them,
        # and neither earns it — one because its heading is template
        # vocabulary, the other because the heading is all it matched on.
        heading_only = _hit("dir_a/a.md", "", heading_path=["Backpressure"])
        template_suppressed = _hit("dir_c/c.md", "", heading_path=["Goals"])
        ranked = Bm25Ranker([heading_only, template_suppressed], "backpressure goals").rank()
        scores = {sh.hit.path: sh.score for sh in ranked}

        assert len(ranked) == 2
        assert scores[Path("dir_a/a.md")] > 0.0
        assert scores[Path("dir_a/a.md")] == pytest.approx(scores[Path("dir_c/c.md")])

    def test_a_token_in_an_ancestor_heading_alone_admits_no_section(self) -> None:
        # Every file has an H1, so admitting on the joined chain would make one
        # word in a file's title match every section that file holds.
        under_matching_title = [
            _hit("a.md", "unrelated prose here", heading_path=["Review Notes", f"Part {i}"])
            for i in range(6)
        ]
        body_match = _hit("b.md", "a review happened here", heading_path=["Other"])

        ranker = Bm25Ranker(under_matching_title + [body_match], "review")
        ranked = ranker.rank()

        assert [sh.hit.path for sh in ranked] == [Path("b.md")]
        assert ranker.doc_frequency()["review"] == 1

    def test_an_ancestor_heading_match_still_boosts_a_section_that_matches_elsewhere(self) -> None:
        # FR-16 scopes the boost to the whole chain; only admission narrows.
        ancestor = _hit(
            "dir_a/a.md", "backpressure discussion", heading_path=["Backpressure", "Details"]
        )
        baseline = _hit("dir_b/b.md", "backpressure discussion", heading_path=["Unrelated"])
        scores = {
            sh.hit.path: sh.score for sh in Bm25Ranker([ancestor, baseline], "backpressure").rank()
        }
        assert scores[Path("dir_a/a.md")] == pytest.approx(
            scores[Path("dir_b/b.md")] * HEADING_BOOST
        )

    def test_document_frequency_counts_every_unit_the_score_reads(self) -> None:
        heading_only = _hit("a.md", "unrelated body", heading_path=["Backpressure"])
        body_only = _hit("b.md", "backpressure in the body", heading_path=["Other"])
        ranker = Bm25Ranker([heading_only, body_only], "backpressure")
        assert ranker.doc_frequency()["backpressure"] == len(ranker.rank())

    def test_an_empty_corpus_ranks_nothing_without_raising(self) -> None:
        ranker = Bm25Ranker([], "sysroot")
        assert ranker.rank() == []
        assert ranker.doc_frequency() == {"sysroot": 0}

    def test_a_corpus_of_empty_bodies_ranks_on_its_heading_chain_alone(self) -> None:
        # Every body is empty, so the average document length is 0.0 and the
        # length-normalisation term takes its protective branch.
        hits = [_hit("a.md", "", heading_path=["Sysroot"])]
        assert len(Bm25Ranker(hits, "sysroot").rank()) == 1

    def test_score_ties_across_kind_when_bm25_and_boost_are_equal(self) -> None:
        # Each hit lives in its own directory so the directory-decay term
        # (which only discounts a second file sharing a parent) cannot
        # break the tie this test asserts.
        hits = [
            _hit("dir_a/a.md", "sysroot text here", kind="failure", fields={"Status": "covered"}),
            _hit("dir_b/b.md", "sysroot text here", kind="docs"),
            _hit("dir_c/c.md", "sysroot text here", kind="untyped"),
        ]
        ranker = Bm25Ranker(hits, "sysroot")
        scores = {sh.score for sh in ranker.rank()}
        assert len(scores) == 1

    def test_query_token_appearing_only_in_the_file_stem_earns_no_boost_or_match(self) -> None:
        hit = _hit("planning/backpressure.md", "unrelated body content", heading_path=["Other"])
        ranker = Bm25Ranker([hit], "backpressure")
        assert ranker.rank() == []

    def test_heading_boost_applies_exactly_2_5x(self) -> None:
        assert HEADING_BOOST == 2.5
        boosted = _hit("dir_a/a.md", "backpressure discussion text", heading_path=["Backpressure"])
        baseline = _hit("dir_b/b.md", "backpressure discussion text", heading_path=["Unrelated"])
        ranker = Bm25Ranker([boosted, baseline], "backpressure")
        scores = {sh.hit.path: sh.score for sh in ranker.rank()}
        assert scores[Path("dir_a/a.md")] == pytest.approx(scores[Path("dir_b/b.md")] * 2.5)

    @pytest.mark.parametrize(
        "heading", ["Goals", "2. Goals and Non-Goals", "Medium", "Findings By Severity"]
    )
    def test_heading_boost_suppressed_for_template_vocabulary(self, heading: str) -> None:
        token = heading.split()[-1].rstrip(".").lower()
        suppressed = _hit("dir_a/a.md", f"{token} discussion text", heading_path=[heading])
        baseline = _hit("dir_b/b.md", f"{token} discussion text", heading_path=["Unrelated"])
        ranker = Bm25Ranker([suppressed, baseline], token)
        scores = {sh.hit.path: sh.score for sh in ranker.rank()}
        assert scores[Path("dir_a/a.md")] == pytest.approx(scores[Path("dir_b/b.md")])


class TestOpenFailurePin:
    @pytest.mark.parametrize("status", ["open", "Open", "OPEN"])
    def test_open_failure_precedes_a_higher_scoring_non_pinned_unit(self, status: str) -> None:
        pinned = _hit(
            "f.md",
            "sysroot",
            kind="failure",
            fields={"Status": status},
        )
        high_score = _hit("b.md", "sysroot " * 20, heading_path=["Sysroot"])
        ranker = Bm25Ranker([pinned, high_score], "sysroot")
        ordered = ranker.rank()
        assert ordered[0].hit.path == Path("f.md")

    def test_qualifier_after_em_dash_still_pins(self) -> None:
        pinned = _hit("f.md", "sysroot", kind="failure", fields={"Status": "open — not reproduced"})
        # Strictly the stronger unit on score, so only the pin can put the
        # failure first — three occurrences in a short body, heading-boosted.
        other = _hit("b.md", "sysroot sysroot sysroot", kind="docs", heading_path=["Sysroot"])
        ranker = Bm25Ranker([pinned, other], "sysroot")
        ordered = ranker.rank()
        assert ordered[0].hit.path == Path("f.md")
        assert ordered[1].score > ordered[0].score

    def test_an_open_failure_outside_the_candidate_cap_is_still_ranked_first(self) -> None:
        # FR-16 puts an open record ahead of every other unit whatever its
        # score, so the cap must not be able to discard one first.
        corpus = [
            _hit(f"d{i}/f{i}.md", "sysroot sysroot sysroot sysroot")
            for i in range(RANKED_CANDIDATE_CAP + 100)
        ]
        buried = _hit(
            "ledger/observed-failures.md",
            "sysroot " + ("filler " * 400),
            kind="failure",
            fields={"Status": "open"},
        )

        ranker = Bm25Ranker(corpus + [buried], "sysroot")
        ranked = ranker.rank()

        assert ranked[0].hit.path == Path("ledger/observed-failures.md")
        assert ranker.omitted() == 100

    @pytest.mark.parametrize("status", ["covered", "waived", "out-of-scope", "unknown"])
    def test_non_open_status_values_do_not_pin(self, status: str) -> None:
        low = _hit("f.md", "sysroot", kind="failure", fields={"Status": status})
        high = _hit("b.md", "sysroot " * 20, heading_path=["Sysroot"])
        ranker = Bm25Ranker([low, high], "sysroot")
        assert ranker.rank()[0].hit.path == Path("b.md")

    def test_failure_section_with_no_status_field_does_not_pin(self) -> None:
        low = _hit("f.md", "sysroot", kind="failure", fields={})
        high = _hit("b.md", "sysroot " * 20, heading_path=["Sysroot"])
        ranker = Bm25Ranker([low, high], "sysroot")
        assert ranker.rank()[0].hit.path == Path("b.md")


class TestBm25RankerRaisesOnUnknownKind:
    def test_unrecognized_kind_reaching_the_ranker_raises(self) -> None:
        bad = _hit("a.md", "text", kind="bogus")
        with pytest.raises(ValueError, match="bogus"):
            Bm25Ranker([bad], "text")


@pytest.mark.parametrize(
    "entry, reason",
    [
        ("7. Trade-offs and Alternatives", "numbering"),
        ("Trade-offs and Alternatives", "case-folded"),
        ("", "non-empty"),
    ],
)
def test_template_heading_suppression_shape_assertion_fires_on_malformed_entry(
    entry: str, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        _assert_template_heading_shape(frozenset({entry}))


def test_template_heading_suppression_shape_assertion_passes_for_the_real_constant() -> None:
    _assert_template_heading_shape(TEMPLATE_HEADING_SUPPRESSIONS)


# ---------------------------------------------------------------------------
# Greedy diversity decay
# ---------------------------------------------------------------------------


class TestGreedyDiversify:
    def test_second_and_third_picks_from_one_file_are_discounted(self) -> None:
        h1 = _hit("a.md", "one")
        h2 = _hit("a.md", "two")
        h3 = _hit("a.md", "three")
        ordered = _greedy_diversify([(h1, 10.0), (h2, 9.0), (h3, 8.0)])
        assert [sh.hit.body for sh in ordered] == ["one", "two", "three"]
        assert ordered[1].score == pytest.approx(9.0 * FILE_DECAY)
        assert ordered[2].score == pytest.approx(8.0 * FILE_DECAY**2)

    def test_second_and_third_files_in_one_directory_are_discounted(self) -> None:
        a = _hit("dir/a.md", "one")
        b = _hit("dir/b.md", "two")
        c = _hit("dir/c.md", "three")
        ordered = _greedy_diversify([(a, 10.0), (b, 9.0), (c, 8.0)])
        assert [sh.hit.path.name for sh in ordered] == ["a.md", "b.md", "c.md"]
        assert ordered[1].score == pytest.approx(9.0 * DIRECTORY_DECAY)
        assert ordered[2].score == pytest.approx(8.0 * DIRECTORY_DECAY**2)

    def test_no_file_is_capped_three_top_sections_in_one_file_all_selected(self) -> None:
        hits = [_hit("a.md", str(i)) for i in range(3)]
        ordered = _greedy_diversify([(h, 5.0) for h in hits])
        assert len(ordered) == 3


class TestRankingAtCorpusScale:
    """Diversification is quadratic in its input, so the input is bounded."""

    @staticmethod
    def _corpus(size: int) -> list[Hit]:
        return [_hit(f"d{i % 40}/f{i % 400}.md", f"sysroot mention {i}") for i in range(size)]

    def test_four_thousand_matching_units_rank_within_a_wall_clock_ceiling(self) -> None:
        ranker = Bm25Ranker(self._corpus(4000), "sysroot")
        start = time.monotonic()
        ranked = ranker.rank()
        elapsed = time.monotonic() - start

        assert len(ranked) == RANKED_CANDIDATE_CAP
        assert ranker.omitted() == 4000 - RANKED_CANDIDATE_CAP
        # Uncapped, this input measured 25 s; the ceiling is loose enough that
        # only a return to corpus-sized diversification can trip it.
        assert elapsed < 5.0

    def test_the_cap_keeps_the_highest_scoring_units_and_drops_the_rest(self) -> None:
        # One file per directory, so no decay reorders the greedy pass and the
        # strongest units are selected in relevance order.
        strong = [_hit(f"s{i}/f.md", "sysroot sysroot sysroot filler") for i in range(20)]
        weak = [_hit(f"w{i}/f.md", "sysroot filler filler filler") for i in range(4000)]

        ranked = Bm25Ranker(strong + weak, "sysroot").rank()

        assert [sh.hit.path for sh in ranked[:20]] == [hit.path for hit in strong]


# ---------------------------------------------------------------------------
# Budget assembly
# ---------------------------------------------------------------------------


def _entry(name: str, words: int) -> RoadmapEntry:
    return RoadmapEntry(path=Path(f"{name}/overview.md"), label=name, text=" ".join([name] * words))


class TestBudgetRanked:
    def test_unit_over_remaining_budget_emits_locator_with_no_body(self) -> None:
        scored = ScoredHit(hit=_hit("a.md", "one two three four five"), score=1.0)
        lines, _dropped = _budget_ranked([scored], budget=2)
        assert lines[0].locator_only is True

    def test_6054_word_section_never_emits_a_body_at_the_full_word_budget(self) -> None:
        big = _hit("a.md", "word " * 6054)
        scored = ScoredHit(hit=big, score=1.0)
        lines, _dropped = _budget_ranked([scored], budget=WORD_BUDGET)
        assert lines[0].locator_only is True

    def test_ranked_half_degrades_to_locators_then_drops_the_tail(self) -> None:
        scored_hits = [ScoredHit(hit=_hit(f"{i}.md", "word " * 50), score=1.0) for i in range(5)]
        lines, dropped = _budget_ranked(scored_hits, budget=60)
        assert [line.locator_only for line in lines] == [False, True]
        assert dropped == 3

    def test_the_ranked_half_emits_no_more_words_than_its_budget_plus_one_line(self) -> None:
        scored_hits = [ScoredHit(hit=_hit(f"{i}.md", "word " * 60), score=1.0) for i in range(400)]
        lines, dropped = _budget_ranked(scored_hits, budget=WORD_BUDGET)
        emitted = sum(len(_render_ranked_line(line).split()) for line in lines)
        widest = max(len(_render_ranked_line(line).split()) for line in lines)
        assert emitted <= WORD_BUDGET + widest
        assert len(lines) + dropped == 400


class TestBudgetRoadmap:
    def test_entries_past_the_share_degrade_to_locator_then_drop(self) -> None:
        # Each entry renders "- <name> <name>" (3 words) in full and
        # "- **<name>** — overview.md" (4 words) as a locator, so a budget of
        # 8 buys two full entries, then one locator, then drops the rest.
        entries = [_entry(str(i), 2) for i in range(5)]
        lines, dropped, _leftover = _budget_roadmap(entries, budget=8)
        assert [line.locator_only for line in lines] == [False, False, True]
        assert dropped == 2

    def test_roadmap_set_smaller_than_its_share_releases_the_remainder(self) -> None:
        entries = [_entry("a", 5)]
        _lines, dropped, leftover = _budget_roadmap(entries, budget=300)
        assert dropped == 0
        assert leftover == 300 - len("- a a a a a".split())

    def test_roadmap_set_that_fully_consumes_its_share_releases_nothing(self) -> None:
        entries = [_entry(str(i), 100) for i in range(30)]
        _lines, _dropped, leftover = _budget_roadmap(entries, budget=300)
        assert leftover == 0


def test_roadmap_leftover_extends_the_ranked_half_budget() -> None:
    roadmap_entries = [_entry("a", 5)]
    big_hit = _hit("b.md", "word " * 750)
    ranked_hits = [ScoredHit(hit=big_hit, score=1.0)]

    digest = _assemble_digest(
        roadmap_entries=roadmap_entries,
        ranked_hits=ranked_hits,
        doc_frequency={},
        resolved_projects=[],
        skips=[],
        zero_related=False,
    )

    # Base ranked share is 700 words; 750 only fits once the ~294-word
    # roadmap leftover is released to it.
    assert digest.ranked_lines[0].locator_only is False


def test_ranked_half_never_borrows_when_roadmap_leaves_no_leftover() -> None:
    roadmap_entries = [_entry(str(i), 100) for i in range(30)]
    big_hit = _hit("b.md", "word " * 750)
    ranked_hits = [ScoredHit(hit=big_hit, score=1.0)]

    digest = _assemble_digest(
        roadmap_entries=roadmap_entries,
        ranked_hits=ranked_hits,
        doc_frequency={},
        resolved_projects=[],
        skips=[],
        zero_related=False,
    )

    assert digest.ranked_lines[0].locator_only is True


# ---------------------------------------------------------------------------
# RoadmapIndex
# ---------------------------------------------------------------------------


def _roadmap_hit(tmp_path: Path, relative: str, body: str, **kwargs) -> Hit:
    return _hit(str(tmp_path / relative), body, kind="roadmap", project_root=tmp_path, **kwargs)


class TestRoadmapIndex:
    def test_entries_order_by_source_file_path(self, tmp_path: Path) -> None:
        hit_b = _roadmap_hit(tmp_path, "b/overview.md", "About text.", heading_path=["About"])
        hit_a = _roadmap_hit(tmp_path, "a/overview.md", "About text.", heading_path=["About"])
        entries = RoadmapIndex().entries([hit_a, hit_b], tmp_path)
        assert [e.path.parent.name for e in entries] == ["a", "b"]

    def test_phase_prefers_state_over_status(self, tmp_path: Path) -> None:
        hit = _roadmap_hit(
            tmp_path, "a/status.md", "", fields={"State": "opened", "Status": "in progress"}
        )
        entries = RoadmapIndex().entries([hit], tmp_path)
        assert "opened" in entries[0].text

    def test_phase_falls_back_to_status_when_state_absent(self, tmp_path: Path) -> None:
        hit = _roadmap_hit(tmp_path, "a/status.md", "", fields={"Status": "closed"})
        entries = RoadmapIndex().entries([hit], tmp_path)
        assert "closed" in entries[0].text

    def test_entry_falls_back_to_folder_label_alone_when_no_phase_present(
        self, tmp_path: Path
    ) -> None:
        hit = _roadmap_hit(tmp_path, "a/status.md", "", fields={})
        entries = RoadmapIndex().entries([hit], tmp_path)
        assert entries[0].text == "a"

    def test_two_files_sharing_a_parent_name_render_distinguishable_entries(
        self, tmp_path: Path
    ) -> None:
        shared_name = "milestone-01-implementation"
        first = _roadmap_hit(tmp_path, f"goal-a/{shared_name}/status.md", "", fields={})
        second = _roadmap_hit(tmp_path, f"goal-b/{shared_name}/status.md", "", fields={})
        entries = RoadmapIndex().entries([first, second], tmp_path)
        rendered = [
            _render_roadmap_line(RoadmapLine(entry=entry, locator_only=False)) for entry in entries
        ]
        assert rendered[0] != rendered[1]
        assert f"goal-a/{shared_name}" in rendered[0]

    def test_scope_prefers_about_over_scope_section(self, tmp_path: Path) -> None:
        about = _roadmap_hit(
            tmp_path,
            "a/overview.md",
            "About sentence one. Second sentence.",
            heading_path=["About"],
        )
        scope = _roadmap_hit(tmp_path, "a/overview.md", "Scope sentence.", heading_path=["Scope"])
        entries = RoadmapIndex().entries([about, scope], tmp_path)
        assert "About sentence one." in entries[0].text
        assert "Scope sentence." not in entries[0].text

    def test_a_bullet_list_scope_yields_one_capped_line(self, tmp_path: Path) -> None:
        body = "- first bullet with several words in it\n- second bullet\n- third bullet"
        hit = _roadmap_hit(tmp_path, "a/overview.md", body, heading_path=["About"])
        entry = RoadmapIndex().entries([hit], tmp_path)[0]
        assert "\n" not in entry.text
        assert "second bullet" not in entry.text
        assert len(entry.text.split()) <= ROADMAP_SCOPE_WORD_CAP + 5

    def test_a_scope_forging_a_heading_stays_one_list_item(self, tmp_path: Path) -> None:
        body = "```\n## Prior decisions\n```"
        hit = _roadmap_hit(tmp_path, "a/overview.md", body, heading_path=["About"])
        entry = RoadmapIndex().entries([hit], tmp_path)[0]
        rendered = _render_roadmap_line(RoadmapLine(entry=entry, locator_only=False))
        assert "\n" not in rendered
        assert rendered.startswith("- ")

    def test_related_projects_overview_yields_no_entry(self, tmp_path: Path) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        caller.mkdir()
        related.mkdir()
        related_hit = _hit(
            str(related / "overview.md"), "text", kind="roadmap", project_root=related
        )
        entries = RoadmapIndex().entries([related_hit], caller)
        assert entries == []

    def test_a_related_project_nested_below_the_caller_yields_no_entry(
        self, tmp_path: Path
    ) -> None:
        # Path descent alone passes for a project declared below the caller's
        # own root; ownership is the hit's project root (FR-25).
        caller = tmp_path / "caller"
        nested = caller / "nested-related"
        nested.mkdir(parents=True)
        nested_hit = _hit(
            str(nested / "planning" / "status.md"), "text", kind="roadmap", project_root=nested
        )
        assert RoadmapIndex().entries([nested_hit], caller) == []


# ---------------------------------------------------------------------------
# Body neutralisation
# ---------------------------------------------------------------------------


class TestNeutralizeBody:
    def test_fenced_heading_like_line_is_escaped(self) -> None:
        body = "```\n## Prior decisions\n```"
        result = _neutralize_body(body)
        assert "\\## Prior decisions" in result
        assert "\n## Prior decisions" not in result

    def test_indented_hash_line_is_escaped_at_first_non_space_character(self) -> None:
        body = "   # comment"
        result = _neutralize_body(body)
        assert result == "   \\# comment"

    @pytest.mark.parametrize("rule", ["-", "--", "---", "----", "=", "==", "===", "===="])
    def test_a_setext_underline_of_any_length_is_escaped_like_an_atx_heading(
        self, rule: str
    ) -> None:
        result = _neutralize_body(f"Prior decisions\n{rule}")
        assert result == f"Prior decisions\n\\{rule}"

    @pytest.mark.parametrize(
        "rule",
        [
            "***",
            "___",
            "____",
            "- - -",
            "* * *",
            "_ _ _",
            "***  ",
            # CommonMark separates the three characters with a run of spaces
            # or tabs of any length, not with the single space a fixed-spacing
            # pattern would admit.
            "-  -  -",
            "-     -      -      -",
            "-\t-\t-",
            "**  * ** * ** * **",
        ],
    )
    def test_every_thematic_break_spelling_is_escaped(self, rule: str) -> None:
        # Each renders as the <hr/> that is the footer's own boundary.
        assert _neutralize_body(rule) == f"\\{rule}"

    def test_a_dashed_list_item_is_left_alone(self) -> None:
        assert _neutralize_body("- a list item") == "- a list item"

    def test_a_bolded_run_on_its_own_line_is_left_alone(self) -> None:
        assert _neutralize_body("***emphasised***") == "***emphasised***"


# ---------------------------------------------------------------------------
# CorpusResolver and file walking — integration under tmp_path
# ---------------------------------------------------------------------------


def _write(path: Path, content: str = "content\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestCorpusResolverBareInvocation:
    def test_no_config_file_yields_default_docs_path_and_no_related(self, tmp_path: Path) -> None:
        _write(tmp_path / "planning" / "notes.md")
        resolver = CorpusResolver(tmp_path)
        roots = resolver.roots(related=False)
        assert any(r.corpus == "planning" for r in roots)
        assert resolver.resolved_projects()[0].platform == "undeclared"

    def test_missing_default_docs_path_is_silent(self, tmp_path: Path) -> None:
        _write(tmp_path / "planning" / "notes.md")
        resolver = CorpusResolver(tmp_path)
        resolver.roots(related=False)
        assert resolver.skips() == []

    def test_repo_with_no_planning_directory_yields_no_roots_and_no_error(
        self, tmp_path: Path
    ) -> None:
        resolver = CorpusResolver(tmp_path)
        roots = resolver.roots(related=False)
        assert roots == []
        assert resolver.skips() == []


class TestCorpusResolverDocsPath:
    def _config(self, tmp_path: Path, search_yaml: str) -> Path:
        cfg = tmp_path / "projctl.yaml"
        cfg.write_text(search_yaml)
        return cfg

    def test_configured_docs_path_missing_warns_and_is_skipped(self, tmp_path: Path) -> None:
        self._config(tmp_path, "search:\n  docs_path: documentation\n")
        resolver = CorpusResolver(tmp_path)
        roots = resolver.roots(related=False)
        assert not any(r.corpus == "docs" for r in roots)
        assert any("configured path is missing" in s.reason for s in resolver.skips())

    def test_docs_path_null_means_no_docs_corpus_silently(self, tmp_path: Path) -> None:
        self._config(tmp_path, "search:\n  docs_path: null\n")
        _write(tmp_path / "docs" / "guide.md")
        resolver = CorpusResolver(tmp_path)
        roots = resolver.roots(related=False)
        assert not any(r.corpus == "docs" for r in roots)
        assert resolver.skips() == []

    def test_docs_path_dot_is_rejected_by_containment(self, tmp_path: Path) -> None:
        self._config(tmp_path, "search:\n  docs_path: '.'\n")
        resolver = CorpusResolver(tmp_path)
        roots = resolver.roots(related=False)
        assert not any(r.corpus == "docs" for r in roots)
        assert any("resolves to the project root" in s.reason for s in resolver.skips())

    def test_docs_path_outside_project_is_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "elsewhere"
        outside.mkdir(exist_ok=True)
        self._config(tmp_path, "search:\n  docs_path: '../elsewhere'\n")
        resolver = CorpusResolver(tmp_path)
        roots = resolver.roots(related=False)
        assert not any(r.corpus == "docs" for r in roots)
        assert any("outside it" in s.reason for s in resolver.skips())

    def test_docs_path_naming_a_regular_file_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "docs_file.md").write_text("x")
        self._config(tmp_path, "search:\n  docs_path: docs_file.md\n")
        resolver = CorpusResolver(tmp_path)
        roots = resolver.roots(related=False)
        assert not any(r.corpus == "docs" for r in roots)
        assert any("not a directory" in s.reason for s in resolver.skips())

    def test_docs_path_nested_inside_planning_is_indexed_once_as_docs(self, tmp_path: Path) -> None:
        self._config(tmp_path, "search:\n  docs_path: planning/sub\n")
        _write(tmp_path / "planning" / "sub" / "guide.md")
        resolver = CorpusResolver(tmp_path)
        roots = resolver.roots(related=False)
        pairs, _skips = _walk_corpus_roots(roots)
        assert len(pairs) == 1
        assert pairs[0][0].corpus == "docs"

    def test_docs_path_naming_planning_directory_collapses_to_one_planning_root(
        self, tmp_path: Path
    ) -> None:
        self._config(tmp_path, "search:\n  docs_path: planning\n")
        _write(tmp_path / "planning" / "notes.md")
        resolver = CorpusResolver(tmp_path)
        roots = resolver.roots(related=False)
        assert len(roots) == 1
        assert roots[0].corpus == "planning"


class TestCorpusResolverRelatedHop:
    def test_related_project_contributes_its_own_docs_path_and_platform(
        self, tmp_path: Path
    ) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        _write(caller / "planning" / "notes.md")
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {related}\n")
        _write(related / "documentation" / "guide.md")
        (related / "projctl.yaml").write_text(
            "platform: github\nsearch:\n  docs_path: documentation\n"
        )

        resolver = CorpusResolver(caller)
        roots = resolver.roots(related=True)
        docs_roots = [r for r in roots if r.repo == "related" and r.corpus == "docs"]
        assert len(docs_roots) == 1
        assert docs_roots[0].platform == "github"
        # A run that resolved a related project is not a zero-related run.
        assert resolver.zero_related() is False

    def test_declared_path_absent_on_disk_warns_and_is_skipped(self, tmp_path: Path) -> None:
        caller = tmp_path / "caller"
        _write(caller / "planning" / "notes.md")
        (caller / "projctl.yaml").write_text("search:\n  related:\n    - ../does-not-exist\n")

        resolver = CorpusResolver(caller)
        resolver.roots(related=True)
        assert any("absent on this machine" in s.reason for s in resolver.skips())
        assert resolver.zero_related() is True

    def test_related_project_with_neither_corpus_contributes_nothing_and_is_not_a_skip(
        self, tmp_path: Path
    ) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        _write(caller / "planning" / "notes.md")
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {related}\n")
        related.mkdir()

        resolver = CorpusResolver(caller)
        roots = resolver.roots(related=True)
        assert not any(r.repo == "related" for r in roots)
        assert resolver.skips() == []
        related_project = next(p for p in resolver.resolved_projects() if p.repo == "related")
        assert related_project.root_kinds == []

    def test_related_project_whose_search_section_has_the_wrong_shape_is_skipped_not_fatal(
        self, tmp_path: Path
    ) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        _write(caller / "planning" / "notes.md")
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {related}\n")
        _write(related / "planning" / "notes.md")
        (related / "projctl.yaml").write_text("search: [not, a, mapping]\n")

        resolver = CorpusResolver(caller)
        roots = resolver.roots(related=True)
        assert not any(r.repo == "related" for r in roots)
        assert any("config fault" in s.reason for s in resolver.skips())

    def test_related_project_with_an_unreadable_config_file_is_skipped_not_fatal(
        self, tmp_path: Path
    ) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        _write(caller / "planning" / "notes.md")
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {related}\n")
        _write(related / "planning" / "notes.md")
        config = related / "projctl.yaml"
        config.write_text("search:\n  docs_path: docs\n")
        config.chmod(0o000)
        if os.access(config, os.R_OK):
            pytest.skip("the test process can read a 0o000 file (running as root)")

        resolver = CorpusResolver(caller)
        try:
            roots = resolver.roots(related=True)
        finally:
            config.chmod(0o644)

        assert not any(r.repo == "related" for r in roots)
        assert any("config fault" in s.reason for s in resolver.skips())

    def test_related_project_with_malformed_yaml_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        _write(caller / "planning" / "notes.md")
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {related}\n")
        _write(related / "planning" / "notes.md")
        (related / "projctl.yaml").write_text("not: valid: yaml: [")

        resolver = CorpusResolver(caller)
        resolver.roots(related=True)
        assert any("config fault" in s.reason for s in resolver.skips())

    def test_zero_related_projects_resolved_sets_the_footer_flag(self, tmp_path: Path) -> None:
        caller = tmp_path / "caller"
        _write(caller / "planning" / "notes.md")
        (caller / "projctl.yaml").write_text("search:\n  related: []\n")
        resolver = CorpusResolver(caller)
        resolver.roots(related=True)
        assert resolver.zero_related() is True

    def test_two_related_entries_naming_one_directory_resolve_once(self, tmp_path: Path) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        _write(caller / "planning" / "notes.md")
        _write(related / "planning" / "notes.md")
        (caller / "projctl.yaml").write_text(
            f"search:\n  related:\n    - {related}\n    - {related}/\n"
        )

        resolver = CorpusResolver(caller)
        roots = resolver.roots(related=True)
        related_roots = [r for r in roots if r.repo == "related"]
        assert len(related_roots) == 1

    def test_related_entry_naming_the_caller_indexes_the_caller_once(self, tmp_path: Path) -> None:
        caller = tmp_path / "caller"
        _write(caller / "planning" / "notes.md")
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {caller}\n")

        resolver = CorpusResolver(caller)
        roots = resolver.roots(related=True)
        pairs, _skips = _walk_corpus_roots(roots)
        assert len(pairs) == 1
        # The caller resolves twice — once as itself, once as the declared
        # entry — and the footer must list it once.
        assert [p.repo for p in resolver.resolved_projects()] == ["caller"]

    def test_related_projects_own_related_list_is_never_read(self, tmp_path: Path) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        third = tmp_path / "third"
        _write(caller / "planning" / "notes.md")
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {related}\n")
        _write(related / "planning" / "notes.md")
        (related / "projctl.yaml").write_text(f"search:\n  related:\n    - {third}\n")
        _write(third / "planning" / "notes.md")

        resolver = CorpusResolver(caller)
        roots = resolver.roots(related=True)
        assert not any(r.repo == "third" for r in roots)

    def test_no_platform_key_legacy_config_and_no_config_file_all_render_undeclared(
        self, tmp_path: Path
    ) -> None:
        no_platform = tmp_path / "no-platform"
        legacy = tmp_path / "legacy"
        no_config = tmp_path / "no-config"
        for repo in (no_platform, legacy, no_config):
            _write(repo / "planning" / "notes.md")
        (no_platform / "projctl.yaml").write_text("search:\n  docs_path: null\n")
        (legacy / "projctl.yaml").write_text(
            "labels:\n  default: []\ngitlab:\n  default_group: g\n"
        )

        caller = tmp_path / "caller"
        _write(caller / "planning" / "notes.md")
        (caller / "projctl.yaml").write_text(
            "search:\n  related:\n" f"    - {no_platform}\n    - {legacy}\n    - {no_config}\n"
        )

        resolver = CorpusResolver(caller)
        with pytest.warns(DeprecationWarning, match="deprecated format"):
            resolver.roots(related=True)
        platforms = {p.repo: p.platform for p in resolver.resolved_projects()}
        assert platforms["no-platform"] == "undeclared"
        assert platforms["legacy"] == "undeclared"
        assert platforms["no-config"] == "undeclared"


class TestCorpusResolverExplicitConfig:
    def test_nonexistent_explicit_config_path_raises_file_not_found(self, tmp_path: Path) -> None:
        resolver = CorpusResolver(tmp_path, config_path=tmp_path / "nope.yaml")
        with pytest.raises(FileNotFoundError):
            resolver.roots(related=False)

    def test_malformed_search_section_in_callers_own_config_raises(self, tmp_path: Path) -> None:
        cfg = tmp_path / "custom.yaml"
        cfg.write_text("search: 3\n")
        resolver = CorpusResolver(tmp_path, config_path=cfg)
        from projctl.config import ConfigurationError

        with pytest.raises(ConfigurationError):
            resolver.roots(related=False)

    def test_related_path_resolves_against_repo_root_not_config_file_directory(
        self, tmp_path: Path
    ) -> None:
        repo_root = tmp_path / "repo"
        elsewhere = tmp_path / "config-dir"
        related = tmp_path / "related"
        _write(repo_root / "planning" / "notes.md")
        _write(related / "planning" / "notes.md")
        elsewhere.mkdir()
        cfg = elsewhere / "custom.yaml"
        cfg.write_text("search:\n  related:\n    - ../related\n")

        resolver = CorpusResolver(repo_root, config_path=cfg)
        roots = resolver.roots(related=True)
        assert any(r.repo == "related" for r in roots)


# ---------------------------------------------------------------------------
# File walking — dedup, C-5 exclusion, symlink escape
# ---------------------------------------------------------------------------


class TestWalkCorpusRoots:
    def test_request_md_files_are_excluded(self, tmp_path: Path) -> None:
        _write(tmp_path / "design-review-request.md")
        _write(tmp_path / "design.md")
        root = CorpusRoot(
            repo="r",
            platform="gitlab",
            corpus="planning",
            project_root=tmp_path,
            root=tmp_path,
            origin="caller",
        )
        pairs, _skips = _walk_corpus_roots([root])
        names = {p.name for _r, p in pairs}
        assert "design.md" in names
        assert "design-review-request.md" not in names

    def test_file_reachable_from_two_roots_belongs_to_the_deepest_one(self, tmp_path: Path) -> None:
        nested = tmp_path / "planning" / "docs"
        _write(nested / "guide.md")
        planning_root = CorpusRoot(
            repo="r",
            platform="gitlab",
            corpus="planning",
            project_root=tmp_path,
            root=tmp_path / "planning",
            origin="caller",
        )
        docs_root = CorpusRoot(
            repo="r",
            platform="gitlab",
            corpus="docs",
            project_root=tmp_path,
            root=nested,
            origin="caller",
        )
        pairs, _skips = _walk_corpus_roots([planning_root, docs_root])
        assert len(pairs) == 1
        assert pairs[0][0].corpus == "docs"

    @requires_symlinks
    def test_symlinked_file_escaping_its_root_is_skipped_and_the_rest_still_indexed(
        self, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside.md"
        outside.write_text("x")
        planning = tmp_path / "planning"
        planning.mkdir()
        _write(planning / "kept.md")
        symlink = planning / "escaped.md"
        symlink.symlink_to(outside)

        root = CorpusRoot(
            repo="r",
            platform="gitlab",
            corpus="planning",
            project_root=tmp_path,
            root=planning,
            origin="caller",
        )
        pairs, skips = _walk_corpus_roots([root])
        names = {p.name for _r, p in pairs}
        assert "kept.md" in names
        assert "escaped.md" not in names
        assert any("resolves outside its corpus root" in s.reason for s in skips)


def test_dedupe_roots_collapses_equal_roots_to_planning() -> None:
    shared = Path("/tmp/shared-root")
    docs_first = CorpusRoot(
        repo="r",
        platform="gitlab",
        corpus="docs",
        project_root=Path("/tmp"),
        root=shared,
        origin="caller",
    )
    planning_second = CorpusRoot(
        repo="r",
        platform="gitlab",
        corpus="planning",
        project_root=Path("/tmp"),
        root=shared,
        origin="caller",
    )
    deduped = _dedupe_roots([docs_first, planning_second])
    assert len(deduped) == 1
    assert deduped[0].corpus == "planning"


# ---------------------------------------------------------------------------
# End-to-end DocsSearchHandler integration
# ---------------------------------------------------------------------------


# Every line a consumer splitting the digest on headings or on the footer's
# own boundary reads as structure. A corpus- or config-derived value that can
# add one of these has forged part of the document (FR-35).
_CONTRACTED_DIGEST_STRUCTURE = [
    "## Roadmap",
    "## Prior decisions",
    "---",
    "### Resolved corpus",
    "### Query token document frequency",
]


def _structural_lines(rendered: str) -> list[str]:
    return [
        line
        for line in rendered.splitlines()
        if line.startswith("## ") or line.startswith("### ") or line == "---"
    ]


class TestDocsSearchHandlerIntegration:
    def test_gitlab_epic_milestone_issue_shape_yields_typed_alternative_hits(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        issue_dir = tmp_path / "planning" / "my-epic" / "milestone-01-x" / "issues" / "007-slug"
        _write(
            issue_dir / "design.md",
            "# Design\n## 7. Trade-offs and Alternatives\nsysroot discussion\n",
        )

        handler = DocsSearchHandler(repo_root=tmp_path)
        digest = handler.search("sysroot", related=False)
        capsys.readouterr()

        kinds = {line.scored.hit.kind for line in digest.ranked_lines}
        assert "alternative" in kinds
        hit = next(
            line.scored.hit for line in digest.ranked_lines if line.scored.hit.kind == "alternative"
        )
        assert hit.heading_path == ["Design", "7. Trade-offs and Alternatives"]

    def test_flat_orphan_tree_yields_typed_failure_hits_with_no_issue_number(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        orphan_dir = tmp_path / "planning" / "reviews-orphan" / "main-0e68987"
        _write(
            orphan_dir / "observed-failures.md",
            "## 2026-08-01 Sysroot bug\n**Status:** covered\nsysroot details\n",
        )

        handler = DocsSearchHandler(repo_root=tmp_path)
        digest = handler.search("sysroot", related=False)
        capsys.readouterr()

        kinds = {line.scored.hit.kind for line in digest.ranked_lines}
        assert "failure" in kinds

    def test_untyped_fallback_hits_render_with_untyped_annotation_and_unchanged_score(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "loose-note.md", "sysroot discussion in a loose note\n")

        handler = DocsSearchHandler(repo_root=tmp_path)
        digest = handler.search("sysroot", related=False)
        out = capsys.readouterr().out

        assert "[untyped]" in out
        assert any(line.scored.hit.kind == "untyped" for line in digest.ranked_lines)

    def test_repository_root_readme_is_never_indexed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        (tmp_path / "README.md").write_text("sysroot mention at repo root\n")
        _write(tmp_path / "planning" / "notes.md", "sysroot mention in planning\n")

        handler = DocsSearchHandler(repo_root=tmp_path)
        digest = handler.search("sysroot", related=False)
        capsys.readouterr()

        paths = {line.scored.hit.path for line in digest.ranked_lines}
        assert not any(p.name == "README.md" for p in paths)

    def test_tracker_less_tree_yields_one_roadmap_entry_per_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(
            tmp_path / "planning" / "goal" / "overview.md",
            "## About\nProject about sentence.\n",
        )
        for i in range(3):
            _write(
                tmp_path / "planning" / "goal" / f"step-{i}" / "status.md",
                f"**State:** opened\nstep {i}\n",
            )

        handler = DocsSearchHandler(repo_root=tmp_path)
        digest = handler.search("sysroot", related=False)
        capsys.readouterr()

        assert len(digest.roadmap_lines) == 4

    def test_query_matching_nothing_yields_empty_prior_decisions_but_roadmap_still_renders(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "goal" / "overview.md", "## About\nSomething.\n")

        handler = DocsSearchHandler(repo_root=tmp_path)
        digest = handler.search("zzzznomatch", related=False)
        out = capsys.readouterr().out

        assert digest.ranked_lines == []
        assert "## Roadmap" in out
        assert "## Prior decisions" in out

    def test_digest_ordering_is_roadmap_then_prior_decisions_then_footer(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "goal" / "overview.md", "## About\nSomething.\n")
        _write(tmp_path / "planning" / "notes.md", "sysroot notes\n")

        handler = DocsSearchHandler(repo_root=tmp_path)
        handler.search("sysroot", related=False)
        out = capsys.readouterr().out

        assert (
            out.index("## Roadmap")
            < out.index("## Prior decisions")
            < out.index("### Resolved corpus")
        )

    def test_document_frequency_reported_for_every_query_token(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "a.md", "sysroot alpha\n")
        _write(tmp_path / "planning" / "b.md", "toolchain beta\n")

        handler = DocsSearchHandler(repo_root=tmp_path)
        handler.search("sysroot toolchain", related=False)
        out = capsys.readouterr().out

        assert "`sysroot`: 1" in out
        assert "`toolchain`: 1" in out

    def test_adding_a_related_project_moves_the_reported_document_frequency(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        _write(caller / "planning" / "a.md", "sysroot alpha\n")
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {related}\n")
        _write(related / "planning" / "b.md", "sysroot beta\n")

        handler_bare = DocsSearchHandler(repo_root=caller)
        handler_bare.search("sysroot", related=False)
        out_bare = capsys.readouterr().out

        handler_related = DocsSearchHandler(repo_root=caller)
        handler_related.search("sysroot", related=True)
        out_related = capsys.readouterr().out

        assert "`sysroot`: 1" in out_bare
        assert "`sysroot`: 2" in out_related

    def test_related_flag_with_zero_related_projects_differs_from_a_bare_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "a.md", "sysroot alpha\n")

        handler_bare = DocsSearchHandler(repo_root=tmp_path)
        handler_bare.search("sysroot", related=False)
        out_bare = capsys.readouterr().out

        handler_related = DocsSearchHandler(repo_root=tmp_path)
        handler_related.search("sysroot", related=True)
        out_related = capsys.readouterr().out

        assert out_bare != out_related
        assert "--related resolved zero related projects" in out_related

    def test_skipped_files_and_faults_are_named_in_the_footer_and_the_rest_still_indexed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "good.md", "sysroot text\n")
        bad = tmp_path / "planning" / "bad.md"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"\xff\xfe garbage")

        handler = DocsSearchHandler(repo_root=tmp_path)
        digest = handler.search("sysroot", related=False)
        out = capsys.readouterr().out

        assert any(line.scored.hit.path.name == "good.md" for line in digest.ranked_lines)
        assert "### Skipped" in out
        assert "bad.md" in out

    def test_stdout_carries_the_digest_alone_while_the_skip_reaches_the_logger(
        self, tmp_path: Path, capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        caller = tmp_path / "caller"
        _write(caller / "planning" / "notes.md", "sysroot text\n")
        (caller / "projctl.yaml").write_text("search:\n  related:\n    - ../does-not-exist\n")

        handler = DocsSearchHandler(repo_root=caller)
        with caplog.at_level("WARNING"):
            handler.search("sysroot", related=True)
        captured = capsys.readouterr()

        # The footer is the durable record of what was skipped (stdout, part
        # of the digest itself); the live warning is a separate signal that
        # goes through the logger rather than being printed directly.
        assert captured.out.startswith("## Roadmap")
        assert "does-not-exist" in caplog.text

    def test_no_git_repository_raises_platform_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _refuse(*_args, **_kwargs):
            raise PlatformError("Not in a git repository.")

        monkeypatch.setattr(docs_search, "get_repo_root", _refuse)
        with pytest.raises(PlatformError):
            DocsSearchHandler().search("sysroot", related=False)

    def test_analysis_ticket_constraints_yield_typed_constraint_hits(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(
            tmp_path / "planning" / "goal" / "issue" / "analysis.md",
            "# Research\n## Ticket Constraints\nthe sysroot must be relocatable\n",
        )

        digest = DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        out = capsys.readouterr().out

        assert "constraint" in {line.scored.hit.kind for line in digest.ranked_lines}
        assert "[constraint]" in out

    def test_one_failure_hit_is_produced_per_dated_record(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(
            tmp_path / "planning" / "goal" / "issue" / "observed-failures.md",
            "# Observed Failures\n"
            "## 2026-08-01 First sysroot fault\n**Status:** covered\ndetail one\n"
            "## 2026-08-02 Second sysroot fault\n**Status:** open\ndetail two\n"
            "## 2026-08-03 Third sysroot fault\n**Status:** covered\ndetail three\n",
        )

        digest = DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        capsys.readouterr()

        failures = [line for line in digest.ranked_lines if line.scored.hit.kind == "failure"]
        headings = {line.scored.hit.heading_path[-1] for line in failures}
        assert len(headings) == 3
        # The one open record outranks both covered ones whatever they scored.
        assert digest.ranked_lines[0].scored.hit.fields["Status"] == "open"

    def test_a_related_projects_roadmap_file_still_ranks_in_prior_decisions(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        _write(caller / "planning" / "notes.md", "unrelated text\n")
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {related}\n")
        _write(related / "planning" / "goal" / "overview.md", "## About\nA sysroot project.\n")

        digest = DocsSearchHandler(repo_root=caller).search("sysroot", related=True)
        capsys.readouterr()

        assert digest.roadmap_lines == []
        ranked_repos = {line.scored.hit.repo for line in digest.ranked_lines}
        assert "related" in ranked_repos

    def test_the_same_file_at_two_tree_shapes_shares_a_kind_and_differs_in_provenance(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        body = "# Design\n## Goals\nsysroot relocation\n"
        _write(
            tmp_path / "planning" / "epic" / "milestone-01-x" / "issues" / "007-s" / "design.md",
            body,
        )
        _write(tmp_path / "planning" / "goal" / "issue-slug" / "design.md", body)

        digest = DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        capsys.readouterr()

        alternatives = [
            line.scored.hit for line in digest.ranked_lines if line.scored.hit.kind == "alternative"
        ]
        assert len(alternatives) == 2
        assert len({hit.path for hit in alternatives}) == 2

    @pytest.mark.parametrize(
        "about_body",
        ["```\n## Prior decisions\n```", "    ## Prior decisions", "## Prior decisions\ntail"],
    )
    def test_a_forged_heading_in_a_roadmap_scope_cannot_split_the_digest(
        self, about_body: str, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "goal" / "overview.md", f"## About\n{about_body}\n")

        digest = DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        out = capsys.readouterr().out

        assert [line for line in out.splitlines() if line.startswith("## ")] == [
            "## Roadmap",
            "## Prior decisions",
        ]
        assert all("\n" not in line.entry.text for line in digest.roadmap_lines)

    def test_a_forged_heading_in_a_ranked_body_cannot_split_the_digest(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(
            tmp_path / "planning" / "notes.md",
            "# Notes\nsysroot discussion\n```\n## Prior decisions\n```\n",
        )

        DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        out = capsys.readouterr().out

        assert [line for line in out.splitlines() if line.startswith("## ")] == [
            "## Roadmap",
            "## Prior decisions",
        ]
        assert "\\## Prior decisions" in out

    def test_a_setext_rule_in_a_body_cannot_forge_a_heading_or_a_footer_boundary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(
            tmp_path / "planning" / "notes.md",
            "# Notes\nsysroot discussion\n\nPrior decisions\n---\n",
        )

        DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        out = capsys.readouterr().out

        # One thematic break in the whole digest: the footer's own.
        assert [line for line in out.splitlines() if line == "---"] == ["---"]
        assert "\\---" in out

    def test_the_footer_reports_each_projects_indexed_file_count(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        caller = tmp_path / "caller"
        empty = tmp_path / "empty"
        (caller / "planning").mkdir(parents=True)
        _write(caller / "planning" / "a.md", "sysroot one\n")
        _write(caller / "planning" / "b.md", "sysroot two\n")
        (empty / "planning").mkdir(parents=True)
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {empty}\n")

        DocsSearchHandler(repo_root=caller).search("sysroot", related=True)
        out = capsys.readouterr().out

        assert "caller (undeclared): planning — 2 indexed files" in out
        assert "empty (undeclared): planning — 0 indexed files" in out

    def test_a_related_projects_unexpandable_docs_path_skips_only_that_project(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        _write(caller / "planning" / "a.md", "sysroot in caller\n")
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {related}\n")
        _write(related / "planning" / "b.md", "sysroot in related\n")
        (related / "projctl.yaml").write_text('search:\n  docs_path: "~nosuchuser/docs"\n')

        digest = DocsSearchHandler(repo_root=caller).search("sysroot", related=True)
        capsys.readouterr()

        assert any("config fault" in skip.reason for skip in digest.skips)
        assert {line.scored.hit.repo for line in digest.ranked_lines} == {"caller"}

    def test_the_callers_own_unexpandable_docs_path_is_a_configuration_error(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path / "planning" / "a.md", "sysroot text\n")
        (tmp_path / "projctl.yaml").write_text('search:\n  docs_path: "~nosuchuser/docs"\n')

        with pytest.raises(ConfigurationError, match="docs_path"):
            DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)

    def test_an_absolute_docs_path_outside_the_project_is_skipped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        outside = tmp_path.parent / "outside-docs"
        outside.mkdir(exist_ok=True)
        repo = tmp_path / "repo"
        _write(repo / "planning" / "a.md", "sysroot text\n")
        (repo / "projctl.yaml").write_text(f"search:\n  docs_path: {outside}\n")

        digest = DocsSearchHandler(repo_root=repo).search("sysroot", related=False)
        capsys.readouterr()

        assert any("outside it" in skip.reason for skip in digest.skips)

    @requires_symlinks
    def test_a_docs_path_reached_through_a_symlinked_directory_is_rejected_by_containment(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        outside = tmp_path / "outside"
        (outside / "guide").mkdir(parents=True)
        repo = tmp_path / "repo"
        _write(repo / "planning" / "a.md", "sysroot text\n")
        (repo / "docs").symlink_to(outside, target_is_directory=True)
        (repo / "projctl.yaml").write_text("search:\n  docs_path: docs\n")

        digest = DocsSearchHandler(repo_root=repo).search("sysroot", related=False)
        capsys.readouterr()

        assert not any(skip for skip in digest.skips if "missing" in skip.reason)
        assert any("outside it" in skip.reason for skip in digest.skips)

    def test_docs_path_dot_is_skipped_and_the_git_directory_is_never_walked(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "a.md", "sysroot text\n")
        _write(tmp_path / ".git" / "hooks" / "notes.md", "sysroot inside dot git\n")
        (tmp_path / "projctl.yaml").write_text("search:\n  docs_path: '.'\n")

        digest = DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        capsys.readouterr()

        assert any("resolves to the project root" in skip.reason for skip in digest.skips)
        assert not any(".git" in str(line.scored.hit.path) for line in digest.ranked_lines)

    def test_a_file_that_cannot_be_opened_is_skipped_and_the_rest_still_indexed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "good.md", "sysroot text\n")
        unreadable = tmp_path / "planning" / "locked.md"
        unreadable.write_text("sysroot secret\n")
        unreadable.chmod(0o000)
        if os.access(unreadable, os.R_OK):
            pytest.skip("the test process can read a 0o000 file (running as root)")

        try:
            digest = DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        finally:
            unreadable.chmod(0o644)
        capsys.readouterr()

        assert any("unreadable" in skip.reason for skip in digest.skips)
        assert any(line.scored.hit.path.name == "good.md" for line in digest.ranked_lines)

    def test_score_components_are_logged_for_units_the_digest_never_shows(
        self, tmp_path: Path, capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        for i in range(RANKED_CANDIDATE_CAP + 5):
            _write(tmp_path / "planning" / f"d{i}" / f"n{i}.md", f"sysroot mention {i}\n")

        with caplog.at_level("DEBUG", logger="projctl.handlers.docs_search"):
            DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        capsys.readouterr()

        score_lines = [ln for ln in caplog.text.splitlines() if "search docs: score" in ln]
        assert len(score_lines) == RANKED_CANDIDATE_CAP + 5
        assert any("not selected" in line for line in score_lines)
        for marker in ("bm25=", "boost=", "pin=", "decay="):
            assert any(marker in line for line in score_lines)

    def test_every_heading_boost_outcome_is_named_distinctly_in_the_debug_channel(
        self, tmp_path: Path, capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Four outcomes, four names: a boost withheld because the heading was
        # the only match has a different remedy from one suppressed as
        # template vocabulary, so a reader tracing a rank must tell them apart.
        _write(tmp_path / "planning" / "a" / "design.md", "## Goals\ngoals discussion\n")
        _write(tmp_path / "planning" / "b" / "notes.md", "a goals mention in prose\n")
        _write(tmp_path / "planning" / "c" / "notes.md", "## Goals Roadmap\ngoals discussion\n")
        _write(tmp_path / "planning" / "d" / "notes.md", "## Goals Roadmap\nunrelated prose\n")

        with caplog.at_level("DEBUG", logger="projctl.handlers.docs_search"):
            DocsSearchHandler(repo_root=tmp_path).search("goals", related=False)
        capsys.readouterr()

        score_lines = [ln for ln in caplog.text.splitlines() if "search docs: score" in ln]
        assert any("a/design.md" in ln and "boost=suppressed" in ln for ln in score_lines)
        assert any("b/notes.md" in ln and "boost=none" in ln for ln in score_lines)
        assert any("c/notes.md" in ln and "boost=applied" in ln for ln in score_lines)
        assert any("d/notes.md" in ln and "boost=floor-only" in ln for ln in score_lines)

    def test_the_footer_names_cap_omitted_units_separately_from_budget_drops(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        for i in range(RANKED_CANDIDATE_CAP + 50):
            _write(tmp_path / "planning" / f"d{i}" / f"n{i}.md", f"sysroot mention {i}\n")

        digest = DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        out = capsys.readouterr().out

        assert digest.ranked_omitted == 50
        assert "### 50 further matches not ranked (candidate cap)" in out
        assert f"### {digest.ranked_dropped} further matches not shown (budget exhausted)" in out

    def test_a_project_whose_every_file_is_unreadable_reports_zero_indexed_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        planning = tmp_path / "planning"
        planning.mkdir()
        for name in ("a.md", "b.md"):
            locked = planning / name
            locked.write_text("sysroot secret\n")
            locked.chmod(0o000)
            if os.access(locked, os.R_OK):
                pytest.skip("the test process can read a 0o000 file (running as root)")

        try:
            DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        finally:
            for name in ("a.md", "b.md"):
                (planning / name).chmod(0o644)
        out = capsys.readouterr().out

        assert f"{tmp_path.name} (undeclared): planning — 0 indexed files" in out

    def test_a_partially_indexed_file_still_counts_as_indexed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        # The unclosed fence costs the headings below it, not the whole file.
        _write(tmp_path / "planning" / "a.md", "# A\n```\nsysroot code\n## B\nbody\n")

        DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        out = capsys.readouterr().out

        assert f"{tmp_path.name} (undeclared): planning — 1 indexed file\n" in out

    def test_no_absolute_path_reaches_the_digest_from_an_unreadable_file_or_an_absent_related(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        caller = tmp_path / "caller"
        _write(caller / "planning" / "good.md", "sysroot text\n")
        locked = caller / "planning" / "locked.md"
        locked.write_text("sysroot secret\n")
        locked.chmod(0o000)
        if os.access(locked, os.R_OK):
            pytest.skip("the test process can read a 0o000 file (running as root)")
        absent = tmp_path / "private" / "secret-project"
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {absent}\n")

        try:
            digest = DocsSearchHandler(repo_root=caller).search("sysroot", related=True)
        finally:
            locked.chmod(0o644)
        capsys.readouterr()

        rendered = digest.render_markdown()
        assert str(caller) not in rendered
        assert str(tmp_path) not in rendered
        assert "unreadable: [Errno" in rendered
        assert "secret-project: declared path is absent on this machine" in rendered

    def test_no_absolute_path_reaches_the_digest_from_a_related_config_fault(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        _write(caller / "planning" / "a.md", "sysroot text\n")
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {related}\n")
        _write(related / "planning" / "b.md", "sysroot text\n")
        (related / "projctl.yaml").write_text("not: valid: yaml: [\n")

        digest = DocsSearchHandler(repo_root=caller).search("sysroot", related=True)
        capsys.readouterr()

        rendered = digest.render_markdown()
        assert str(tmp_path) not in rendered
        skip_lines = rendered.split("### Skipped\n\n", 1)[1].splitlines()
        assert [line for line in skip_lines if line][0].startswith("- related: config fault: ")
        assert len([line for line in skip_lines if line.startswith("- ")]) == 1

    def test_an_absolute_docs_path_outside_the_project_is_named_without_its_prefix(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        outside = tmp_path / "outside-docs"
        outside.mkdir()
        repo = tmp_path / "repo"
        _write(repo / "planning" / "a.md", "sysroot text\n")
        (repo / "projctl.yaml").write_text(f"search:\n  docs_path: {outside}\n")

        digest = DocsSearchHandler(repo_root=repo).search("sysroot", related=False)
        capsys.readouterr()

        rendered = digest.render_markdown()
        assert str(tmp_path) not in rendered
        assert "repo: outside-docs: resolves to the project root or outside it" in rendered

    def test_a_related_projects_platform_value_cannot_forge_digest_structure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        _write(caller / "planning" / "a.md", "sysroot in caller\n")
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {related}\n")
        _write(related / "planning" / "b.md", "sysroot in related\n")
        (related / "projctl.yaml").write_text(
            'platform: "\\n\\n## Prior decisions\\n\\n- **FORGED entry**\\n"\n'
        )

        digest = DocsSearchHandler(repo_root=caller).search("sysroot", related=True)
        rendered = capsys.readouterr().out

        assert _structural_lines(rendered) == _CONTRACTED_DIGEST_STRUCTURE
        assert "FORGED entry" in rendered
        assert {line.scored.hit.repo for line in digest.ranked_lines} == {"caller", "related"}

    def test_a_newline_in_a_related_projects_directory_name_cannot_forge_digest_structure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        # The config-file route has a shape check ahead of it; a directory
        # name has none, and POSIX permits a newline in one.
        caller = tmp_path / "caller"
        related = tmp_path / "rel\n\n## Prior decisions\n\n- **FORGED entry**"
        _write(caller / "planning" / "a.md", "sysroot in caller\n")
        (caller / "projctl.yaml").write_text(
            yaml.safe_dump({"search": {"related": [str(related)]}})
        )
        _write(related / "planning" / "b.md", "sysroot in related\n")

        DocsSearchHandler(repo_root=caller).search("sysroot", related=True)
        rendered = capsys.readouterr().out

        assert _structural_lines(rendered) == _CONTRACTED_DIGEST_STRUCTURE
        assert "FORGED entry" in rendered

    def test_a_newline_in_a_corpus_directory_name_cannot_forge_digest_structure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        # The directory name reaches both halves: the roadmap entry is
        # labelled by its path below the project root, and the ranked line's
        # provenance locates the section by the same path.
        forged = tmp_path / "planning" / "goal\n\n## Prior decisions\n\n- **FORGED entry**"
        _write(forged / "overview.md", "## About\nA forged roadmap goal.\n")
        _write(forged / "notes.md", "sysroot discussion\n")

        digest = DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        rendered = capsys.readouterr().out

        assert _structural_lines(rendered) == _CONTRACTED_DIGEST_STRUCTURE
        assert rendered.count("FORGED entry") == 2
        assert len(digest.roadmap_lines) == 1
        assert len(digest.ranked_lines) == 1

    def test_a_related_projects_platform_mapping_skips_only_that_project(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        _write(caller / "planning" / "a.md", "sysroot in caller\n")
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {related}\n")
        _write(related / "planning" / "b.md", "sysroot in related\n")
        (related / "projctl.yaml").write_text("platform:\n  name: gitlab\n")

        digest = DocsSearchHandler(repo_root=caller).search("sysroot", related=True)
        capsys.readouterr()

        assert any("config fault" in skip.reason for skip in digest.skips)
        assert {line.scored.hit.repo for line in digest.ranked_lines} == {"caller"}

    def test_two_related_projects_sharing_a_directory_name_keep_separate_footer_rows(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        caller = tmp_path / "caller"
        first = tmp_path / "a" / "shared"
        second = tmp_path / "b" / "shared"
        _write(caller / "planning" / "c.md", "sysroot in caller\n")
        (caller / "projctl.yaml").write_text(
            f"search:\n  related:\n    - {first}\n    - {second}\n"
        )
        _write(first / "planning" / "one.md", "sysroot in the first\n")
        _write(second / "planning" / "two.md", "sysroot in the second\n")
        (second / "projctl.yaml").write_text("platform: github\n")

        digest = DocsSearchHandler(repo_root=caller).search("sysroot", related=True)
        out = capsys.readouterr().out

        assert [(p.repo, p.platform) for p in digest.resolved_projects] == [
            ("caller", "undeclared"),
            ("shared", "undeclared"),
            ("shared", "github"),
        ]
        assert "shared (undeclared): planning — 1 indexed file\n" in out
        assert "shared (github): planning — 1 indexed file\n" in out

    def test_a_related_entry_naming_no_resolvable_home_is_contained_and_named(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        # The entry never reaches the hop's own guard: the footer locator is
        # built first, and expanding `~nosuchuser/` is itself the fault.
        _write(tmp_path / "planning" / "a.md", "sysroot text\n")
        (tmp_path / "projctl.yaml").write_text('search:\n  related:\n    - "~nosuchuser/x"\n')

        digest = DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=True)
        out = capsys.readouterr().out

        rendered = digest.render_markdown()
        assert "### Skipped" in rendered
        assert "~nosuchuser/x" in rendered
        assert str(tmp_path) not in rendered
        assert any(line.scored.hit.path.name == "a.md" for line in digest.ranked_lines)
        assert out.startswith("## Roadmap")

        # No config file was opened, so the reason must name the expansion —
        # `config fault:` would point an operator at a file nothing read.
        (skip,) = [s for s in digest.skips if "nosuchuser" in s.locator]
        assert "expanded" in skip.reason
        assert "config fault" not in skip.reason

    def test_an_unreadable_corpus_subdirectory_is_named_and_the_rest_still_indexed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "open" / "good.md", "sysroot text\n")
        locked = tmp_path / "planning" / "locked"
        _write(locked / "hidden.md", "sysroot secret\n")
        locked.chmod(0o000)
        if os.access(locked, os.R_OK):
            pytest.skip("the test process can read a 0o000 directory (running as root)")

        try:
            digest = DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        finally:
            locked.chmod(0o755)
        out = capsys.readouterr().out

        assert any("unreadable directory" in skip.reason for skip in digest.skips)
        assert "planning/locked: unreadable directory: [Errno" in out
        assert [line.scored.hit.path.name for line in digest.ranked_lines] == ["good.md"]

    def test_only_units_matching_a_query_token_reach_the_score_channel(
        self, tmp_path: Path, capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The channel answers "why did the document I expected not appear", so
        # it must carry the matched set rather than the whole corpus.
        for i in range(30):
            _write(tmp_path / "planning" / f"d{i}" / f"n{i}.md", f"unrelated prose {i}\n")
        _write(tmp_path / "planning" / "match" / "hit.md", "sysroot discussion\n")

        with caplog.at_level("DEBUG", logger="projctl.handlers.docs_search"):
            DocsSearchHandler(repo_root=tmp_path).search("sysroot", related=False)
        capsys.readouterr()

        score_lines = [ln for ln in caplog.text.splitlines() if "search docs: score" in ln]
        assert len(score_lines) == 1
        assert "match/hit.md" in score_lines[0]

    def test_resolved_roots_are_logged_with_their_origin(
        self, tmp_path: Path, capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        _write(caller / "planning" / "a.md", "sysroot\n")
        _write(related / "planning" / "b.md", "sysroot\n")
        (caller / "projctl.yaml").write_text(f"search:\n  related:\n    - {related}\n")

        with caplog.at_level("DEBUG", logger="projctl.handlers.docs_search"):
            DocsSearchHandler(repo_root=caller).search("sysroot", related=True)
        capsys.readouterr()

        assert "origin=caller" in caplog.text
        assert "origin=declared entry" in caplog.text
        # Root resolution explains the score lines below it, so it is emitted
        # where it happens rather than after everything it accounts for.
        lines = caplog.text.splitlines()
        last_root = max(i for i, ln in enumerate(lines) if "resolved root" in ln)
        first_score = min(i for i, ln in enumerate(lines) if "search docs: score" in ln)
        assert last_root < first_score


# ---------------------------------------------------------------------------
# Config accessor — search: section (see also tests/test_config.py)
# ---------------------------------------------------------------------------


class TestSearchConfigDocsPathUnexpanded:
    def test_related_entries_are_returned_unexpanded(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "projctl.yaml"
        cfg_path.write_text("search:\n  related:\n    - ~/src/shared-lib\n")
        config = Config(cfg_path)
        search_config = config.get_search_config()
        assert search_config.related == ["~/src/shared-lib"]


# ---------------------------------------------------------------------------
# Small pure-function edges not exercised by the integration paths above
# ---------------------------------------------------------------------------


class TestFirstScopeLine:
    def test_empty_text_yields_empty_string(self) -> None:
        assert _first_scope_line("") == ""

    def test_whitespace_only_text_yields_empty_string(self) -> None:
        assert _first_scope_line("   \n  ") == ""

    def test_terminated_prose_yields_its_first_sentence(self) -> None:
        assert _first_scope_line("One. Two.") == "One."

    def test_an_unterminated_bullet_list_yields_only_its_first_bullet(self) -> None:
        assert _first_scope_line("- first bullet\n- second bullet") == "first bullet"

    @pytest.mark.parametrize("marker", ["-", "*", "+", "1.", "2)"])
    def test_every_list_marker_shape_is_stripped(self, marker: str) -> None:
        assert _first_scope_line(f"{marker} scoped line\nmore") == "scoped line"

    def test_an_unterminated_body_longer_than_the_cap_is_truncated(self) -> None:
        assert _first_scope_line("word " * 200).split() == ["word"] * ROADMAP_SCOPE_WORD_CAP + ["…"]

    def test_a_terminated_sentence_longer_than_the_cap_is_truncated(self) -> None:
        text = ("word " * 200) + "end. Second sentence."
        assert len(_first_scope_line(text).split()) == ROADMAP_SCOPE_WORD_CAP + 1

    def test_a_body_of_bare_list_markers_yields_empty_string(self) -> None:
        assert _first_scope_line("- \n- \n") == ""


class TestWalkCorpusRootsSkipsNonFiles:
    def test_a_directory_matching_the_md_glob_is_not_treated_as_a_file(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "weird.md").mkdir()
        _write(tmp_path / "real.md")
        root = CorpusRoot(
            repo="r",
            platform="gitlab",
            corpus="planning",
            project_root=tmp_path,
            root=tmp_path,
            origin="caller",
        )
        pairs, _skips = _walk_corpus_roots([root])
        assert [p.name for _r, p in pairs] == ["real.md"]

    def test_a_root_that_is_not_a_directory_is_skipped_without_error(self, tmp_path: Path) -> None:
        vanished = tmp_path / "vanished"
        root = CorpusRoot(
            repo="r",
            platform="gitlab",
            corpus="planning",
            project_root=tmp_path,
            root=vanished,
            origin="caller",
        )
        pairs, skips = _walk_corpus_roots([root])
        assert pairs == []
        assert skips == []


class TestRenderLines:
    def test_locator_only_ranked_line_renders_only_the_provenance_header(self, tmp_path) -> None:
        hit = _hit(
            str(tmp_path / "planning" / "a.md"),
            "some body text",
            heading_path=["Heading"],
            project_root=tmp_path,
        )
        line = RankedLine(scored=ScoredHit(hit=hit, score=1.0), locator_only=True)
        rendered = _render_ranked_line(line)
        assert "some body text" not in rendered
        assert "planning/a.md" in rendered
        assert str(tmp_path) not in rendered

    def test_full_ranked_line_with_empty_body_renders_only_the_header(self) -> None:
        hit = _hit("a.md", "", heading_path=["Heading"])
        line = RankedLine(scored=ScoredHit(hit=hit, score=1.0), locator_only=False)
        rendered = _render_ranked_line(line)
        assert rendered.count("\n") == 0

    def test_provenance_falls_back_to_the_absolute_path_outside_its_project_root(
        self, tmp_path: Path
    ) -> None:
        outside = tmp_path.parent / "elsewhere" / "a.md"
        hit = _hit(str(outside), "body", project_root=tmp_path)
        line = RankedLine(scored=ScoredHit(hit=hit, score=1.0), locator_only=True)
        assert str(outside) in _render_ranked_line(line)

    def test_an_unrecognized_kind_reaching_the_renderer_raises(self) -> None:
        hit = Hit(
            path=Path("a.md"),
            kind="bogus",
            heading_path=[],
            body="text",
            cost_words=1,
            fields={},
            repo="r",
            platform="gitlab",
            project_root=Path("."),
        )
        line = RankedLine(scored=ScoredHit(hit=hit, score=1.0), locator_only=True)
        with pytest.raises(ValueError, match="bogus"):
            _render_ranked_line(line)

    def test_locator_only_roadmap_line_renders_a_locator(self) -> None:
        entry = RoadmapEntry(path=Path("a/overview.md"), label="a", text="entry")
        rendered = _render_roadmap_line(RoadmapLine(entry=entry, locator_only=True))
        assert "overview.md" in rendered

    def test_a_multi_line_roadmap_text_renders_as_one_list_item(self) -> None:
        entry = RoadmapEntry(
            path=Path("a/overview.md"), label="a", text="a\n\n## Prior decisions\n\n---"
        )
        rendered = _render_roadmap_line(RoadmapLine(entry=entry, locator_only=False))
        assert rendered == "- a ## Prior decisions ---"


def test_digest_footer_names_the_number_of_roadmap_entries_dropped_for_budget() -> None:
    # 100 entries at 5 words each render to 6 words a line, so the 300-word
    # roadmap share (30% of WORD_BUDGET) fits 50 in full and leaves nothing
    # for a locator tier — every entry past that is dropped outright.
    entries = [_entry(str(i), 5) for i in range(100)]
    digest = _assemble_digest(
        roadmap_entries=entries,
        ranked_hits=[],
        doc_frequency={},
        resolved_projects=[],
        skips=[],
        zero_related=False,
    )
    rendered = digest.render_markdown()
    assert digest.roadmap_dropped == 50
    assert "dropped for budget: 50" in rendered


def test_digest_footer_names_the_ranked_matches_the_budget_could_not_show() -> None:
    ranked_hits = [ScoredHit(hit=_hit(f"{i}.md", "word " * 60), score=1.0) for i in range(400)]
    digest = _assemble_digest(
        roadmap_entries=[],
        ranked_hits=ranked_hits,
        doc_frequency={},
        resolved_projects=[],
        skips=[],
        zero_related=False,
        ranked_omitted=7,
    )
    rendered = digest.render_markdown()
    assert digest.ranked_dropped == 400 - len(digest.ranked_lines)
    assert f"### {digest.ranked_dropped} further matches not shown (budget exhausted)" in rendered


def test_cap_omitted_units_are_named_apart_from_the_budget_exhausted_ones() -> None:
    # Nothing overflows the budget here, so any "budget exhausted" line would
    # be naming a cause that did not apply.
    digest = _assemble_digest(
        roadmap_entries=[],
        ranked_hits=[ScoredHit(hit=_hit("a.md", "word " * 5), score=1.0)],
        doc_frequency={},
        resolved_projects=[],
        skips=[],
        zero_related=False,
        ranked_omitted=50,
    )
    rendered = digest.render_markdown()
    assert digest.ranked_dropped == 0
    assert digest.ranked_omitted == 50
    assert "### 50 further matches not ranked (candidate cap)" in rendered
    assert "budget exhausted" not in rendered


def test_a_multi_line_skip_reason_renders_as_one_footer_list_item() -> None:
    digest = _assemble_digest(
        roadmap_entries=[],
        ranked_hits=[],
        doc_frequency={},
        resolved_projects=[],
        skips=[docs_search.Skip(locator="related", reason="config fault: line one\nline two")],
        zero_related=False,
    )
    footer = digest.render_markdown().split("### Skipped\n\n", 1)[1]
    assert footer.splitlines()[:1] == ["- related: config fault: line one line two"]


def test_the_two_halves_together_are_bounded_by_the_word_budget() -> None:
    entries = [_entry(str(i), 20) for i in range(100)]
    ranked_hits = [ScoredHit(hit=_hit(f"{i}.md", "word " * 60), score=1.0) for i in range(400)]
    digest = _assemble_digest(
        roadmap_entries=entries,
        ranked_hits=ranked_hits,
        doc_frequency={},
        resolved_projects=[],
        skips=[],
        zero_related=False,
    )
    rendered = digest.render_markdown()
    halves = rendered[: rendered.rindex("\n---\n")]
    # Each half can overshoot by at most the one locator line that crosses
    # its remaining share, so the pair is bounded well under twice the budget.
    assert len(halves.split()) <= WORD_BUDGET + 100


# ---------------------------------------------------------------------------
# Corpus-shaped invariants — asserted against a committed tree, then again
# against this repository's own
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SYNTHETIC_CORPUS = _REPO_ROOT / "tests" / "fixtures" / "docs_search_corpus"

# A roadmap entry is a folder label, an optional phase, and a one-line scope
# (FR-25 models 7-22 words); anything past this bound is a section body that
# escaped into the entry.
_ROADMAP_ENTRY_WORD_BOUND = 40

# The two halves are budgeted to WORD_BUDGET between them, and each half can
# overshoot by at most the width of the one locator line that crosses its
# remaining share. A locator is a provenance header, so 100 words covers two
# of them with room to spare — and unlike a multiple of the budget, this
# ceiling is close enough to the bound that an unbudgeted locator tier
# overruns it on a corpus of any size.
_DIGEST_WORD_CEILING = WORD_BUDGET + 100


class _CorpusInvariants:
    """Invariants that hold for any corpus, over whichever tree `_root` names.

    Nothing here asserts on what a corpus contains — only on the shape of
    what the handler makes of it, so the same five cases hold against a
    committed fixture and against a tree that changes every session.
    """

    def _root(self) -> Path:
        raise NotImplementedError

    def _run(self, capsys: pytest.CaptureFixture) -> tuple[DocsDigest, str]:
        handler = DocsSearchHandler(repo_root=self._root())
        digest = handler.search("observed failure ledger", related=False)
        return digest, capsys.readouterr().out

    def test_every_roadmap_entry_names_a_distinct_source_file(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        # The folder label is not an identity — the sanctioned layout permits
        # an overview.md beside a status.md — but the source file is.
        digest, _out = self._run(capsys)
        paths = {line.entry.path for line in digest.roadmap_lines}
        assert len(paths) == len(digest.roadmap_lines)

    def test_every_roadmap_entry_stays_within_the_entry_word_bound(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        digest, _out = self._run(capsys)
        oversized = [
            line.entry.text
            for line in digest.roadmap_lines
            if len(line.entry.text.split()) > _ROADMAP_ENTRY_WORD_BOUND
        ]
        assert oversized == []

    def test_no_absolute_repository_path_reaches_the_digest(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        _digest, out = self._run(capsys)
        assert str(self._root()) not in out

    def test_the_two_halves_together_stay_within_the_word_ceiling(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        _digest, out = self._run(capsys)
        halves = out[: out.rindex("\n---\n")]
        assert len(halves.split()) <= _DIGEST_WORD_CEILING

    def test_stdout_carries_exactly_the_two_documented_headings(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        _digest, out = self._run(capsys)
        headings = [line for line in out.splitlines() if line.startswith("## ")]
        assert headings == ["## Roadmap", "## Prior decisions"]


class TestSyntheticCorpus(_CorpusInvariants):
    """The five invariants against a committed tree, so no checkout skips them."""

    def _root(self) -> Path:
        return _SYNTHETIC_CORPUS

    def test_every_roadmap_entry_is_labelled_by_its_path_below_the_project_root(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        # Pinned as literal data rather than as a distinctness count: the
        # fixture holds two goals that each carry a milestone-01-core, so a
        # label built from one path component alone collapses four labels
        # into three and still passes any count-based assertion.
        digest, _out = self._run(capsys)
        assert [line.entry.label for line in digest.roadmap_lines] == [
            "planning/goal-alpha/milestone-01-core",
            "planning/goal-alpha",
            "planning/goal-alpha",
            "planning/goal-beta/milestone-01-core",
            "planning/goal-beta",
        ]

    def test_the_fixture_exercises_every_tier_the_invariants_bound(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        # An invariant asserted over a corpus that never reaches a tier is a
        # guard over nothing, and this fixture is the one input under our
        # control — so its shape is pinned alongside the invariants.
        digest, _out = self._run(capsys)
        assert digest.roadmap_lines
        assert len({line.entry.path.parent for line in digest.roadmap_lines}) < len(
            digest.roadmap_lines
        )
        assert any(line.locator_only for line in digest.ranked_lines)
        assert digest.ranked_dropped > 0
        assert digest.ranked_lines[0].scored.hit.kind == "failure"


class TestRealPlanningCorpus(_CorpusInvariants):
    """The same five invariants against this repository's own planning tree.

    The tree is gitignored, so this pass skips on a fresh checkout; the
    committed fixture above is what keeps the invariants guarded there.
    """

    def _root(self) -> Path:
        planning = _REPO_ROOT / "planning"
        if not planning.is_dir() or not any(planning.rglob("*.md")):
            pytest.skip("this repository has no planning/ tree to index")
        return _REPO_ROOT
