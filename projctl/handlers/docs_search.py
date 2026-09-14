# pylint: disable=too-many-lines
# One feature, one import site: nothing outside `search docs` uses any of it,
# so splitting the five components into modules would buy separate files and
# no reuse. cli.py carries the same suppression for the same reason.
"""Local docs/planning corpus search — network-free, platform-independent.

`projctl search docs "<query>" [--related]` ranks Markdown sections from the
current repository's `planning/` tree and docs root (and, under `--related`,
every project declared in `search.related`) and renders a bounded two-part
Markdown digest: `## Roadmap` (unranked, from `overview.md`/`status.md`) then
`## Prior decisions` (BM25-ranked). No Config gate, no platform dispatch, no
network — matching `ActivityHandler`'s precedent (see its module docstring).

Five components, one direction of flow (see design.md §4/§5):
CorpusResolver (which directories) -> SectionExtractor (which units) ->
Bm25Ranker (in what order) + RoadmapIndex (what is live) -> DocsDigest
(what fits).
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple, Optional

from ..config import PROJECT_LOCAL_CONFIG_NAMES, Config, ConfigurationError, SearchConfig
from ..utils.git_helpers import get_repo_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ranking policy — every tunable constant in one block, with a shape
# assertion beside it (the pattern RSYNC_EXCLUDES / _assert_exclude_shape
# establishes in handlers/sync.py). Changing a value here is a behaviour
# change; §6 of design.md names which of them a test pins.
# ---------------------------------------------------------------------------

BM25_K1 = 1.2
BM25_B = 0.75
HEADING_BOOST = 2.5
FILE_DECAY = 0.5
DIRECTORY_DECAY = 0.75
WORD_BUDGET = 1000
ROADMAP_SHARE = 0.30
PRIOR_DECISIONS_SHARE = 0.70
CASE_EXACT_MIN_LEN = 2
CASE_EXACT_MAX_LEN = 5
ROADMAP_SCOPE_WORD_CAP = 25
RANKED_CANDIDATE_CAP = 200


def _assert_ranking_policy_shape() -> None:
    """Validate the ranking-policy block's own invariants at import time.

    A future edit leaving the two shares out of sync, or picking a
    non-positive decay/budget, would silently misbudget every digest instead
    of failing loudly — this is RSYNC_EXCLUDES's shape-assertion pattern
    applied to this block. It raises rather than asserting so `python -O`
    cannot strip the guard the docstring above claims.

    Raises:
        ValueError: If any constant in the block falls outside its range, or
            the two shares do not sum to 1.0.
    """
    checks = (
        (BM25_K1 > 0.0, "BM25_K1 must be positive"),
        (0.0 <= BM25_B <= 1.0, "BM25_B must be in [0, 1]"),
        (HEADING_BOOST > 1.0, "HEADING_BOOST must amplify, not shrink or leave unchanged"),
        (0.0 < FILE_DECAY < 1.0, "FILE_DECAY must be a genuine decay factor"),
        (0.0 < DIRECTORY_DECAY < 1.0, "DIRECTORY_DECAY must be a genuine decay factor"),
        (WORD_BUDGET > 0, "WORD_BUDGET must be positive"),
        (
            abs((ROADMAP_SHARE + PRIOR_DECISIONS_SHARE) - 1.0) < 1e-9,
            "ROADMAP_SHARE and PRIOR_DECISIONS_SHARE must sum to 1.0",
        ),
        (1 <= CASE_EXACT_MIN_LEN <= CASE_EXACT_MAX_LEN, "case-exact window must be non-empty"),
        (ROADMAP_SCOPE_WORD_CAP > 0, "ROADMAP_SCOPE_WORD_CAP must be positive"),
        (RANKED_CANDIDATE_CAP > 0, "RANKED_CANDIDATE_CAP must be positive"),
    )
    for holds, message in checks:
        if not holds:
            raise ValueError(f"ranking policy is misconfigured: {message}")


_assert_ranking_policy_shape()

# Template-vocabulary headings whose heading BOOST is suppressed — indexing
# is never affected (FR-17). Seeded from DESIGN-TEMPLATE.md's own section
# headings (the dominant heading source measured across the corpus — see
# analysis.md's heading-frequency table) plus the review-report vocabulary
# those templates produce, plus the three severity words. Entries are stored
# already normalized (stripped of "N. " numbering, case-folded) — see
# _normalize_heading, which applies the same transform to a candidate heading
# before comparing.
TEMPLATE_HEADING_SUPPRESSIONS: frozenset[str] = frozenset(
    {
        "problem statement",
        "goals and non-goals",
        "goals",
        "non-goals",
        "implementation context",
        "architecture overview",
        "detailed design",
        "test requirements",
        "unit tests",
        "integration tests",
        "e2e tests",
        "tests not written",
        "trade-offs and alternatives",
        "open questions",
        "ticket constraints",
        "recommendation",
        "summary",
        "findings by severity",
        "verification gaps",
        "requirement coverage",
        "review output",
        "high",
        "medium",
        "low",
    }
)


def _assert_template_heading_shape(headings: frozenset[str]) -> None:
    """Validate every suppression entry is a bare, already-normalized heading string.

    An entry carrying leading "N. " numbering or mixed case would never equal
    the normalized heading text the ranker compares it against, silently
    disabling suppression for that heading. Raises rather than asserting so
    `python -O` cannot strip the guard.

    Raises:
        ValueError: If any entry is empty, carries "N. " numbering, or is
            not already case-folded.
    """
    for heading in headings:
        if not isinstance(heading, str) or not heading:
            raise ValueError("suppression entries must be non-empty strings")
        if re.match(r"^\d+\.\s", heading):
            raise ValueError(
                f"{heading!r} carries leading numbering — store the normalized "
                "(post-strip) form; matching strips a candidate heading's own "
                "numbering, not the constant's"
            )
        if heading != heading.casefold():
            raise ValueError(f"{heading!r} must already be case-folded")


_assert_template_heading_shape(TEMPLATE_HEADING_SUPPRESSIONS)

_ALTERNATIVE_HEADINGS: frozenset[str] = frozenset(
    {"trade-offs and alternatives", "goals and non-goals", "goals", "non-goals"}
)
_CONSTRAINT_HEADINGS: frozenset[str] = frozenset({"ticket constraints"})

_VALID_KINDS: frozenset[str] = frozenset(
    {"failure", "alternative", "constraint", "docs", "roadmap", "untyped"}
)

# Per-kind policy, named rather than spelled as literals at each consuming
# site (FR-11/§5.4). A sixth kind must be named by at least one of these four
# sets, which _assert_kind_policy_shape enforces at import — a literal
# membership test would have let it be silently dropped by whichever site
# never learned to ask for it. The sets overlap where a kind carries two
# policies: `failure` both bears fields and pins, `roadmap` both bears fields
# and folds into an entry.
_FIELD_BEARING_KINDS: frozenset[str] = frozenset({"failure", "roadmap"})
_PINNABLE_KINDS: frozenset[str] = frozenset({"failure"})
_ROADMAP_ENTRY_KINDS: frozenset[str] = frozenset({"roadmap"})
_PLAIN_KINDS: frozenset[str] = frozenset({"alternative", "constraint", "docs", "untyped"})


def _assert_kind_policy_shape() -> None:
    """Validate every kind in the closed set is named by at least one policy set.

    Raises:
        ValueError: If a policy set names a kind outside _VALID_KINDS, or a
            valid kind is named by none of them.
    """
    assigned = _FIELD_BEARING_KINDS | _ROADMAP_ENTRY_KINDS | _PINNABLE_KINDS | _PLAIN_KINDS
    unknown = assigned - _VALID_KINDS
    if unknown:
        raise ValueError(f"kind policy names values outside the closed set: {sorted(unknown)}")
    unassigned = _VALID_KINDS - assigned
    if unassigned:
        raise ValueError(f"kind policy leaves values undecided: {sorted(unassigned)}")


_assert_kind_policy_shape()


def _validate_kind(kind: str) -> str:
    """Return `kind` unchanged, or raise if it falls outside the closed discriminant set.

    Raises:
        ValueError: If `kind` is not one of _VALID_KINDS.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"unrecognized Hit.kind: {kind!r}")
    return kind


_NUMBERING_PREFIX_RE = re.compile(r"^\d+\.\s+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
# CommonMark: a fence opens only at an indent of 0-3 (4 makes it an indented
# code block), and everything after the marker is the info string.
_FENCE_RE = re.compile(r"^ {0,3}(?P<marker>`{3,}|~{3,})(?P<info>.*)$")
_BOLD_FIELD_RE = re.compile(r"^\*\*([^*:]+):\*\*\s*(.*)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_LIST_MARKER_RE = re.compile(r"^(?:[-*+]|\d+[.)])(?:\s+|$)")
# A setext underline of any length, and every thematic-break spelling: both
# read as digest structure to a consumer splitting on headings or on the
# footer's own `---` boundary (FR-35). CommonMark lets any run of spaces or
# tabs separate the three break characters, so the separator is matched as a
# run rather than as the single space an earlier spelling of this pattern
# demanded.
_SETEXT_RULE_RE = re.compile(r"^(?:-+|=+|(?:[-*_][ \t]*){3,})[ \t]*$")
_STRIP_CHARS = ".,;:?()[]{}\"'`"


def _normalize_heading(text: str) -> str:
    """Strip a leading "N. " numbering prefix and case-fold, for exact heading comparisons."""
    return _NUMBERING_PREFIX_RE.sub("", text.strip()).casefold()


def _relative_display(path: Path, base: Path) -> str:
    """Render `path` relative to `base`, falling back to the absolute form.

    Every locator in the digest goes through here: the repository name is
    already rendered beside it, so an absolute prefix is both noise against a
    budget and a leak of the operator's home-directory layout into a document
    written to be pasted elsewhere (§5.6).
    """
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def _footer_locator(path: Path, base: Path) -> str:
    """Render a footer locator relative to `base`, never as an absolute path.

    `_relative_display`'s absolute fallback is safe only where a project root
    is rendered beside the locator to qualify it. A declared related project
    or a configured docs_path can name a path outside any project the digest
    knows, and there its absolute form is the operator's home-directory
    layout in a document written to be pasted elsewhere (§5.6) — so such a
    path degrades to its final component instead.
    """
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        pass
    if not path.is_absolute():
        return path.as_posix()
    return path.name or path.anchor


def _declared_locator(declared: str, base: Path) -> str:
    """Render a `search.related` entry for the footer without an absolute prefix.

    The entry is operator-authored text, so `~` may name a home that does not
    resolve on this machine; that is the very fault being reported, and it
    must not become a second one.
    """
    try:
        candidate = Path(declared).expanduser()
    except RuntimeError:
        candidate = Path(declared)
    return _footer_locator(candidate, base)


def _fault_detail(exc: BaseException, base: Path) -> str:
    """Render an exception for a footer reason, carrying no absolute path (§5.6).

    `OSError.__str__` appends the filename exactly as the failing call
    received it, which is always absolute here, and the locator beside the
    reason already names the file — so only errno and strerror are kept.
    Every other exception is rendered as written, with any path below `base`
    made relative: the alternative is an enumerated exception set, which
    §5.3 rejects for this contract.
    """
    if isinstance(exc, OSError):
        detail = exc.strerror or type(exc).__name__
        return f"[Errno {exc.errno}] {detail}" if exc.errno is not None else str(detail)
    return str(exc).replace(f"{base}{os.sep}", "").replace(str(base), ".")


def _classify_kind(corpus: str, filename: str, heading_chain: list[str]) -> str:
    """Assign one section's kind from corpus, then filename, then heading ancestry (§5.4).

    Tree layout is never consulted — only these three inputs decide kind.
    """
    normalized_chain = {_normalize_heading(h) for h in heading_chain}

    kind = "untyped"
    if corpus == "docs":
        kind = "docs"
    elif filename == "observed-failures.md":
        kind = "failure"
    elif filename in ("overview.md", "status.md"):
        kind = "roadmap"
    elif filename == "design.md" and normalized_chain & _ALTERNATIVE_HEADINGS:
        kind = "alternative"
    elif filename == "analysis.md" and normalized_chain & _CONSTRAINT_HEADINGS:
        kind = "constraint"
    return _validate_kind(kind)


# ---------------------------------------------------------------------------
# Tokenisation and matching (FR-14)
# ---------------------------------------------------------------------------


def tokenize(query: str) -> list[str]:
    """Split a query on whitespace, trimming prose-wrapper punctuation per token.

    A term holding no alphanumeric character (e.g. ".*") is taken verbatim —
    stripping it could remove its entire meaning. Every other term loses
    leading/trailing characters from the prose-wrapper strip set. Neither
    rule can empty a term.
    """
    tokens = []
    for term in query.split():
        if any(ch.isalnum() for ch in term):
            tokens.append(term.strip(_STRIP_CHARS))
        else:
            tokens.append(term)
    return tokens


def _compile_token_pattern(token: str) -> re.Pattern[str]:
    """Compile one token into a literal, word-bounded pattern.

    A token 2-5 characters long carrying no lowercase letter matches
    case-exact (acronyms like "CAN"/"OTA"); every other token case-folds.
    `(?<!\\w)`/`(?!\\w)` replace `\\b`, which after a trailing "+" (e.g.
    "C++") demands a following word character and so never fires.
    """
    case_exact = CASE_EXACT_MIN_LEN <= len(token) <= CASE_EXACT_MAX_LEN and not any(
        ch.islower() for ch in token
    )
    flags = 0 if case_exact else re.IGNORECASE
    return re.compile(r"(?<!\w)" + re.escape(token) + r"(?!\w)", flags)


def _heading_match_state(
    heading_path: list[str], patterns: list[re.Pattern[str]]
) -> tuple[bool, bool]:
    """Return (a non-template heading matched, a match was suppressed) for this chain.

    The two are reported separately because they answer different support
    questions: a unit with no heading match and one whose only match is
    template vocabulary both score unboosted, and only the log distinguishes
    them (NFR-9). A non-template match qualifies the chain for the boost; it
    does not by itself apply it — see Bm25Ranker._score_parts.
    """
    applies = False
    suppressed = False
    for heading in heading_path:
        if not any(p.search(heading) for p in patterns):
            continue
        if _normalize_heading(heading) in TEMPLATE_HEADING_SUPPRESSIONS:
            suppressed = True
        else:
            applies = True
    return applies, suppressed


class _TermEvidence(NamedTuple):
    """One (token, section) cell: the term frequency BM25 reads, and where it came from.

    `in_body` is what separates a floored own-heading match from a single body
    occurrence — both are `tf == 1`. The boost is decided once per hit rather
    than per cell: Bm25Ranker._score_parts withholds it only where no token at
    all matched in the body, so a floored cell beside another token's body
    match keeps it. A tuple rather than a dataclass because one is built per
    token per section across the whole corpus (NFR-1).
    """

    tf: int
    in_body: bool


def _term_evidence(pattern: re.Pattern[str], hit: "Hit") -> _TermEvidence:
    """Return the term frequency BM25 reads for one token in one hit, and its source.

    A match in the section's **own** heading floors at 1 rather than scoring
    zero, so a section headed with a query token is rankable rather than
    filtered out (FR-16). An ancestor-only match earns no floor: every file
    has an H1, so flooring on the joined chain admits every section of any
    file whose title carries the token — collapsing idf and letting BM25's
    length normalisation put empty sections above sections that actually
    discuss it (FR-15). The boost still reads the whole chain; only admission
    narrows. A token present in the body scores that count alone — the floor
    never stacks.
    """
    tf = len(pattern.findall(hit.body))
    if tf:
        return _TermEvidence(tf, True)
    own_heading = hit.heading_path[-1] if hit.heading_path else ""
    matched_heading = bool(own_heading) and pattern.search(own_heading) is not None
    return _TermEvidence(1, False) if matched_heading else _TermEvidence(0, False)


# ---------------------------------------------------------------------------
# Extraction — Hit, Skip, SectionExtractor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Skip:
    """A path or project that could not be used, with the reason why (for the digest footer)."""

    locator: str
    reason: str


@dataclass(frozen=True)
class Hit:
    """One heading-scoped Markdown section, with its classification and provenance.

    `kind` is the closed discriminant {failure, alternative, constraint,
    docs, roadmap, untyped}; ranking and rendering switch exhaustively on it
    and raise on an unrecognized value. `fields` is populated only for
    `failure` and `roadmap` kind sections. `project_root` is the base every
    rendered path is made relative to, so no digest carries an operator's
    home-directory layout.
    """

    # pylint: disable=too-many-instance-attributes
    # Provenance (FR-22) is five of these nine on its own — repo, platform,
    # kind, path, heading chain — and splitting it out would put the rendered
    # locator one indirection away from the unit it locates.

    path: Path
    kind: str
    heading_path: list[str]
    body: str
    cost_words: int
    fields: dict[str, str]
    repo: str
    platform: str
    project_root: Path


def _read_bold_fields(body: str) -> dict[str, str]:
    """Extract `**Name:** value` lines anchored at the start of a line.

    "Status::sigterm" in prose does not match — the leading "**" anchor is
    what separates the 139 real observed-failure records from the 144 lines
    a bare "Status:" grep would return.
    """
    fields: dict[str, str] = {}
    for text_line in body.splitlines():
        match = _BOLD_FIELD_RE.match(text_line)
        if match:
            fields[match.group(1).strip()] = match.group(2).strip()
    return fields


def _split_sections(text: str) -> tuple[list[tuple[list[str], str]], bool]:
    """Split Markdown text into (heading_path, body) pairs, fence-aware.

    A heading-like line inside a fenced block is body text, not a heading —
    the fence exclusion is the unit boundary, not a refinement of it. Content
    before the first heading, where non-empty, is one section with an empty
    heading chain. A headingless file with any content is wholly one section.

    A block opened with N or more of one character, at an indent of at most
    3, closes only on that same character repeated at least N times and
    followed by nothing but whitespace. Treating any fence line as a toggle
    makes a file mixing ``` and ~~~ swallow every heading below the first
    one; treating a line carrying an info string as a close invents a heading
    out of code-block content, since the text below it is then read as
    document structure.

    One divergence from CommonMark remains, in the opening info string:
    CommonMark forbids a backtick there in a backtick fence, making such a
    line ordinary text, while any info string opens a fence here. The cost is
    a file reported as holding an unclosed fence when it holds none — a named
    loss in the footer rather than a silent one, which is why the simpler
    rule is kept.

    Returns:
        (sections, fence_left_open) — the second is True when the text ended
        inside a fenced block, which means every heading after that fence was
        absorbed as body text.
    """
    stack: list[tuple[int, str]] = []
    heading_path: list[str] = []
    body_lines: list[str] = []
    open_fence: Optional[tuple[str, int]] = None
    started = False
    sections: list[tuple[list[str], str]] = []

    def flush() -> None:
        body_text = "\n".join(body_lines).strip("\n")
        if started or body_text.strip():
            sections.append((list(heading_path), body_text))

    for raw_line in text.splitlines():
        fence_match = _FENCE_RE.match(raw_line)
        if fence_match:
            marker = fence_match.group("marker")
            if open_fence is None:
                open_fence = (marker[0], len(marker))
            elif (
                marker[0] == open_fence[0]
                and len(marker) >= open_fence[1]
                and not fence_match.group("info").strip()
            ):
                open_fence = None
            body_lines.append(raw_line)
            continue

        if open_fence is None:
            heading_match = _HEADING_RE.match(raw_line)
            if heading_match:
                flush()
                level = len(heading_match.group(1))
                stack = [item for item in stack if item[0] < level]
                stack.append((level, heading_match.group(2)))
                heading_path = [heading_text for _, heading_text in stack]
                body_lines = []
                started = True
                continue

        body_lines.append(raw_line)

    flush()
    return sections, open_fence is not None


class SectionExtractor:
    """Splits Markdown files into heading-scoped, kind-classified sections.

    One call per file returns every unit in it (FR-11); this is the only
    component that reads file content.
    """

    def __init__(self) -> None:
        self._skips: list[Skip] = []
        self._read_faults = 0

    def extract(
        self, path: Path, *, corpus: str, repo: str, platform: str, project_root: Path
    ) -> list[Hit]:
        """Return every section in `path`, or none (with a recorded Skip) on a read fault.

        A file that cannot be stat'ed, opened, or decoded as UTF-8 is skipped
        and the walk continues (FR-36) — reachable rather than theoretical,
        since Markdown files are otherwise assumed UTF-8 throughout the
        package. A file left inside an unclosed fence is indexed as far as it
        parsed and named in the footer, so the absorbed headings are a visible
        loss rather than a silent one.
        """
        locator = _relative_display(path, project_root)
        try:
            path.stat()
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self._read_faults += 1
            self._record_skip(locator, f"unreadable: {_fault_detail(exc, project_root)}")
            return []

        sections, fence_left_open = _split_sections(text)
        if fence_left_open:
            self._record_skip(locator, "unclosed code fence — headings below it were not indexed")

        filename = path.name
        hits = []
        for section_heading_path, body in sections:
            kind = _classify_kind(corpus, filename, section_heading_path)
            fields = _read_bold_fields(body) if kind in _FIELD_BEARING_KINDS else {}
            hits.append(
                Hit(
                    path=path,
                    kind=kind,
                    heading_path=section_heading_path,
                    body=body,
                    cost_words=len(body.split()),
                    fields=fields,
                    repo=repo,
                    platform=platform,
                    project_root=project_root,
                )
            )
        return hits

    def _record_skip(self, locator: str, reason: str) -> None:
        self._skips.append(Skip(locator=locator, reason=reason))
        logger.warning("search docs: skipping %s — %s", locator, reason)

    def skips(self) -> list[Skip]:
        """Return every file skipped so far, each with its reason."""
        return list(self._skips)

    def read_faults(self) -> int:
        """Return how many files faulted before a single section could be read (FR-36).

        A file left inside an unclosed fence is not one of them: it was
        indexed as far as it parsed, so the count separates "walked but never
        read" from "read and found to be partial" (§5.6).
        """
        return self._read_faults


# ---------------------------------------------------------------------------
# Ranking — Bm25Ranker, ScoredHit
# ---------------------------------------------------------------------------

_LEDGER_OPEN_VALUE = "open"


def _is_open_failure(hit: Hit) -> bool:
    """Return whether this failure-kind hit's Status: reads "open" (up to its first em dash)."""
    if hit.kind not in _PINNABLE_KINDS:
        return False
    status = hit.fields.get("Status")
    if not status:
        return False
    value = status.split("—", 1)[0].strip().casefold()
    return value == _LEDGER_OPEN_VALUE


@dataclass(frozen=True)
class ScoredHit:
    """One Hit with its final BM25 x boost x diversity-decay score."""

    hit: Hit
    score: float


@dataclass(frozen=True)
class _ScoreParts:
    """One hit's score broken into the components NFR-9's debug channel reports.

    `heading_match` and `heading_suppressed` describe the heading chain on its
    own; `body_matched` is what decides between a qualifying chain boosting
    the score and the boost being withheld.
    """

    bm25: float
    heading_match: bool
    heading_suppressed: bool
    body_matched: bool
    total: float


def _boost_state(parts: _ScoreParts) -> str:
    """Name the heading-boost outcome for the debug channel.

    Four outcomes, not three: a reader tracing why a unit ranked where it did
    has to tell a boost withheld from a unit whose only match is its own
    heading apart from one suppressed as template vocabulary, since the two
    have different remedies.
    """
    if parts.heading_match:
        return "applied" if parts.body_matched else "floor-only"
    return "suppressed" if parts.heading_suppressed else "none"


class Bm25Ranker:
    """BM25 over heading-scoped sections, with heading-boost, suppression, the
    open-failure pin, and greedy per-file/per-directory diversity decay (§5.5).
    """

    def __init__(self, hits: list[Hit], query: str) -> None:
        for hit in hits:
            _validate_kind(hit.kind)
        self._hits = list(hits)
        # A token repeated in the query would otherwise contribute its term
        # twice, doubling its weight against the rest of the query while the
        # footer's frequency table still reports it once.
        self._tokens = list(dict.fromkeys(tokenize(query)))
        self._patterns = [_compile_token_pattern(t) for t in self._tokens]
        self._avg_doc_len = (
            sum(h.cost_words for h in self._hits) / len(self._hits) if self._hits else 0.0
        )
        # One tf table, three consumers: the document frequency, the match
        # filter, and the score all have to read the identical text, or the
        # frequency the footer reports stops being the one idf used (FR-19).
        # Computing it once also removes two of the three full-corpus regex
        # passes the ranker used to make (NFR-1).
        self._tf_table = [[_term_evidence(p, hit) for p in self._patterns] for hit in self._hits]
        self._doc_freq = self._compute_doc_frequency()
        self._omitted = 0

    def doc_frequency(self) -> dict[str, int]:
        """Return each query token's document frequency — the section count sharing the BM25 idf term."""
        return dict(self._doc_freq)

    def omitted(self) -> int:
        """Return how many unpinned matched units the candidate cap dropped before ranking."""
        return self._omitted

    def _compute_doc_frequency(self) -> dict[str, int]:
        freq: dict[str, int] = {}
        for index, token in enumerate(self._tokens):
            freq[token] = sum(1 for row in self._tf_table if row[index].tf > 0)
        return freq

    def _score_parts(self, index: int) -> _ScoreParts:
        hit = self._hits[index]
        n = len(self._hits)
        bm25 = 0.0
        for token_index, token in enumerate(self._tokens):
            tf = self._tf_table[index][token_index].tf
            if tf == 0:
                continue
            n_t = self._doc_freq.get(token, 0)
            idf = math.log(1 + (n - n_t + 0.5) / (n_t + 0.5))
            norm_len = hit.cost_words / self._avg_doc_len if self._avg_doc_len else 0.0
            length_norm = 1 - BM25_B + BM25_B * norm_len
            bm25 += idf * (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * length_norm)
        heading_match, heading_suppressed = _heading_match_state(hit.heading_path, self._patterns)
        # A unit whose body holds no query token is here only because its own
        # heading floored tf at 1, so the heading is already the whole of its
        # score — boosting it would count the same match twice (FR-16). The
        # floor itself stays: withholding the boost keeps such a unit
        # rankable, which is what the floor exists to guarantee.
        body_matched = any(cell.in_body for cell in self._tf_table[index])
        return _ScoreParts(
            bm25=bm25,
            heading_match=heading_match,
            heading_suppressed=heading_suppressed,
            body_matched=body_matched,
            total=bm25 * HEADING_BOOST if heading_match and body_matched else bm25,
        )

    def rank(self) -> list[ScoredHit]:
        """Return the retained matched Hits, decay-diversified then pin-reordered.

        Diversification is O(n^2) in its input, so the **unpinned** candidate
        set is capped at RANKED_CANDIDATE_CAP by base score first — several
        times what the half's budget can render, and the difference is
        ordering work whose result the budget discards. Open-failure units
        are partitioned out before the cap applies: FR-16 puts them ahead of
        every other unit whatever their score, so a cap that could discard
        one would make the guarantee conditional on exactly what it
        overrides. omitted() reports the remainder so the footer can name it.
        """
        matched = [
            (self._hits[index], self._score_parts(index))
            for index, row in enumerate(self._tf_table)
            # Not `any(row)` — every cell is a non-empty tuple and so truthy
            # whatever count it carries, which would put every unit in the
            # corpus on the NFR-9 score channel instead of every *matched*
            # one, diluting the one signal that answers "why did the document
            # I expected not appear".
            if any(cell.tf for cell in row)
        ]
        scored = sorted(
            ((hit, parts) for hit, parts in matched if parts.total > 0.0),
            key=lambda pair: pair[1].total,
            reverse=True,
        )
        pinned = [pair for pair in scored if _is_open_failure(pair[0])]
        unpinned = [pair for pair in scored if not _is_open_failure(pair[0])]
        self._omitted = max(0, len(unpinned) - RANKED_CANDIDATE_CAP)

        candidates = pinned + unpinned[:RANKED_CANDIDATE_CAP]
        ordered = _greedy_diversify([(hit, parts.total) for hit, parts in candidates])

        self._log_score_components(matched, ordered)

        return [sh for sh in ordered if _is_open_failure(sh.hit)] + [
            sh for sh in ordered if not _is_open_failure(sh.hit)
        ]

    @staticmethod
    def _log_score_components(
        matched: list[tuple[Hit, _ScoreParts]], ordered: list[ScoredHit]
    ) -> None:
        """Log every matched unit's score components, selected or not (NFR-9).

        The support question is "why did the document I expected not appear",
        so a unit the ranker dropped is exactly the one that must be logged.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        # Hit carries list and dict fields, so it is unhashable; identity is
        # what keys a selected unit back to its decayed score.
        effective = {id(sh.hit): sh.score for sh in ordered}
        for hit, parts in matched:
            final = effective.get(id(hit))
            decay = final / parts.total if final is not None and parts.total else None
            logger.debug(
                "search docs: score %s §%s bm25=%.4f boost=%s pin=%s decay=%s final=%s",
                _relative_display(hit.path, hit.project_root),
                " › ".join(hit.heading_path),
                parts.bm25,
                _boost_state(parts),
                _is_open_failure(hit),
                "-" if decay is None else f"{decay:.3g}",
                "not selected" if final is None else f"{final:.3g}",
            )


def _greedy_diversify(candidates: list[tuple[Hit, float]]) -> list[ScoredHit]:
    """Greedily select in relevance order, applying per-file and per-directory decay (FR-18).

    Each already-picked section discounts its own file's remaining sections
    x FILE_DECAY; each already-picked file discounts every other file in its
    immediate parent directory x DIRECTORY_DECAY. Both compound. No file is
    capped — several top-scoring sections in one file can all be selected.
    """
    remaining = list(candidates)
    file_picks: dict[Path, int] = {}
    dir_picked_files: dict[Path, set[Path]] = {}
    ordered: list[ScoredHit] = []

    while remaining:
        best_idx = 0
        best_effective = -1.0
        for idx, (hit, base_score) in enumerate(remaining):
            picked_in_dir = dir_picked_files.get(hit.path.parent)
            other_files_in_dir = 0
            if picked_in_dir:
                other_files_in_dir = len(picked_in_dir) - (1 if hit.path in picked_in_dir else 0)
            effective = (
                base_score
                * (FILE_DECAY ** file_picks.get(hit.path, 0))
                * (DIRECTORY_DECAY**other_files_in_dir)
            )
            if effective > best_effective:
                best_effective = effective
                best_idx = idx
        hit, _base_score_value = remaining.pop(best_idx)
        ordered.append(ScoredHit(hit=hit, score=best_effective))
        file_picks[hit.path] = file_picks.get(hit.path, 0) + 1
        dir_picked_files.setdefault(hit.path.parent, set()).add(hit.path)

    return ordered


# ---------------------------------------------------------------------------
# Roadmap — RoadmapEntry, RoadmapIndex
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoadmapEntry:
    """One overview.md/status.md file's roadmap entry: folder, phase, and one-line scope."""

    path: Path
    label: str
    text: str


def _first_content_line(text: str) -> str:
    """Return the first line carrying content beyond a list marker, or ""."""
    for line in text.splitlines():
        candidate = _LIST_MARKER_RE.sub("", line.strip()).strip()
        if candidate:
            return candidate
    return ""


def _first_scope_line(text: str) -> str:
    """Return a one-line scope for a roadmap entry, or "" for empty input.

    The first sentence where the text has one; otherwise the first line
    carrying content, which is the ordinary shape of a bullet-list `## About`
    — there a sentence split returns the whole section, and FR-25's entry is
    one line of scope, not a section body. Both branches are capped, because
    a handful of absorbed sections exhaust the roadmap share on their own.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    first = _SENTENCE_SPLIT_RE.split(stripped, maxsplit=1)[0].strip()
    if "\n" in first:
        first = _first_content_line(stripped)
    else:
        first = _LIST_MARKER_RE.sub("", first).strip() or _first_content_line(stripped)
    words = first.split()
    if len(words) > ROADMAP_SCOPE_WORD_CAP:
        return " ".join(words[:ROADMAP_SCOPE_WORD_CAP]) + " …"
    return " ".join(words)


class RoadmapIndex:
    """Folds one caller-repo file's roadmap-kind Hits into one entry per file (FR-25)."""

    # pylint: disable=too-few-public-methods
    # A single-responsibility fold with one public entry point (entries()),
    # matching ActivityHandler's precedent for this shape. Scoped to the class
    # body: a bare module-scope disable would exempt every class below it too.

    def entries(self, hits: list[Hit], caller_root: Path) -> list[RoadmapEntry]:
        """Return one RoadmapEntry per overview.md/status.md file under `caller_root`.

        A related project's roadmap-kind hits are dropped — the roadmap half
        covers the caller alone; its own sections still rank in the ranked
        half via Bm25Ranker. Ownership is the hit's own project root, not
        path descent: a related project declared *below* the caller's
        repository passes a containment test and would otherwise contribute
        entries against FR-25.
        """
        caller = caller_root.resolve()
        by_path: dict[Path, list[Hit]] = {}
        for hit in hits:
            if hit.kind not in _ROADMAP_ENTRY_KINDS:
                continue
            if hit.project_root.resolve() != caller:
                continue
            by_path.setdefault(hit.path, []).append(hit)

        result = []
        for path in sorted(by_path, key=lambda p: p.as_posix()):
            result.append(self._fold_file(path, by_path[path]))
        return result

    @staticmethod
    def _fold_file(path: Path, file_hits: list[Hit]) -> RoadmapEntry:
        phase: Optional[str] = None
        about_scope: Optional[str] = None
        plain_scope: Optional[str] = None
        for hit in file_hits:
            if phase is None:
                phase = hit.fields.get("State") or hit.fields.get("Status")
            own_heading = _normalize_heading(hit.heading_path[-1]) if hit.heading_path else ""
            if own_heading == "about" and about_scope is None:
                about_scope = _first_scope_line(hit.body)
            elif own_heading == "scope" and plain_scope is None:
                plain_scope = _first_scope_line(hit.body)

        scope = about_scope or plain_scope
        # The folder's own name repeats across goals in every sanctioned tree
        # shape (milestone-XX-<name> under each), so the path below the
        # project root is what makes one entry distinguishable from another.
        label = _relative_display(path.parent, file_hits[0].project_root)
        text = label
        if phase:
            text = f"{text} — {phase}"
        if scope:
            text = f"{text}: {scope}"
        return RoadmapEntry(path=path, label=label, text=text)


# ---------------------------------------------------------------------------
# Corpus resolution — CorpusRoot, ResolvedProject, CorpusResolver
# ---------------------------------------------------------------------------


_ORIGIN_CALLER = "caller"
_ORIGIN_DECLARED = "declared entry"


@dataclass(frozen=True)
class CorpusRoot:
    """One resolved planning-or-docs directory belonging to one project.

    `origin` names how the project was reached — the caller's own repository
    or a `search.related` entry — which the debug channel reports alongside
    the resolved path (§5.7).
    """

    repo: str
    platform: str
    corpus: str  # "planning" | "docs"
    project_root: Path
    root: Path
    origin: str


@dataclass(frozen=True)
class ResolvedProject:
    """One project's resolved-corpus footer summary — present even with zero roots.

    A project present with neither corpus renders with an empty root_kinds
    list, which is what keeps it distinguishable in the footer from a
    project that was skipped outright. `indexed_files` separates a corpus
    that was walked and held nothing from one that held 300 files — the
    digest's consumer has no other error channel (§5.6).

    `project_root` is the identity, not `repo`: `repo` is the directory
    basename, and two declared projects can share one. Keying on the
    basename merges their footer rows and attributes one project's platform
    to both (FR-22).
    """

    repo: str
    platform: str
    project_root: Path
    root_kinds: list[str] = field(default_factory=list)
    indexed_files: int = 0


def _expand_docs_path(docs_path: str) -> Path:
    """Expand `~` in a configured docs_path, naming the key when it cannot be expanded.

    `Path.expanduser()` raises RuntimeError for a `~unknownuser/` prefix.
    Converting it here keeps the caller's own fault an exit-1 error naming
    the offending key, and lets a related project's identical fault be
    contained as a skip by that hop's own guard (FR-27).

    Raises:
        ConfigurationError: If the leading `~` names no resolvable home.
    """
    try:
        return Path(docs_path).expanduser()
    except RuntimeError as exc:
        raise ConfigurationError(
            f"search.docs_path {docs_path!r} cannot be expanded: {exc}"
        ) from exc


def _probe_project_config(root: Path) -> Optional[Path]:
    """Return the first of PROJECT_LOCAL_CONFIG_NAMES that is a file under `root`, or None.

    `exists()` alone would hand a *directory* named projctl.yaml to Config,
    which opens it and raises IsADirectoryError from a code path whose
    contract is "no config here" — a probe answers whether there is a config
    file to read, not whether the name is taken.
    """
    for name in PROJECT_LOCAL_CONFIG_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


class CorpusResolver:
    """Resolves the caller's own corpus roots and, under `related=True`, each
    declared project's roots — one hop, one direction (C-1); owns every path
    warning.
    """

    def __init__(self, repo_root: Path, config_path: Optional[Path] = None) -> None:
        self._repo_root = repo_root.resolve()
        self._config_path = config_path
        self._skips: list[Skip] = []
        self._resolved_projects: list[ResolvedProject] = []
        self._zero_related = False

    def skips(self) -> list[Skip]:
        """Return every skipped path or project so far, each with its reason."""
        return list(self._skips)

    def resolved_projects(self) -> list[ResolvedProject]:
        """Return one entry per successfully resolved project (caller plus each present related one)."""
        return list(self._resolved_projects)

    def zero_related(self) -> bool:
        """Return whether --related was exercised but resolved to zero related projects (FR-31)."""
        return self._zero_related

    def roots(self, *, related: bool) -> list[CorpusRoot]:
        """Return every resolved, deduplicated CorpusRoot for this run."""
        caller_search, caller_platform = self._load_caller_config()
        roots = list(
            self._resolve_project_corpora(
                repo=self._repo_root.name,
                project_root=self._repo_root,
                platform=caller_platform,
                search_config=caller_search,
                origin=_ORIGIN_CALLER,
            )
        )

        if related:
            related_count = 0
            for declared in caller_search.related:
                resolved = self._resolve_related_project(declared)
                if resolved is not None:
                    related_count += 1
                    roots.extend(resolved)
            if related_count == 0:
                self._zero_related = True

        self._resolved_projects = _dedupe_by_project_root(self._resolved_projects)
        return _dedupe_roots(roots)

    def _record_skip(self, locator: str, reason: str) -> None:
        self._skips.append(Skip(locator=locator, reason=reason))
        logger.warning("search docs: skipping %s — %s", locator, reason)

    def _load_caller_config(self) -> tuple[SearchConfig, str]:
        if self._config_path is not None:
            # An explicit --config path that doesn't exist stays a hard error
            # (FR-4) — FileNotFoundError propagates to the caller.
            config = Config(self._config_path)
            return config.get_search_config(), config.get_raw_platform_or_undeclared()

        probed = _probe_project_config(self._repo_root)
        if probed is None:
            return (
                SearchConfig(docs_path="docs", docs_path_configured=False, related=[]),
                "undeclared",
            )
        config = Config(probed)
        return config.get_search_config(), config.get_raw_platform_or_undeclared()

    def _resolve_related_project(self, declared: str) -> Optional[list[CorpusRoot]]:
        """Resolve one declared related-project path, or record a Skip and return None."""
        locator = _declared_locator(declared, self._repo_root)
        try:
            expanded = Path(declared).expanduser()
        except RuntimeError as exc:
            # Expansion runs before any config file is named, let alone opened,
            # so this outcome owns its reason: `config fault:` would send an
            # operator reading a file nothing here ever touched (§5.3).
            self._record_skip(
                locator, f"declared path cannot be expanded: {_fault_detail(exc, self._repo_root)}"
            )
            return None

        # A related project's config directory belongs to whoever wrote it — any
        # fault between naming that file and holding its resolved corpus roots
        # (a bad YAML parse, an invalid search: shape, an unreadable file) skips
        # that project rather than aborting a run in which every other project
        # resolved cleanly (§5.3, FR-27). Containing this by outcome rather than
        # by exception name is deliberate: an enumerated exception set would key
        # the contract on a list the search: shape check keeps extending.
        fault_base = self._repo_root
        try:
            candidate_root = expanded if expanded.is_absolute() else (self._repo_root / expanded)
            if not candidate_root.exists() or not candidate_root.is_dir():
                self._record_skip(locator, "declared path is absent on this machine")
                return None
            project_root = candidate_root.resolve()
            fault_base = project_root

            probed = _probe_project_config(project_root)
            if probed is None:
                search_config = SearchConfig(
                    docs_path="docs", docs_path_configured=False, related=[]
                )
                platform = "undeclared"
            else:
                config = Config(probed)
                search_config = config.get_search_config()
                platform = config.get_raw_platform_or_undeclared()

            return self._resolve_project_corpora(
                repo=project_root.name,
                project_root=project_root,
                platform=platform,
                search_config=search_config,
                origin=_ORIGIN_DECLARED,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self._record_skip(locator, f"config fault: {_fault_detail(exc, fault_base)}")
            return None

    def _resolve_project_corpora(
        self,
        *,
        repo: str,
        project_root: Path,
        platform: str,
        search_config: SearchConfig,
        origin: str,
    ) -> list[CorpusRoot]:
        roots: list[CorpusRoot] = []

        planning_dir = project_root / "planning"
        resolved_planning = self._check_corpus_root(
            planning_dir, project_root, locator=f"{repo}: planning/", silent_if_missing=True
        )
        if resolved_planning is not None:
            roots.append(
                CorpusRoot(
                    repo=repo,
                    platform=platform,
                    corpus="planning",
                    project_root=project_root,
                    root=resolved_planning,
                    origin=origin,
                )
            )

        if search_config.docs_path is not None:
            raw_docs = _expand_docs_path(search_config.docs_path)
            docs_dir = raw_docs if raw_docs.is_absolute() else (project_root / raw_docs)
            resolved_docs = self._check_corpus_root(
                docs_dir,
                project_root,
                locator=f"{repo}: {_footer_locator(docs_dir, project_root)}",
                silent_if_missing=not search_config.docs_path_configured,
            )
            if resolved_docs is not None:
                roots.append(
                    CorpusRoot(
                        repo=repo,
                        platform=platform,
                        corpus="docs",
                        project_root=project_root,
                        root=resolved_docs,
                        origin=origin,
                    )
                )

        self._resolved_projects.append(
            ResolvedProject(
                repo=repo,
                platform=platform,
                project_root=project_root,
                root_kinds=[r.corpus for r in roots],
            )
        )
        return roots

    def _check_corpus_root(
        self, candidate: Path, project_root: Path, *, locator: str, silent_if_missing: bool
    ) -> Optional[Path]:
        """Validate one corpus root, returning its resolved Path when usable.

        Returns None (recording a Skip unless silent_if_missing) when the
        path is absent, not a directory, or fails containment (FR-28) — a
        directory strictly below its own project root, tested after resolve()
        so a symlink cannot step outside it.
        """
        if not candidate.exists():
            if not silent_if_missing:
                self._record_skip(locator, "configured path is missing")
            return None
        if not candidate.is_dir():
            self._record_skip(locator, "path exists but is not a directory")
            return None

        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(project_root)
        except ValueError:
            relative = None
        if relative is None or str(relative) == ".":
            self._record_skip(locator, "resolves to the project root or outside it")
            return None
        return resolved


def _dedupe_by_project_root(projects: list[ResolvedProject]) -> list[ResolvedProject]:
    """Collapse projects resolving to the same directory, keeping the first."""
    seen: set[Path] = set()
    deduped = []
    for project in projects:
        if project.project_root in seen:
            continue
        seen.add(project.project_root)
        deduped.append(project)
    return deduped


def _dedupe_roots(roots: list[CorpusRoot]) -> list[CorpusRoot]:
    """Collapse roots resolving to the same directory; two corpora at one path collapse to planning."""
    by_root: dict[Path, CorpusRoot] = {}
    order: list[Path] = []
    for root in roots:
        existing = by_root.get(root.root)
        if existing is None:
            by_root[root.root] = root
            order.append(root.root)
        elif existing.corpus == "docs" and root.corpus == "planning":
            by_root[root.root] = root
    return [by_root[path] for path in order]


def _markdown_files_under(root: CorpusRoot) -> tuple[list[Path], list[Skip]]:
    """List every .md file below one corpus root, naming each directory that could not be read.

    `Path.rglob` swallows a directory-level PermissionError inside pathlib's
    own iteration, so an unreadable subtree would vanish with no Skip, no
    warning and a confident indexed_files count — the one outcome the footer
    exists to prevent (§5.6). `os.walk`'s onerror hook hands the fault back
    instead, matching SectionExtractor.extract()'s file-level contract.
    """
    files: list[Path] = []
    skips: list[Skip] = []

    def _record(error: OSError) -> None:
        faulted = Path(error.filename) if error.filename else root.root
        skips.append(
            Skip(
                locator=_relative_display(faulted, root.project_root),
                reason=f"unreadable directory: {_fault_detail(error, root.project_root)}",
            )
        )

    for dirpath, _dirnames, filenames in os.walk(root.root, onerror=_record):
        files.extend(Path(dirpath) / name for name in filenames if name.endswith(".md"))
    return sorted(files), skips


def _walk_corpus_roots(roots: list[CorpusRoot]) -> tuple[list[tuple[CorpusRoot, Path]], list[Skip]]:
    """List every indexed .md file across `roots`, deepest root wins on overlap (FR-32/C-5).

    `*-request.md` intermediates are excluded here (C-5). A file whose
    resolved path escapes its own corpus root (a symlink pointing outside) is
    skipped and named in the footer; the rest of the walk continues (FR-28).
    """
    claimed: set[Path] = set()
    skips: list[Skip] = []
    pairs: list[tuple[CorpusRoot, Path]] = []

    for root in sorted(roots, key=lambda r: len(r.root.parts), reverse=True):
        if not root.root.is_dir():
            continue
        paths, walk_skips = _markdown_files_under(root)
        for skip in walk_skips:
            skips.append(skip)
            logger.warning("search docs: skipping %s — %s", skip.locator, skip.reason)
        for path in paths:
            if not path.is_file() or path.name.endswith("-request.md"):
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(root.root)
            except ValueError:
                reason = "symlinked file resolves outside its corpus root"
                locator = _relative_display(path, root.project_root)
                skips.append(Skip(locator=locator, reason=reason))
                logger.warning("search docs: skipping %s — %s", locator, reason)
                continue
            if resolved in claimed:
                continue
            claimed.add(resolved)
            pairs.append((root, path))

    return pairs, skips


# ---------------------------------------------------------------------------
# Digest assembly and rendering — DocsDigest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoadmapLine:
    """One roadmap entry as it will render: in full, or degraded to a locator."""

    entry: RoadmapEntry
    locator_only: bool


@dataclass(frozen=True)
class RankedLine:
    """One ranked hit as it will render: in full, or degraded to a locator."""

    scored: ScoredHit
    locator_only: bool


@dataclass(frozen=True)
class DocsDigest:
    """Frozen digest result: two ordered halves plus the footer — renders Markdown."""

    # pylint: disable=too-many-instance-attributes
    # The footer's whole purpose is that a partial result cannot read as a
    # complete one (§5.6), so each thing it must disclose — what each half
    # dropped, what was never ranked, what was skipped, what resolved — is
    # its own field. ranked_dropped and ranked_omitted stay apart because
    # they report different losses: a budget-dropped unit was ranked and then
    # did not fit, a cap-omitted one was never ranked at all, so one summed
    # number would name neither cause.

    roadmap_lines: list[RoadmapLine]
    roadmap_dropped: int
    ranked_lines: list[RankedLine]
    ranked_dropped: int
    ranked_omitted: int
    doc_frequency: dict[str, int]
    resolved_projects: list[ResolvedProject]
    skips: list[Skip]
    zero_related: bool

    def render_markdown(self) -> str:
        """Render the full digest: Roadmap, then Prior decisions, then the footer."""
        parts = ["## Roadmap", ""]
        parts.extend(_render_roadmap_line(line) for line in self.roadmap_lines)
        parts.append("")
        parts.append("## Prior decisions")
        parts.append("")
        parts.extend(_render_ranked_line(line) for line in self.ranked_lines)
        parts.append("")
        parts.append(_render_footer(self))
        return "\n".join(parts)


def _neutralize_body(body: str) -> str:
    """Escape every body line that a heading-splitting consumer could misread (FR-35).

    Two shapes qualify: a leading '#', and a setext-or-thematic rule line,
    which is a heading under any non-blank line and is also the digest's own
    footer boundary. Applies inside fenced bodies too — a shell comment
    quoted in a code sample renders with a leading backslash, because a
    mis-split digest costs more than a decorated comment.
    """
    out_lines = []
    for text_line in body.splitlines():
        stripped = text_line.lstrip(" ")
        indent = len(text_line) - len(stripped)
        if stripped.startswith("#") or _SETEXT_RULE_RE.match(stripped):
            text_line = f"{text_line[:indent]}\\{text_line[indent:]}"
        out_lines.append(text_line)
    return "\n".join(out_lines)


def _one_line(text: str) -> str:
    """Collapse `text` to a single line, for rendering inside a Markdown list item.

    Corpus prose, another project's config values and exception messages all
    reach list items in the digest. A value spanning several lines stops
    being one list item the moment it wraps: its tail becomes orphaned
    paragraphs, and any line of it opening with `#` or a rule reads as digest
    structure to a consumer splitting on headings or on the footer's own
    `---` boundary (§5.6, FR-35). Collapsed, the value can only continue the
    `- ` its caller already emitted, so it can open neither.
    """
    return " ".join(text.split())


def _kind_annotation(kind: str) -> str:
    """Return `kind` for rendering, raising on an unrecognized value (FR-11)."""
    return _validate_kind(kind)


def _provenance(hit: Hit) -> str:
    """Render `repo (platform) [kind] path §heading › chain` (FR-22).

    `repo`, `platform` and the path are corpus- or config-derived and carry
    no shape of their own — `platform` is whatever another project's config
    file declares, and POSIX admits a newline in a directory name — so each
    is collapsed before it is interpolated into the list item this returns.
    The heading chain needs no collapsing: _split_sections matches a heading
    against one line at a time, so no heading holds a line break.
    """
    heading = " › ".join(hit.heading_path)
    path = _one_line(_relative_display(hit.path, hit.project_root))
    repo = _one_line(hit.repo)
    platform = _one_line(hit.platform)
    locator = f"{repo} ({platform}) [{_kind_annotation(hit.kind)}] {path}"
    if heading:
        locator = f"{locator} §{heading}"
    return locator


def _render_ranked_line(line: RankedLine) -> str:
    hit = line.scored.hit
    header = f"- **{_provenance(hit)}**"
    if line.locator_only:
        return header
    body = _neutralize_body(hit.body)
    if not body.strip():
        return header
    return f"{header}\n\n{body}"


def _render_roadmap_line(line: RoadmapLine) -> str:
    """Render one roadmap entry as a Markdown list item.

    The `- ` prefix is what makes the line unforgeable: the entry text always
    opens with its own path label, and collapsing it leaves nothing able to
    open a line of its own, so no `#` or rule inside it can be read as digest
    structure (FR-35).
    """
    if line.locator_only:
        return f"- **{_one_line(line.entry.label)}** — {_one_line(line.entry.path.name)}"
    return f"- {_one_line(line.entry.text)}"


def _render_footer(digest: DocsDigest) -> str:
    lines = ["---", "", "### Resolved corpus", ""]
    for project in digest.resolved_projects:
        roots_desc = ", ".join(project.root_kinds) if project.root_kinds else "(none)"
        plural = "" if project.indexed_files == 1 else "s"
        lines.append(
            f"- {_one_line(project.repo)} ({_one_line(project.platform)}): {roots_desc} "
            f"— {project.indexed_files} indexed file{plural}"
        )

    lines.append("")
    lines.append("### Query token document frequency")
    lines.append("")
    for token, freq in digest.doc_frequency.items():
        lines.append(f"- `{token}`: {freq}")

    if digest.skips:
        lines.append("")
        lines.append("### Skipped")
        lines.append("")
        for skip in digest.skips:
            lines.append(f"- {_one_line(skip.locator)}: {_one_line(skip.reason)}")

    if digest.roadmap_dropped:
        lines.append("")
        lines.append(f"### Roadmap entries dropped for budget: {digest.roadmap_dropped}")

    if digest.ranked_dropped:
        lines.append("")
        lines.append(f"### {digest.ranked_dropped} further matches not shown (budget exhausted)")

    if digest.ranked_omitted:
        lines.append("")
        lines.append(f"### {digest.ranked_omitted} further matches not ranked (candidate cap)")

    if digest.zero_related:
        lines.append("")
        lines.append("### --related resolved zero related projects")

    return "\n".join(lines)


def _line_cost(rendered: str) -> int:
    """Return what a rendered line costs its half's budget: the words it emits.

    Charging the rendered form rather than a nominal constant is what makes
    the budget a bound — a locator carries a full provenance header, so a
    nominal charge lets an arbitrarily long tail of them through (NFR-6).
    """
    return len(rendered.split())


def _budget_roadmap(entries: list[RoadmapEntry], budget: int) -> tuple[list[RoadmapLine], int, int]:
    """Consume the roadmap share: full entries, then locators, then drops (§5.6).

    Returns (lines, dropped_count, unspent_budget) — unspent_budget is
    released to the ranked half, never the reverse.
    """
    remaining = budget
    lines: list[RoadmapLine] = []
    dropped = 0
    for entry in entries:
        full = RoadmapLine(entry=entry, locator_only=False)
        cost = _line_cost(_render_roadmap_line(full))
        if remaining >= cost:
            lines.append(full)
            remaining -= cost
        elif remaining > 0:
            locator = RoadmapLine(entry=entry, locator_only=True)
            lines.append(locator)
            remaining -= _line_cost(_render_roadmap_line(locator))
        else:
            dropped += 1
    return lines, dropped, max(0, remaining)


def _budget_ranked(hits: list[ScoredHit], budget: int) -> tuple[list[RankedLine], int]:
    """Consume the ranked-half budget: full entries, then locators, then drops.

    FR-21's guarantee is that no emitted body is truncated, and the three
    tiers keep it. The tail drops rather than emitting a locator apiece,
    because a match count in the hundreds would otherwise size the digest
    instead of WORD_BUDGET doing it; the footer names how many.

    Returns:
        (lines, dropped_count).
    """
    remaining = budget
    lines: list[RankedLine] = []
    dropped = 0
    for scored in hits:
        full = RankedLine(scored=scored, locator_only=False)
        cost = _line_cost(_render_ranked_line(full))
        if remaining >= cost:
            lines.append(full)
            remaining -= cost
        elif remaining > 0:
            locator = RankedLine(scored=scored, locator_only=True)
            lines.append(locator)
            remaining -= _line_cost(_render_ranked_line(locator))
        else:
            dropped += 1
    return lines, dropped


def _assemble_digest(
    *,
    roadmap_entries: list[RoadmapEntry],
    ranked_hits: list[ScoredHit],
    doc_frequency: dict[str, int],
    resolved_projects: list[ResolvedProject],
    skips: list[Skip],
    zero_related: bool,
    ranked_omitted: int = 0,
) -> DocsDigest:
    roadmap_budget = round(WORD_BUDGET * ROADMAP_SHARE)
    roadmap_lines, roadmap_dropped, leftover = _budget_roadmap(roadmap_entries, roadmap_budget)

    ranked_budget = round(WORD_BUDGET * PRIOR_DECISIONS_SHARE) + leftover
    ranked_lines, ranked_dropped = _budget_ranked(ranked_hits, ranked_budget)

    return DocsDigest(
        roadmap_lines=roadmap_lines,
        roadmap_dropped=roadmap_dropped,
        ranked_lines=ranked_lines,
        ranked_dropped=ranked_dropped,
        ranked_omitted=ranked_omitted,
        doc_frequency=doc_frequency,
        resolved_projects=resolved_projects,
        skips=skips,
        zero_related=zero_related,
    )


# ---------------------------------------------------------------------------
# Orchestration — DocsSearchHandler
# ---------------------------------------------------------------------------


def _log_resolved_roots(roots: list[CorpusRoot]) -> None:
    """Log each resolved root with its origin at debug level (NFR-9).

    Skips are not repeated here: each is already logged at warning level
    where it is discovered, and named in the footer.
    """
    for root in roots:
        logger.debug(
            "search docs: resolved root %s (repo=%s, corpus=%s, origin=%s, platform=%s)",
            root.root,
            root.repo,
            root.corpus,
            root.origin,
            root.platform,
        )


class DocsSearchHandler:
    """Searches the local planning/docs corpora and renders a bounded digest.

    No Config gate, no platform dispatch, no network — matching
    ActivityHandler's precedent (see its module docstring).
    """

    # pylint: disable=too-few-public-methods
    # One public entry point (search()), matching ActivityHandler's precedent;
    # an artificial second method would break single-responsibility for
    # nothing. Scoped to the class body so it exempts this class alone.

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        """Initialize the handler.

        Args:
            repo_root: Overrides `git rev-parse --show-toplevel`. Test seam
                only, mirroring ActivityHandler.__init__(cwd=...); production
                callers always pass None.
        """
        self._repo_root = repo_root

    def search(
        self, query: str, related: bool = False, config_path: Optional[Path] = None
    ) -> DocsDigest:
        """Resolve corpora, extract, rank, assemble a digest; print it; return it.

        Args:
            query: Non-empty, non-whitespace-only query text.
            related: Also search every project declared in search.related.
            config_path: Explicit --config path for the caller's own config.

        Raises:
            PlatformError: Not in a git repository (only when repo_root was
                not injected), or git is not installed.
            FileNotFoundError: An explicitly named --config path is absent.
            OSError: The caller's own config file cannot be read.
            ConfigurationError: The caller's own search: section is malformed.
            yaml.YAMLError: The caller's own config file is not valid YAML.
        """
        repo_root = (self._repo_root or get_repo_root(context="'projctl search docs'")).resolve()

        resolver = CorpusResolver(repo_root, config_path)
        roots = resolver.roots(related=related)
        _log_resolved_roots(roots)
        file_pairs, walk_skips = _walk_corpus_roots(roots)

        extractor = SectionExtractor()
        hits: list[Hit] = []
        indexed_files: dict[Path, int] = {}
        for root, path in file_pairs:
            faults_before = extractor.read_faults()
            hits.extend(
                extractor.extract(
                    path,
                    corpus=root.corpus,
                    repo=root.repo,
                    platform=root.platform,
                    project_root=root.project_root,
                )
            )
            if extractor.read_faults() == faults_before:
                indexed_files[root.project_root] = indexed_files.get(root.project_root, 0) + 1

        roadmap_entries = RoadmapIndex().entries(hits, repo_root)

        ranker = Bm25Ranker(hits, query)
        ranked = ranker.rank()

        all_skips = resolver.skips() + extractor.skips() + walk_skips
        digest = _assemble_digest(
            roadmap_entries=roadmap_entries,
            ranked_hits=ranked,
            doc_frequency=ranker.doc_frequency(),
            resolved_projects=[
                replace(project, indexed_files=indexed_files.get(project.project_root, 0))
                for project in resolver.resolved_projects()
            ],
            skips=all_skips,
            zero_related=related and resolver.zero_related(),
            ranked_omitted=ranker.omitted(),
        )

        print(digest.render_markdown())
        return digest
