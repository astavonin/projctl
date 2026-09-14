# Design — ledger reader

## 5. Detailed Design

The reader walks each observed failure ledger once and yields one record per dated
heading, carrying the bold fields below it. Nothing is cached: the ledger is read at
the moment the question is asked, so a record appended a minute ago is visible.

## 7. Trade-offs and Alternatives

### Option A — read the ledger on every query

Cost is one pass over a few hundred kilobytes. No staleness contract is owed, and a
ledger edited mid-session is reflected immediately. Rejected alternatives below all
trade that freshness for a saving the measurement does not justify.

### Option B — build an index beside the ledger

A second artefact to keep truthful, and every observed failure record it holds is a
copy that can disagree with the ledger it came from. The failure mode is silent.
