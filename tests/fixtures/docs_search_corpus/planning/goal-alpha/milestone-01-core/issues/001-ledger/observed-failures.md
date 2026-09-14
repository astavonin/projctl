# Observed Failures — ledger reader

## 2026-04-02 The ledger reader dropped every record after the first

**Observed in:** manual testing against a ledger holding four observed failure records.
**Root cause:** the reader returned on the first match instead of accumulating, so a
ledger with several entries reported one. The failure was invisible in the fixture,
which held a single record.
**Status:** covered
**Test:** a ledger fixture carrying four records asserts four parsed entries.
**Evidence:** before fix the assertion read 1 == 4; after fix it reads 4 == 4.

## 2026-04-09 A failure record with no Status field aborted the whole run

**Observed in:** a real ledger whose last record was still being written.
**Root cause:** the reader indexed the Status field directly rather than asking for it,
so a partial record raised instead of being treated as unclassified. One malformed
record cost the whole ledger.
**Status:** covered
**Test:** a ledger whose final record carries no Status parses the records above it.
**Evidence:** before fix KeyError; after fix three entries and one unclassified.

## 2026-04-17 The open-failure pin never fired against a real ledger

**Observed in:** a query whose top result should have been the one open record.
**Root cause:** real records write a qualifier after an em dash, so the status value
never compared equal to the bare word the pin looked for. Every ledger in the corpus
reads that way, so the pin had never fired outside its own fixture.
**Status:** open — not reproduced under the synthetic fixture
**Test:** a record whose status carries a trailing qualifier still pins first.
**Evidence:** the comparison hit outranks the record on score, so only the pin orders it.
