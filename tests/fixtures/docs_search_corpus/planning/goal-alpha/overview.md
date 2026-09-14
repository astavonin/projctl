# Goal Alpha

**State:** in progress

## About

- The observed failure ledger is the durable record of every fault that actually happened
- Each entry names the symptom, the root cause, the test that reproduces it, and the evidence
- Entries are appended, never rewritten, so the history of a defect survives its fix
- A fix that lands without a ledger entry fails the verification gate

## Scope

Ledger discipline across every milestone under this goal.
