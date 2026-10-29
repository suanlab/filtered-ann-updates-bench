EMA files of follow-up 2's end-to-end campaign (tag e2e, 2026-10-26), superseded by follow-up 4's e2e4
(paper_audit/followup4_prereg.md, B2). bench/stream_ema.py had no E2E mode then, so these runs were timed like the
kernel runs: one batched call over all 500 queries, with the DNF predicates built outside the timer and no result
validation inside it (CHECK.md B05). The runner now has an E2E mode, and runners without one refuse E2E=1.
