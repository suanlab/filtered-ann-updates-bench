E5 first run (tag r2, 2026-10-23 01:58-11:38), superseded. bench/stream_diskann_policy.py asked the stale index for
K / s results with s the surviving share of ALL the build's points; under correlated deletions the query label's
survival is lower, so many queries got fewer than K live results at every L and the rounds between rebuilds missed
recall 0.95 (round-2 maximum recall 0.946 for P >= 3 on SIFT-500K-adv). The delta preparation was also outside every
timer. The corrected runner (tag r2b) uses the label's survival, re-searches with twice as many results when short
(inside the timer), and times the delta preparation; in a smoke test the same round reaches 0.9998. Found through the
review in IDEA_ICDE.md (P0-3).
