Concurrency runs of follow-up 2 (tag cc, 2026-10-26), superseded by follow-up 4's cc4 (paper_audit/followup4_prereg.md, B1).
The post-filtering (hnswlib) and EMA runs applied insertions and deletions while searches ran, with no lock. Neither
library permits this: hnswlib does not document add_items or mark_deleted as safe during knn_query, and EMA's README
describes maintenance as stop-the-world. The IVF runs held a reader-writer lock, so the two groups did not pay the same
synchronization cost. The stale check also rebuilt its deletion times every round, so a result deleted in an earlier
round was not counted (CHECK.md B01, B02). No crash or corrupted result was observed. These files are kept for
provenance; the paper does not use them.
