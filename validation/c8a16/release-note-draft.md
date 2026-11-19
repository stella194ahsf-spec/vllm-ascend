# Release note draft

待分配正式 PR 编号后，写入 `release_notes/fragments/<PR number>.yaml`，不虚构 PR 编号。

Category: feature

Text: Added single-query FIA Mode 5 routing for ragged dynamic INT8 KV speculative verification on Ascend, including mixed prefill batches. This path requires eager or supported piecewise execution and introduces exact-length synchronization; A3 validation is required before merge.
