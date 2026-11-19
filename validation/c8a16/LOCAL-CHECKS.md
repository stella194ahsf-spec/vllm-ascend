# Local checks and A3 handoff

Baseline: ec5cecf3dceedbd301a199596c7dd886b3e92034.
Paired vLLM: 544f182d9f0c9bfe8bd7645525951d5e3692a542.

- 25 attention tests passed in a Windows CPU isolation harness. Framework/NPU initialization was stubbed; FullAttentionSpec definitions came from the paired vLLM revision. This is not an A3 pytest result.
- Added an independent CPU full causal-attention reference comparison for ragged split routing.
- Focused Ruff check and format checks passed (local Ruff 0.17.4).
- Bash syntax checks passed for both new scripts and all 14 shell code blocks in A3-MTP-V2.md; embedded Python compiled.
- Metric parsing and first-divergence comparison checks passed, including exclusion of external prefix metrics and refusal to auto-pass tie candidates.
- markdownlint-cli 0.45.0 passed for the runbook and release-note draft.
- git diff --check passed.
- Repository-wide format.sh ci was attempted but stopped because pre-commit is not installed on this computer. Full hooks, ShellCheck and the repository-pinned Ruff version must run in CI/A3 before merge.
- No A3 operator, MTP/GDN, PD or long-context performance execution took place on this computer.

Run A3-MTP-V2.md in the existing container. Keep all results on A3; no automatic log uploads are performed. The code branch is for validation and must not replace the already validated baseline until the hardware gates pass.
