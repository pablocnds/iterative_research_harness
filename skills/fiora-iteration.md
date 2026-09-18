# FIORA experiment execution

- Make setup commands fail closed (`set -euo pipefail` or an equivalent), and
  test the exact imports and executable entry points needed by the experiment.
  Do not mask a setup failure with a successful trailing command.
- Preserve pinned source revision, dependency/environment record, commands,
  configuration, split identity, checkpoint metadata, and hashes for acquired
  data or important outputs.
- Establish one tested scorer contract before optimizing a model. Include edge
  cases for matching tolerance, precursor removal, intensity transforms, and
  aggregation by spectrum versus compound.
- Run the smallest useful gates first: a CPU smoke test, a tiny overfit test,
  GPU numerical/equivalence check, and a measured throughput sample. Stop and
  replan if a gate fails.
- Estimate the cost of a larger run from measured throughput. Prefer one or two
  controlled arms over broad sweeps, and evaluate every arm on the same frozen
  held-out examples.
- Runpod jobs are asynchronous. Publish outputs atomically, request explicit
  output paths, wait through the Runpod tool, and verify synchronized artifacts
  and terminal state before drawing conclusions.
- Report measurements with units and aggregation methods. Separate observed
  results from hypotheses about optimization, data, labels, or model capacity.
