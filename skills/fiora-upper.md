# FIORA planning and review

- Keep the experiment sequence short and conditional. Define what evidence
  would justify each larger or more expensive step before scheduling it.
- Before comparing model changes, lock one scorer contract: peak matching
  tolerance, intensity transform, thresholds, precursor handling, aggregation
  unit, and split/exposure categories.
- Treat reference scores as empirical references unless they are actually
  proven upper bounds under that scorer.
- Require checkpoint identity and evaluation provenance from machine-readable
  metadata rather than filenames or prose.
- Before full training or multi-arm ablations, require a tiny overfit check,
  CPU/GPU numerical comparison where applicable, and measured GPU throughput.
- Compare at most one or two well-motivated changes at a time on the same
  held-out spectra. Require uncertainty by compound and precursor-removed
  scores where feasible.
- Check the iteration's artifacts and units directly. Do not accept conclusions
  based only on the iteration summary.
