# Nano30B Fork Disclosure

This vendored tree is executable project code. It is not an untouched reference
checkout, and the initial import did not preserve an upstream Git commit ID.
That missing pin prevents byte-for-byte claims about parity with the paper
implementation.

Material Nano30B adaptations include:

- Nemotron-H architecture, layer, cache, and tokenizer integration;
- Nano activation injection with scale `75` instead of the reference recipe's
  `150` (no clean R33 scale sweep has yet established optimality);
- project-specific directional-FVE and reward reporting;
- frozen-critic RL experiments rather than a jointly trained critic;
- packed-path equivalence guards, rollout synchronization, queue telemetry,
  source contracts, and system metrics;
- edits in `nla/loss.py`, `nla/injection.py`, `nla/reward.py`, actor training,
  rollout, and Miles patch integration.

External reports must describe this as an adaptation of the NLA method. They
must not describe the directory as a pristine upstream implementation or claim
that Nano results reproduce the reference recipe without an explicit
method-by-method comparison.
