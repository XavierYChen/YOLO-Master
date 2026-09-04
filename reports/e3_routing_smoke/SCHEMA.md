# Routing snapshot schema

`routing_snapshot.json` uses schema ID `e3.routing_snapshot.v1`.

- Root: purpose, non-accuracy disclaimer, fixed input, seed, and profiles.
- Profile: family, status, source YAML, initialization, captured layer counts, validation errors, and overhead.
- Layer: name/type, expert count, top-k, normalized expert usage, dominant share, entropy, Gini, dead experts, and captured tensor shape.
- Overhead: method/scope, warmup and iteration counts, baseline and instrumented latency summaries, median delta, and median ratio.

The schema intentionally keeps raw per-position routing tensors out of Git; aggregate evidence is enough for admission and remains small and reviewable.
