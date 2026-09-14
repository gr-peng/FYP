# Experiment protocol status

MVP-0 is a deterministic software-validation run, not a behavioral experiment. A run fixes
the JSON configuration, random seed and code revision; it records all submitted orders,
trades, market closes and terminal accounts. Re-running the same inputs should yield
byte-identical CSV records and config snapshots (the metadata run timestamp varies).
Market price changes only after trades. Aggregate cash and
shares must remain constant. The default 30-period run is a smoke test for the exchange.

Stage 1 will use an immutable historical path and separate agent behavioral outcomes.
Stage 2 will initialize from a price and then use executed trades alone for subsequent
prices. Causal data timestamps, lagged mean-field publication, counterfactual benchmark
definitions, calibration targets and multiple seeds must be finalized before scientific
claims. See the TDD and ADR register for the decisions still pending.
