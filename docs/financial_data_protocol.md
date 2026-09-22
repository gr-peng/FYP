# Financial human-data development protocol

2026-09-22. This freezes data access and partition rules before behavioral EDA.
It does not preregister a final model or claim a market replication.

## Sources and decision-time constraints

The [Yechiam et al. market methods](https://doi.org/10.1371/journal.pone.0189359)
specify 12 four-minute periods, 7–8 traders, single-unit proposals and explicit
acceptance. Dividends are 20 or 40 tokens; expected remaining value is
`30 * (13 - period)`. BuyANDSell starts at 3,600 cash/10 shares per trader;
BuyORSell buyers start at 7,200/0 and sellers at 0/20. Accepted trades remove
the buyer's pending bids and seller's asks. Cash and shares constrain actions.
The interface displays holdings, cash, pending proposals and transaction history.
The archive lacks explicit timestamps/dividend realizations in its listed schema.
Our audit must determine whether event order, identities and inventories can be
reconstructed. Activity-only rows cannot identify a population HOLD rate.
The existing automatic matching engine requires a separate institution adapter.

The [Liêu–Pelster task description](https://pmc.ncbi.nlm.nih.gov/articles/PMC7256287/)
specifies 14 trading periods, six exogenous price processes, 10,000 initial Talers,
four pre-task historical prices, and unknown stock-type assignments. Two fixed
peers are compared after periods 5, 10 and 14; rankings depend on treatment.
Risk/loss elicitation occurs after trading. It cannot be an observed initial
trait for held-out forecasts. Our metadata has 17 rows per participant; those
rows must be reconciled with 14 trading periods before constructing decisions.
Exogenous-price decisions support micro evaluation, not an endogenous-market claim.

## Frozen partitions

`scripts/freeze_finance_partitions.py` reads explicit allowlists of session,
condition, participant, group and round metadata only. The committed JSON records
source hashes and exact memberships. No action, price, wealth or preference
response selects a partition.

- CDA: whole SessionID, condition-stratified; 11 train / 5 development / 5 test.
  BuyANDSell contributes 5/2/2 and BuyORSell 6/3/3. Numeric session alone is not unique.
- Scopic: whole session.code, 2/2/2 sessions. Select deterministically among
  all partitions with both treatments represented in every split, using a seeded
  SHA256 ordering. Treatment coverage is metadata only. Entire fixed peer groups
  and all participant rounds stay together. Two test sessions cannot support
  precise population inference; group or person counts do not replace session N.

Train is for schema reconstruction, data-quality checks, behavior summaries and
fitting. Development is for subsequently declared comparisons. Test behavior stays
closed until policy, metrics, exclusions, uncertainty and rollout rules are frozen.
The loader exposes only train/development, filters before behavioral conversion,
and verifies source hashes. Underlying CSV/XML bytes necessarily pass through the
parser; held-out action/outcome fields are not inspected or summarized.

## Next gates

1. Freeze and commit partitions before reading train behavior.
2. Audit train conservation, chronology, repeated rows, treatments and visible state.
3. Define faithful decision targets: active-event prediction is distinct from a
   timed trading policy; teacher-forced historical states are distinct from rollout.
4. Compare simple behavioral/statistical baselines with plain/calibrated LLMs and
   Jev when available, using the same information and feasible actions.
5. Freeze independent micro evaluation before opening test behavior. Only then
   define the institution-matched micro-to-macro comparison and its human reference.

The completed training-only timing audit and restricted state adapter are described
in `financial_training_audit_20260922.md`. In particular, `result_final` marks
holdings at the next price realization, and the market workbook annotates quote
rows with eventual acceptance. Neither is a safe current-state input.

Future information (later prices, current action, post-action holdings, final
wealth, post-trading risk answers and unrevealed peers' outcomes) is excluded from
decision inputs unless the original interface made that value available already.
