# Restricted financial baseline comparison

Frozen before reading development behavior, 2026-09-22. Exploratory development
study; not an independent test or an endogenous-market simulation.

Use the already frozen whole-session train/development split. Build only main
periods 1–5 under the documented raw-round mapping; retain each asset opportunity
where the participant starts with positive inventory. Outcome: any sale of that
asset in the period. All six assets' pre-decision context remains available.
Current quantities traded and subsequent prices/wealth/preferences are targets or
excluded diagnostics, never predictors. Later peer-ranking periods are excluded.

Compare five fixed baselines:

1. HOLD probability of selling = 0.
2. Coin probability = 0.5.
3. Training participant-balanced unconditional sell probability.
4. Gain/loss/flat conditional training sell rate, using calculated average cost;
   each bin is shrunk with one pseudo-observation at the training global rate.
5. Ridge logistic regression on unrealized return, gain indicator, last return,
   three-step return, portfolio weight, cash fraction, period fraction, log quantity.
   Standardize using training person-balanced moments. Minimize person-balanced
   log loss plus 0.01/2 times squared nonintercept coefficients; no tuning search.

Each participant with eligible opportunities has equal total fitting weight.
Primary descriptive metric is participant-balanced Brier score. Also report
opportunity-average Brier, participant-balanced log loss (clip only for scoring
at 1e-6), mean predicted versus actual selling and separate session metrics.
Do not rank by raw accuracy, which rewards the prevalent HOLD action. Two
development sessions do not support an IID confidence interval over asset rows.
Report direction and magnitude without a formal population superiority claim.

Freeze exact code/protocol/partition hashes and commit before evaluation. Do not
change features or regularization after development feedback in this comparison.
Report every baseline, including failures. Test data remains inaccessible in the
loader. A later LLM/Jev comparison must use the same information and specify
probability elicitation, action sampling, counterbalance and inference budget
before opening test behavior. This task predicts actions in recorded human states;
it does not establish consistency of a full persistent agent rollout.
