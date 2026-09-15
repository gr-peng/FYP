# Meta Compute 2026 real-data pilot

Configuration: `config/meta_compute_2026.json`. This implements the first real experiment
from the parent `META_COMPUTE_2026_REAL_EXPERIMENT_PLAN.md`, retaining the original CDA.

## Reproduce

Use Python 3.11. The experiment environment records exact versions in each run.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-experiment.lock
.venv/bin/pip install -e . --no-deps
cp .env.example .env  # only on a new checkout; preserve an existing .env
chmod 600 .env
# Set SILICONFLOW_API_KEY locally; never include it in a command or Git commit.
.venv/bin/python scripts/download_meta_compute_2026.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_meta_compute_suite.py --provider mock --seeds 42 --shocks 0
.venv/bin/python scripts/evaluate_meta_compute_2026.py --provider mock
# Commit tested source before running live inference; a dirty worktree is rejected.
.venv/bin/python scripts/run_historical_event.py --provider siliconflow --agents 1
.venv/bin/python scripts/run_meta_compute_suite.py --modes historical
.venv/bin/python scripts/run_meta_compute_suite.py --modes endogenous --shocks 0
.venv/bin/python scripts/run_meta_compute_suite.py --modes endogenous --shocks -0.02 -0.05
.venv/bin/python scripts/evaluate_meta_compute_2026.py
.venv/bin/python scripts/audit_meta_compute_2026.py
.venv/bin/python scripts/report_api_usage.py
# Alternative HKUST(GZ) relay batch (uses an independent output/cache namespace):
# Set AIGC_API_KEY in the ignored .env file.
.venv/bin/python scripts/run_historical_event.py --provider aigc_relay --agents 1
.venv/bin/python scripts/run_meta_compute_suite.py --provider aigc_relay
.venv/bin/python scripts/evaluate_meta_compute_2026.py --provider aigc_relay
.venv/bin/python scripts/audit_meta_compute_2026.py --provider aigc_relay
.venv/bin/python scripts/report_api_usage.py outputs/meta_compute_2026/aigc_relay
# When the API is unavailable, replay only fully cached runs without a key:
.venv/bin/python scripts/run_meta_compute_suite.py --cache-only --modes historical
```

Run from the repository root. Wrappers also locate the project from other working
directories. A completed run is reused only if config, dataset and source hashes match.
An interrupted run reconstructs state from the start using cached responses; it does
not repeat paid requests already cached. Attempt logs remain append-only. Observations
and final tables are rebuilt. To run a fresh replicate, change the simulator seed.
The cloud model itself may be nondeterministic despite temperature 0.2; reproducibility
means replaying archived responses, not claiming identical future model outputs.

## Data and clock

- Yahoo chart fallback: 11 symbols, April 1–July 17 daily OHLCV, 74 XNYS sessions each.
  Alpha Vantage daily/news adapters are present; no key was supplied for that service.
- OHLC is the provider-returned basis without dividend adjustment. Corporate actions
  are archived. No NVDA split/dividend occurs within the simulation window. KLAC has
  a pre-window split on June 12; its raw provider series is used only for robustness
  event studies. Returns are price returns, not total returns.
- Two canonical source-verified neutral events enter prompts. E1 July 1 10:39 ET becomes
  visible July 2 at 09:30 ET; E2 July 9 14:04 ET becomes visible July 10 at 09:30 ET.
  See event config for original URLs. GDELT discovery is archived separately: its
  observation time is not treated as publication time. No price-reaction headlines
  or future-return summaries are inserted into event text.
- SEC companyfacts use only records with `filed < decision date` and fiscal period
  before the decision. This conservatively delays same-date filings. Unknown ratios
  stay null. Fundamental value is the June 12 close plus a controlled shock, not a
  discounted-cash-flow estimate. SEC retrieval time is not confused with filing time.
- XNYS open/close timezone conversions include holidays, early closes and DST.

## Experimental design

30 agents, initial cash USD 10,000 and 100 NVDA shares each, 15 fundamental and 15
technical. Each submits one daily JSON order; style is reviewed after 10 and 20 sessions.
Memory holds 10 sessions. Population context is lagged one session. Low/high herding
use means 0.2/0.8 and otherwise identical seeded persona draws. Vanilla sees no persona
trait block. Rule ABM uses deterministic fundamental/momentum signals plus seeded noise.
No Calibrated treatment is claimed: empirical human calibration data are unavailable.

Stage A fixes market prices to history. An order is filled only after the decision:
buy limit >= daily low or sell limit <= daily high, at its own limit price. This is
an OHLC-touch execution approximation without queue priority, slippage or fees.
It may be pessimistic for marketable limits; it does not identify intraday execution.

Stage B owns only pre-start real bars. Subsequent primary price, volume and indicators
come solely from agent CDA trades. Context benchmark returns after start are omitted;
there is no hidden real-price anchor or injected market maker. Existing exchange
reservations, price/time priority and DAY expiration remain authoritative. Cash/shares
are conserved. The -2%/-5% shock changes only the fundamental signal when E1 becomes
visible; a 0% shock is news-only. It does not force a market price drop.

The alternative-style return is a deterministic frictionless long/cash signal proxy,
explicitly labeled in prompts. Actual and proxy block returns reset after style review.
This is cheaper than shadow LLMs but is not a matched executable counterfactual account.

Primary pilot: historical 4 treatments × 3 seeds = 12 runs, followed by endogenous
4 treatments × 3 shocks × 3 seeds = 36 runs. Each LLM run has 690 order decisions and
60 style decisions. Rule runs call no model. N=1 preflight is separate from statistics.
E1-only, alternate agent counts and 20-seed calibrated experiments remain follow-ups.

The `aigc_relay` provider uses the exact allowlisted endpoint recorded in the checked-in
configuration and reads only `AIGC_API_KEY` from `.env`. Its advertised model identifier
is `DeepSeek-V4-Flash`, while SiliconFlow reports `deepseek-ai/DeepSeek-V4-Flash`.
Relay results use separate outputs and caches and form a complete provider/time batch;
do not fill missing SiliconFlow cells with relay runs in a single treatment comparison.

## Reliability, security and analysis

SiliconFlow model availability is checked before inference, with no silent model
substitution. JSON/schema errors get one repair request, then HOLD/stay. Unaffordable
orders or overselling become HOLD without retry. Retryable transport failures have
five retries with backoff; authorization/quota/unsupported requests stop the run.
Suite expansion stops if order fallback exceeds 5%. Per-run progress and per-attempt
logs are saved. Concurrency began at eight; after more than 140 successful calls with
no 429 or transport failures, the concurrent request cap was raised to 32. Four parallel
runs later produced 912 recorded 429 replies during the Sept 14 pilot. SiliconFlow says
429 may be RPM/RPD/TPM/TPD/IPM/IPD; the provider response body was not archived, so the
specific quota dimension is unknown. The resumed suite now shares a dispatch pacer at
four requests/second and honors numeric Retry-After. This cap is conservative given the
observed historical throughput; it is not a claim about the account's published limit.
If a 429 lacks Retry-After, all runs pause dispatch for at least 30 seconds before
another attempt, avoiding a simultaneous retry burst.
[Provider error guide](https://docs.siliconflow.cn/en/faqs/error-code).
The run stopped on HTTP 402, which requires account/payment state to be checked by the
key owner before paid inference can continue. No model substitution is automatic.
All decisions are gathered before seeded CDA arrival. Earlier run metadata preserves
the initial settings. No retry is used to change an undesirable valid decision.

`--cache-only` accepts no API key and fails on any missing cached response; it will not
turn a missing model decision into HOLD. A historical batch whose entire response set
is cached can be rebuilt with no completion calls. Partial endogenous runs still need
a valid API account to complete. Logs are append-only across restarts; a resumed run
reconstructs state from the beginning and verifies the same input hash.

Credentials stay in ignored `.env` (0600), never in prompts, cache keys, output config
or URLs. Data, outputs and cache are Git-ignored. Raw data are content-addressed with
SHA-256 metadata; processed and run artifacts have hashes. Transport errors do not
print authenticated request URLs. The API key is redacted from response logs.

Evaluation distinguishes micro behavior from macro path fit. Event-study alpha/beta
use pre-experiment returns and QQQ/SPY benchmarks. Statistics use seed-level runs,
with mean/std/range, exploratory Mann–Whitney and Cliff's delta, plus paired seed
differences. Three seeds do not establish significance or human behavioral validity.
Volume in a 3,000-share toy market is not compared to real float turnover without a
verified real float denominator. Pricing is a separate dated configuration, not
embedded in the policy; an estimate is not the account invoice.
Simulated CAR differences reuse the real NVDA pre-event beta and real QQQ path only
during ex-post evaluation. Historical replay should have zero CAR difference by
construction. An endogenous agent never sees future QQQ returns. The first funded
batch stopped at 24 of 48 main runs: all 12 historical, 9 rule-based endogenous and
3 LLM endogenous runs. This is a partial pilot, not the completed treatment matrix.

The current server's base Conda site-packages had incomplete async dependencies and
NumPy-1 compiled optional extensions. Local venv overlays repair these; a fresh
isolated venv with the lock file avoids relying on the shared base installation.
