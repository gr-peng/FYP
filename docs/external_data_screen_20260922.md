# External data screening, 2026-09-22

This is a source and schema audit, not a fitted benchmark or a results report.
External risk-choice rates and financial action/price outcome columns were not
used to select the calibration-v2 method. Human trial/market splits are still to
be frozen before behavioral model development on the financial data.

**CPC18**

Downloaded the official 210-problem complete-distribution workbook, plus the
complete raw archive and dictionary from [Zenodo 2571510](https://zenodo.org/records/2571510).
The Zenodo metadata identifies CC-BY-4.0; downloaded CSV/PDF match the upstream
checksums. The raw CSV is 65,363,460 bytes. Source metadata, content hashes and
files are under `data/raw/cpc18_external_20260922/`.

`scripts/audit_cpc18_inputs.py` deliberately excludes the workbook's B.1–B.5
response-rate columns, reads only visible gamble fields from the local Choices13k
source, and compares canonical fingerprints. Of 210 workbook problems, 182 are
non-ambiguous. Five match Choices13k source gambles: GameIDs 3, 4, 24, 29 and 201.
This leaves 177 non-ambiguous candidate problems after the conservative exclusion
against all Choices13k source rows, not only previously selected subsets.

The eligible IDs and both source hashes are recorded in `feature_overlap_screen.json`.
This is a near-family cross-source test, not proof against pretraining contamination.
The workbook covers the calibration 210; the later full raw archive may cover
additional competition problems and must not be assumed identical. Ambiguous
worksheet padding must be reconciled with human presentation before including it.

**Human single-asset markets**

The [2017 PLOS article](https://doi.org/10.1371/journal.pone.0189359) points to
[OSF bqykw](https://osf.io/bqykw/). Its workbook was successfully downloaded and
hashed under `data/raw/finance_cda_2017_20260922/`.

The metadata-only scan finds 25,042 rows, 21 distinct SessionIDs, two conditions
(BuyANDSell / BuyORSell), and rounds 1–12. The numeric Session field is reused
across conditions: split/group on the full SessionID, not numeric Session alone.
There are nine BuyANDSell sessions and twelve BuyORSell sessions.

Available columns include action, maker, accepter, offered price, transaction flag,
market price, offer flag, contract/trade IDs, volume, and fundamental value. The
current scan read only session/condition/round/FV metadata, not human action/price
outcomes. `schema_screen.json` records the scope.

This is a promising micro/macro source, but the article describes explicit
acceptance and cancellation of related proposals after a trade. Our current
automatic price/time-priority matching is not identical. Before a fidelity claim,
audit event ordering, quote lifetime, initial endowments, dividends, per-round
opportunities and whether unrecorded inactivity can be observed. Do not infer a
HOLD probability from a log containing only active orders, or call a mismatched
mechanism an exact human experiment replication.

**Human trading and risk preferences**

The [Mendeley dataset](https://data.mendeley.com/datasets/jfg8s32xdm/1) accompanying
Liêu and Pelster's [data article](https://doi.org/10.1016/j.dib.2020.105680) provides
`data_de_scopic.csv` and an RData equivalent, licensed CC-BY-4.0. The downloaded
CSV matches the published SHA256:
`327f7c512733fffe0efb8ee83944cefb4320ab538a930b920b5cdd25595ac333`.

The local archive is `data/raw/finance_scopic_20260922/`. Headers include anonymous
participant/session/group IDs, round, six assets' pre-trade holdings, buy/sell
quantities, post-trade holdings and prices, and ten Holt–Laury response fields.
This may connect individual risk measures with repeated financial decisions more
directly than separate Choices13k draws. Only headers and group/round metadata have
been inspected in this screening, not the risk or trade response columns.

Before use, reconstruct decision-time cash and price histories from instructions,
identify the treatment and the displayed comparison group, and determine the
correct independent split unit. Risk elicitation or final wealth measured after
the trading task must not become pre-decision test inputs. Six coupled assets
also mean that a single-asset projection requires an explicit task definition;
it is not automatically equivalent to the source experiment.

**Other candidates and next action**

The Drehmann et al. herding replication has a verified file listing but local
retrieval returned HTTP 403; no raw file was obtained. The 166-market replication
paper's OSF materials and linked nodes provided papers, preregistrations and plot
images in the inspected listings; raw transaction availability remains unverified.

Prioritize the two downloaded financial datasets for information-set and schema
audits, then freeze session/group splits before inspecting training behavior.
Use training groups to reproduce source statistics and development groups to
select adapters. Preserve held-out human groups for micro and macro evaluation.
Do not calibrate market liquidity directly against the held-out macro target.
