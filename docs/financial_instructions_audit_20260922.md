# Original financial instructions: follow-up audit

The [authors' working paper](https://en.wiwi.uni-paderborn.de/fileadmin-wiwi/cetar/TAF_Working_Paper_Series/TAF_WP_043_MinhLyPelster_2019.pdf),
Appendix A, confirms three practice rounds followed by a reset to 10,000 Talers.
It announces both ranking treatments before trading. The interface shows cash,
holdings and prices. This resolves
the previously provisional practice/main mapping and establishes that anticipated
ranking rules belong in financial prompt context even before the first ranking.

Archived PDF SHA256:
`f32eabfb37b88961be59ba89c2f4fb70c280ee3c18771007bf1bb46eb83d5c1f`.
Source manifest and PDF are under `data/raw/finance_scopic_20260922/`.
The [original OSF registration](https://osf.io/4nrmx/) dated 2019-06-20 also
provides [experimental instructions](https://osf.io/download/hv2un/); that PDF and
its hash are archived separately and confirm the same practice/reset/ranking rules.
The share-type table and interface screenshot were inspected from rendered pages.
No actual asset-to-type mapping from the researcher's table enters model inputs.

## A constraint discrepancy found in the data

`scripts/audit_scopic_instruction_constraints.py` reuses only the already-opened
training and development partitions. Four of 150 training decisions and four
of 135 development decisions spend more on purchases than the reconstructed
cash at the start of that period. All eight are fully funded when same-period
sales are included. Previous share/cash conservation checks continue to pass.

The instructions describe spending within current cash and immediately crediting
sales. At the archived period resolution, these records are compatible with sales
funding purchases within the period. A stricter gross-buy-at-period-start rule
would reject observed human records; the exact interface/validation sequence is
not established from this screenshot. Preserve that ambiguity rather than
silently excluding the eight decisions or changing the source data.

The current sell-any target remains identifiable: owned assets can be sold,
and post-trade quantity is never used in its state. A complete joint portfolio
executor needs an explicit settlement convention consistent with these records
and, ideally, the original task code. The existing net period ledger is a data
reconstruction, not proof of every within-period feasibility condition.

## Remaining information-set work

For the next LLM/Jev financial comparison, include the known announced ranking
rule, horizon, incentives, cash/share limits and common stock-type distribution.
Keep hidden type assignments and later rankings out of the prompt. Include
code-computed arithmetic consistently across models and mark average cost as
derived information, not necessarily a field on the original screen.

The frozen statistical comparison used a deliberately restricted feature set;
preserve its protocol and results. New context or practice-history features need
a separately named comparison. A same-state binary forecast does not by itself
validate joint quantity decisions, personal persistence or autonomous rollout.
