# Architecture decision register

| ADR | Status | Decision |
| --- | --- | --- |
| 001 transaction-price rule | Accepted for MVP-0 | Incoming trade executes at the resting order's limit price. |
| 002 order expiration | Accepted for MVP-0 | All open/partially filled DAY orders expire at period end. |
| 003 counterfactual return | Planned | Start with deterministic `rule_proxy`; define exact benchmark with Phase 2. |
| 004 mean-field timing | Planned | Publish period `t-1` aggregate to agents at period `t`. |
| 005 fundamental-value process | Planned | Select and document an exogenous synthetic process before Stage 2. |
| 006 LLM output schema | Planned | Validate structured style/order decisions before exchange submission. |
| 007 calibration target | Planned | Use directional theory until a sourced quantitative human target exists. |

The planned decisions are design constraints from the TDD, not implemented MVP-0 behavior.
