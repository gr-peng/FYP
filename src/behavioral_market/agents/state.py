import random
from dataclasses import asdict, dataclass, field

from behavioral_market.market import Portfolio


@dataclass(frozen=True)
class AgentPersona:
    agent_id: str
    risk_tolerance: float
    loss_aversion: float
    herding: float
    wealth_sensitivity: float
    mispricing_sensitivity: float
    initial_style: str


@dataclass
class AgentState:
    persona: AgentPersona
    portfolio: Portfolio
    current_style: str
    initial_equity: float
    block_start_equity: float
    memory: list = field(default_factory=list)
    shadow_growth: float = 1.0
    peak_equity: float = 0.0

    def context(self):
        equity = float(self.portfolio.total_equity)
        return {
            "persona": asdict(self.persona),
            "current_style": self.current_style,
            "portfolio": {
                k: (v if isinstance(v, int) else round(float(v), 6))
                for k, v in asdict(self.portfolio).items()
            },
            "memory": self.memory,
            "counterfactual": {
                "method": "long_cash_signal_proxy",
                "actual_block_return": equity / self.block_start_equity - 1,
                "alternative_block_return": self.shadow_growth - 1,
                "gap": self.shadow_growth - equity / self.block_start_equity,
            },
            "drawdown": 1 - equity / max(self.peak_equity, equity),
        }


def population(config, treatment, price):
    rng = random.Random(config["experiment"]["seed"])
    spec = config["population"]
    n = spec["n_agents"]
    styles = ["fundamental"] * round(n * spec["initial_fundamental_share"])
    styles += ["technical"] * (n - len(styles))
    rng.shuffle(styles)
    agents = []
    for i, style in enumerate(styles):
        risk, loss, wealth, mispricing = [round(rng.uniform(0.2, 0.8), 5) for _ in range(4)]
        herding_z = rng.gauss(0, 1)
        mean = {"low_herding": 0.2, "high_herding": 0.8}.get(treatment, 0.5)
        persona = AgentPersona(
            f"A{i:03}",
            risk,
            loss,
            round(min(1, max(0, mean + 0.05 * herding_z)), 5),
            wealth,
            mispricing,
            style,
        )
        p = Portfolio(spec["initial_cash"], spec["initial_shares"], price)
        p.mark_to_market(price)
        equity = float(p.total_equity)
        agents.append(AgentState(persona, p, style, equity, equity, peak_equity=equity))
    return agents
