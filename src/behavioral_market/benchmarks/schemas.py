from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BenchmarkChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    choice: Literal["A", "B"]


class Choices13kItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: int = Field(ge=0)
    problem_id: int
    n_humans: int = Field(gt=0)
    feedback: bool
    human_b_rate: float = Field(ge=0, le=1)
    option_a: list[tuple[float, float]]
    option_b: list[tuple[float, float]]
    ambiguity: bool
    correlation: int
    block: int
    has_loss: bool
    ev_a: float
    ev_b: float
    var_a: float = Field(ge=0)
    var_b: float = Field(ge=0)

    @field_validator("option_a", "option_b")
    @classmethod
    def valid_gamble(cls, value):
        if not value:
            raise ValueError("gamble must contain at least one outcome")
        if any(probability < 0 or probability > 1 for probability, _ in value):
            raise ValueError("outcome probabilities must lie in [0, 1]")
        if abs(sum(probability for probability, _ in value) - 1) > 1e-6:
            raise ValueError("outcome probabilities must sum to one")
        return value

