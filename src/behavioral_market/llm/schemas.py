from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Style = Literal["fundamental", "technical"]


class OrderDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action: Literal["buy", "sell", "hold"]
    quantity: int = Field(strict=True, ge=0)
    limit_price: float | None
    rationale: str = Field(max_length=2000)
    sentiment: Literal["strongly_bearish", "bearish", "neutral", "bullish", "strongly_bullish"]

    @model_validator(mode="after")
    def coherent(self):
        if self.action == "hold":
            if self.quantity != 0 or self.limit_price is not None:
                raise ValueError("hold requires zero quantity and null price")
        elif self.quantity <= 0 or self.limit_price is None or self.limit_price <= 0:
            raise ValueError(
                "buy/sell requires positive integer quantity and finite positive price"
            )
        return self


class StyleDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["stay", "switch"]
    target_style: Style
    rationale: str = Field(max_length=2000)


def hold(reason="validation fallback") -> OrderDecision:
    return OrderDecision(
        action="hold", quantity=0, limit_price=None, rationale=reason, sentiment="neutral"
    )
