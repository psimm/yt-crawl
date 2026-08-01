"""Account and usage models for GET /me."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Account(BaseModel):
    """Monthly credit usage for the authenticated account."""

    current_month_usage: int = Field(..., description="Credits used this month")
    monthly_allowance: int = Field(..., description="Monthly credit allowance")
    remaining_credits: int = Field(..., description="Credits remaining this month")


class ApiUsage(BaseModel):
    """Hourly search rate-limit counters."""

    searches_this_hour: int = Field(..., description="Searches performed this hour")
    hourly_rate_limit: int = Field(..., description="Hourly search rate limit")


class Subscription(BaseModel):
    """Current subscription billing period."""

    period_start: str = Field(..., description="Subscription period start")
    period_end: str = Field(..., description="Subscription period end")


class AccountResponse(BaseModel):
    """Response from GET /api/v1/me."""

    account: Account
    api_usage: ApiUsage
    subscription: Subscription | None = None
