"""Cost estimate for the frozen frontier panel, per model, per route, per AWS account.

Every number here is either a published price or a measured token count. Nothing is a guess
about how many tokens a prompt takes: the means below were measured by sending the real
AgriFacts and AgriAdvice prompts to each model through the same router the study uses, and
reading the token counts the providers returned.

The routing model matches the harness exactly, per model:

    Grok 4.3              xAI direct        ->  OpenRouter key 1  ->  OpenRouter key 2
    GPT-5.6 Luna          OpenAI direct     ->  OpenRouter key 1  ->  OpenRouter key 2
    Nova 2 Lite           Bedrock acct A/B  ->  OpenRouter key 1  ->  OpenRouter key 2
    Nemotron Nano 3 30B   Bedrock acct A/B

Bedrock requests alternate between the two AWS accounts, so each carries half of whatever
Bedrock serves. The fallbacks cost nothing while the primary works, so the headline figure
is the primary-route cost; the fallback column is what you would pay if a primary went down
for a whole run.

Prices are USD per million tokens, on-demand, excluding tax, verified 2026-09-05. Re-check
them before quoting the figure anywhere: provider prices move, and both of these moved
during 2026.

Run:  python CPU_Run/estimate_api_cost.py
      python CPU_Run/estimate_api_cost.py --advice-max-tokens 256
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, write_csv
from GPU_Run.common.paths import AGRIADVICE_PAIRS, RESULTS_DIR, TEST_INSTANCES_FROZEN

logger = get_logger("estimate_api_cost")

PRICE_VERIFIED_ON = "2026-09-05"

# USD per 1,000,000 tokens, by the route that actually serves each model.
PRIMARY_PRICES = {
    # Grok 4.3 is one generation behind 4.5 and 4.6, which xAI prices at $2.00 and $6.00.
    "frontier-grok-4-3":            {"route": "xai-direct", "input": 1.25, "output": 2.50},
    # OpenAI cut Luna from $1.00/$6.00 to this in July.
    "frontier-gpt-5-6-luna":        {"route": "openai-direct", "input": 0.20, "output": 1.20},
    "frontier-nova-2-lite":         {"route": "aws-bedrock", "input": 0.30, "output": 2.50},
    "frontier-nemotron-nano-3-30b": {"route": "aws-bedrock", "input": 0.06, "output": 0.24},
}
FALLBACK_PRICES = {
    "frontier-grok-4-3":            {"route": "openrouter", "input": 1.25, "output": 2.50},
    "frontier-gpt-5-6-luna":        {"route": "openrouter", "input": 0.20, "output": 1.20},
    "frontier-nova-2-lite":         {"route": "openrouter", "input": 0.30, "output": 2.50},
    # Nemotron Nano 3 30B is absent from the OpenRouter catalogue, so it has no fallback.
}
# Only Bedrock traffic is split across the two AWS accounts.
BEDROCK_ROUTED = {"frontier-nova-2-lite", "frontier-nemotron-nano-3-30b"}

# Measured on the real prompts over each model's PRIMARY route, 2026-09-05. mcq covers the
# frozen-test, rotation and identity-swap passes, which share a prompt shape; advice covers
# the AgriAdvice pairs. Route matters: Luna emits about half as many reasoning tokens through
# OpenRouter as it does through OpenAI direct, so the same model costs different amounts on
# different routes even at identical per-token prices.
MEASURED_TOKENS = {
    "frontier-grok-4-3":            {"mcq_in": 347.0, "mcq_out": 8.0, "advice_in": 219.0, "advice_out": 477.2},
    "frontier-gpt-5-6-luna":        {"mcq_in": 164.2, "mcq_out": 543.5, "advice_in": 33.2, "advice_out": 867.0},
    "frontier-nova-2-lite":         {"mcq_in": 200.1, "mcq_out": 15.0, "advice_in": 73.0, "advice_out": 512.0},
    "frontier-nemotron-nano-3-30b": {"mcq_in": 184.4, "mcq_out": 10.0, "advice_in": 43.8, "advice_out": 512.0},
}
# Luna's reasoning length varies a lot item to item: a mean of 543 with a maximum of 1564
# over the same ten prompts. Its line of the estimate is therefore the least stable, and the
# real bill for that model can run meaningfully above the figure below.
# Three of the four hit the advice output ceiling exactly, so their advice output is the
# budget rather than the model's natural length. Lower ADVICE_MAX_TOKENS and that line of
# the bill falls proportionally; raise it and it rises.
ADVICE_CEILING_DEFAULT = 512

DISPLAY = {
    "frontier-grok-4-3": "Grok 4.3",
    "frontier-gpt-5-6-luna": "GPT-5.6 Luna",
    "frontier-nova-2-lite": "Nova 2 Lite",
    "frontier-nemotron-nano-3-30b": "Nemotron Nano 3 30B",
}

COLUMNS = [
    "model_display_name", "route_scenario", "requests_total",
    "input_tokens_total", "output_tokens_total",
    "input_cost_usd", "output_cost_usd", "total_cost_usd",
    "cost_per_aws_account_usd", "price_input_usd_per_million", "price_output_usd_per_million",
    "prices_verified_on",
]


@dataclass
class Workload:
    frozen_test_items: int
    advice_pairs: int
    rotation_subset: int
    swap_subset: int

    @property
    def mcq_requests(self) -> int:
        # the main pass, two rotations, and the swapped half of the counterfactual pairs;
        # the unswapped half reuses answers the main pass already bought
        return self.frozen_test_items + 2 * self.rotation_subset + self.swap_subset

    @property
    def advice_requests(self) -> int:
        return 2 * self.advice_pairs

    @property
    def total_requests(self) -> int:
        return self.mcq_requests + self.advice_requests


def measure_workload(advice_ceiling: int) -> Workload:
    test = read_jsonl(TEST_INSTANCES_FROZEN)
    pairs = read_jsonl(AGRIADVICE_PAIRS)
    if not test or not pairs:
        raise SystemExit("Run the dataset_prep stage first; the split and the pairs must exist.")
    def capped(env_name: str, available: int) -> int:
        """The subset cap if one is set and positive, else everything available.

        The string "0" is truthy, so this cannot be written as a plain `or` chain: doing
        that silently costs a full run its entire workload and reports a bill of zero."""
        try:
            requested = int(os.environ.get(env_name, "0") or 0)
        except ValueError:
            requested = 0
        return min(requested, available) if requested > 0 else available

    return Workload(
        frozen_test_items=capped("FRONTIER_EVAL_SUBSET_SIZE", len(test)),
        advice_pairs=capped("FRONTIER_ADVICE_SUBSET_SIZE", len(pairs)),
        rotation_subset=min(int(os.environ.get("FRONTIER_ROTATION_SUBSET_SIZE", "150")), len(test)),
        swap_subset=min(int(os.environ.get("FRONTIER_SWAP_SUBSET_SIZE", "150")), len(test)),
    )


def token_totals(tier: str, work: Workload, advice_ceiling: int):
    t = dict(MEASURED_TOKENS[tier])
    # the three models that hit the ceiling scale with it; the one that did not is unchanged
    if t["advice_out"] >= ADVICE_CEILING_DEFAULT:
        t["advice_out"] = float(advice_ceiling)
    input_tokens = work.mcq_requests * t["mcq_in"] + work.advice_requests * t["advice_in"]
    output_tokens = work.mcq_requests * t["mcq_out"] + work.advice_requests * t["advice_out"]
    return input_tokens, output_tokens


def cost(input_tokens: float, output_tokens: float, price: Dict[str, float]):
    return (input_tokens / 1e6) * price["input"], (output_tokens / 1e6) * price["output"]


def main():
    ap = argparse.ArgumentParser(description="Frontier panel cost estimate")
    ap.add_argument("--advice-max-tokens", type=int, default=int(os.environ.get("FRONTIER_ADVICE_MAX_TOKENS", "512")),
                    help="output ceiling for AgriAdvice answers; the dominant cost driver")
    args = ap.parse_args()

    work = measure_workload(args.advice_max_tokens)
    logger.info("Workload per model: %d multiple-choice requests + %d advice requests = %d total.",
                work.mcq_requests, work.advice_requests, work.total_requests)
    logger.info("  (%d frozen-test items, %d x %d rotation, %d swapped, %d advice pairs x 2)",
                work.frozen_test_items, 2, work.rotation_subset, work.swap_subset, work.advice_pairs)

    rows: List[Dict] = []
    primary_total = 0.0
    bedrock_total = 0.0
    by_route: Dict[str, float] = {}
    for tier, display in DISPLAY.items():
        input_tokens, output_tokens = token_totals(tier, work, args.advice_max_tokens)
        for scenario, table in (("primary", PRIMARY_PRICES), ("fallback_openrouter", FALLBACK_PRICES)):
            price = table.get(tier)
            if price is None:
                continue
            in_cost, out_cost = cost(input_tokens, output_tokens, price)
            total = in_cost + out_cost
            if scenario == "primary":
                primary_total += total
                by_route[price["route"]] = by_route.get(price["route"], 0.0) + total
                if tier in BEDROCK_ROUTED:
                    bedrock_total += total
            rows.append({
                "model_display_name": display,
                "route_scenario": f'{scenario}:{price["route"]}',
                "requests_total": work.total_requests,
                "input_tokens_total": int(input_tokens), "output_tokens_total": int(output_tokens),
                "input_cost_usd": round(in_cost, 4), "output_cost_usd": round(out_cost, 4),
                "total_cost_usd": round(total, 4),
                "cost_per_aws_account_usd": round(total / 2, 4)
                if (scenario == "primary" and tier in BEDROCK_ROUTED) else "",
                "price_input_usd_per_million": price["input"],
                "price_output_usd_per_million": price["output"],
                "prices_verified_on": PRICE_VERIFIED_ON,
            })

    out_path = RESULTS_DIR / "frontier_panel_cost_estimate.csv"
    write_csv(out_path, rows, COLUMNS)

    print(f"\nWorkload per model: {work.total_requests} requests "
          f"({work.mcq_requests} multiple-choice, {work.advice_requests} advice)")
    print(f"Advice output ceiling: {args.advice_max_tokens} tokens (the dominant cost driver)\n")
    print(f"{'Model':<22s}{'Primary route':<20s}{'Primary':>10s}{'If it falls to OpenRouter':>28s}")
    for tier, display in DISPLAY.items():
        p = next(r for r in rows if r["model_display_name"] == display
                 and r["route_scenario"].startswith("primary"))
        f = next((r for r in rows if r["model_display_name"] == display
                  and r["route_scenario"].startswith("fallback")), None)
        print(f"  {display:<20s}{p['route_scenario'].split(':')[1]:<20s}"
              f"{'$%.2f' % p['total_cost_usd']:>10s}"
              f"{('$%.2f' % f['total_cost_usd']) if f else 'no fallback route':>28s}")

    print(f"\n  Full panel run on the primary routes: ${primary_total:.2f}")
    print("\n  Billed to:")
    for route, amount in sorted(by_route.items(), key=lambda kv: -kv[1]):
        if route == "aws-bedrock":
            print(f"    {'AWS Bedrock':<22s} ${amount:6.2f}   -> ${amount / 2:.2f} per AWS account "
                  f"(round-robined across the two)")
        else:
            print(f"    {route:<22s} ${amount:6.2f}")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
