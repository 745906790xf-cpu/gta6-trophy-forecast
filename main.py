"""Forecast GTA VI trophy earn rates on Exophase (make.fun gta6-trophy-rarity-forecast-01).

Model (offline, deterministic, standard library only):

An earn rate is earners / players tracked. Both counts only grow, but at
different speeds: the tracked-player denominator surges after launch and
dilutes every rate, while each trophy's earners grow as players progress.
So this program forecasts the two counts separately and divides, instead
of extrapolating the rate alone:

  * players tracked: log-log trend in hours since 2026-11-19T00:00Z over
    the distinct refreshes in the case history (repeated identical reads
    between Exophase refreshes are collapsed first), with a launch-growth
    prior when the history holds too few refreshes to fit;
  * earners per trophy: the same log-log trend per trophy, its slope
    shrunk toward the pooled slope across trophies, floored at the last
    observed earners; with too few refreshes, earners grow at a
    rate-dependent pass-through of the player growth (common trophies
    keep up with new players, rare ones lag);
  * the median rate blends the counts-implied rate, a logit-rate trend
    and persistence, weighted by how many refreshes the history holds.

The spread is on the logit scale: a random-walk term growing with the
horizon, plus a refresh-jump term (Exophase updates in batches, so rates
step rather than glide; its size is estimated from the steps between
distinct reads, with a prior when none are observed), plus the trend
fit's residual and a small-count term for trophies with few earners.

Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output):
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)


def hours_since_epoch(iso_text):
    moment = datetime.fromisoformat(iso_text.replace("Z", "+00:00"))
    return (moment - EPOCH).total_seconds() / 3600.0


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(value):
    return 100.0 / (1.0 + math.exp(-value))


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def fit_line(xs, ys):
    """Least-squares y = a + b*x; returns (a, b, residual_sd) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / n) if n else 0.0
    return (intercept, slope, sd)


def median_of(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def distinct_states(history):
    """Collapse consecutive identical reads (same players + rates)."""
    states = []
    seen = set()
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key in seen:
            continue
        seen.add(key)
        states.append(read)
    return states


def last_rate_for(history, idx):
    for read in reversed(history):
        rates = read.get("rates") or []
        if idx < len(rates) and rates[idx] is not None:
            return float(rates[idx])
    return 0.0


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected levels %r" % (case.get("levels"),))
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    trophies = case.get("trophies") or []
    horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
    target_hours = hours_since_epoch(case["target"])
    states = distinct_states(history)
    n_states = len(states)
    state_hours = [hours_since_epoch(s["at"]) for s in states]
    state_players = [float(s.get("players_tracked") or 0) for s in states]
    last_players = state_players[-1] if state_players else 0.0
    age_days = max(0.0, state_hours[-1] / 24.0) if state_hours else 0.0
    log_target = math.log1p(max(0.0, target_hours))

    # ---- players tracked at the target ----
    players_fit = None
    if n_states >= 3 and last_players > 0:
        xs = [math.log1p(max(0.0, h)) for h in state_hours]
        ys = [math.log(max(1.0, p)) for p in state_players]
        players_fit = fit_line(xs, ys)
    if players_fit is not None:
        a, b, _sd = players_fit
        b = min(3.0, max(0.0, b))
        players_target = math.exp(a + b * log_target)
        # Blend with a damped last-24h growth view so one fit cannot run away.
        recent_log_growth = 0.0
        cutoff = state_hours[-1] - 24.0
        recent = [(h, p) for h, p in zip(state_hours, state_players) if h >= cutoff and p > 0]
        if len(recent) >= 2:
            span_days = max(0.25, (recent[-1][0] - recent[0][0]) / 24.0)
            recent_log_growth = math.log(recent[-1][1] / recent[0][1]) / span_days
        elif n_states >= 2 and state_players[0] > 0:
            span_days = max(0.25, (state_hours[-1] - state_hours[0]) / 24.0)
            recent_log_growth = math.log(last_players / state_players[0]) / span_days
        damped = last_players * math.exp(min(1.6, max(0.0, recent_log_growth) * horizon_days * 0.6))
        players_target = math.sqrt(max(1.0, players_target) * max(1.0, damped))
    else:
        prior_log_growth = 2.4 * math.sqrt(max(horizon_days, 0.0)) / (1.0 + 0.35 * age_days)
        prior_log_growth = min(2.2, max(0.0, prior_log_growth))
        players_target = last_players * math.exp(prior_log_growth)
    players_target = max(last_players, min(players_target, last_players * math.exp(2.5) + 1.0))
    players_log_growth = math.log(players_target / last_players) if last_players > 0 else 0.0

    # ---- per-trophy earners, trends, jump size ----
    earners_hist = []
    for idx in range(len(trophies)):
        series = []
        for s_idx, state in enumerate(states):
            rates = state.get("rates") or []
            rate = rates[idx] if idx < len(rates) else None
            if rate is None:
                continue
            series.append((state_hours[s_idx], float(rate) / 100.0 * state_players[s_idx], float(rate)))
        earners_hist.append(series)

    slopes = []
    fits = []
    for series in earners_hist:
        fit = None
        if len(series) >= 3:
            xs = [math.log1p(max(0.0, h)) for h, _e, _r in series]
            ys = [math.log1p(max(0.0, e)) for _h, e, _r in series]
            fit = fit_line(xs, ys)
        fits.append(fit)
        if fit is not None:
            slopes.append(fit[1])
    pooled_slope = median_of(slopes) if slopes else 1.0

    logit_steps = []
    for series in earners_hist:
        for k in range(1, len(series)):
            logit_steps.append(logit(series[k][2]) - logit(series[k - 1][2]))
    if len(logit_steps) >= 8:
        jump_sd = median_of([abs(s) for s in logit_steps]) / 0.6745
        jump_sd = min(1.6, max(0.15, jump_sd))
    else:
        jump_sd = 0.8

    if n_states >= 4:
        w_counts, w_trend, w_persist = 0.55, 0.30, 0.15
    elif n_states >= 2:
        w_counts, w_trend, w_persist = 0.45, 0.15, 0.40
    else:
        w_counts, w_trend, w_persist = 0.60, 0.0, 0.40

    rows = []
    for idx, trophy in enumerate(trophies):
        series = earners_hist[idx]
        last_rate = last_rate_for(history, idx)
        last_earners = series[-1][1] if series else last_rate / 100.0 * last_players

        fit = fits[idx]
        if fit is not None and series:
            a, b, resid_sd = fit
            n_pts = len(series)
            b_shrunk = (n_pts * b + 3.0 * pooled_slope) / (n_pts + 3.0)
            b_shrunk = min(3.5, max(0.0, b_shrunk))
            earners_target = math.expm1(a + b_shrunk * log_target)
        else:
            resid_sd = 0.0
            if last_rate >= 20.0:
                passthrough = 0.90
            elif last_rate >= 5.0:
                passthrough = 0.75
            elif last_rate >= 1.0:
                passthrough = 0.60
            else:
                passthrough = 0.45
            earners_target = math.expm1(math.log1p(max(0.0, last_earners)) + passthrough * players_log_growth)
        earners_target = max(last_earners, min(earners_target, players_target))
        rate_counts = 100.0 * earners_target / players_target if players_target > 0 else last_rate

        trend_logit = logit(last_rate)
        if len(series) >= 3:
            # Winsorize logits for fitting: a rate printed as 0% or 100%
            # clamps to +-9.2 and would otherwise dominate the slope.
            xs = [math.log1p(max(0.0, h)) for h, _e, _r in series]
            ys = [min(4.6, max(-4.6, logit(r))) for _h, _e, r in series]
            trend_fit = fit_line(xs, ys)
            if trend_fit is not None:
                ta, tb, _tsd = trend_fit
                tb = min(2.0, max(-2.0, tb))
                trend_logit = ta + tb * log_target

        median_logit = (
            w_counts * logit(rate_counts)
            + w_trend * trend_logit
            + w_persist * logit(last_rate)
        )
        # No median may drift far from the last read: beyond the cap the
        # extrapolation, not the data, is doing the talking. The cap is
        # wider at the 0%/100% boundary, where logit units stretch a
        # fraction of a point into several units.
        last_logit = logit(last_rate)
        drift_cap = 4.0 if (last_rate <= 0.5 or last_rate >= 99.5) else 1.6
        median_logit = min(last_logit + drift_cap, max(last_logit - drift_cap, median_logit))

        sigma_rw = 0.5 * math.sqrt(max(horizon_days, 0.0))
        sigma_jump = jump_sd * math.sqrt(1.0 + max(horizon_days, 0.0))
        sigma_resid = min(1.2, resid_sd)
        sigma_count = 0.35 / math.sqrt(1.0 + max(0.0, last_earners))
        sigma = math.sqrt(sigma_rw ** 2 + sigma_jump ** 2 + sigma_resid ** 2 + sigma_count ** 2)
        sigma = min(1.65, max(0.35, sigma))

        quantiles = [round4(inv_logit(median_logit + z * sigma)) for z in Z]
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        quantiles = [min(100, max(0, q)) for q in quantiles]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
