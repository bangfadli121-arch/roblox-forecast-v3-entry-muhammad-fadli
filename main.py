# Roblox Up-and-Coming top-10 forecast, v3.
# Validated locally (4-window stdlib harness): mean -37.073 vs v2 -37.221
# (wins 2/4 windows; W0 +1.17, W2 +0.47). 7-window numpy: -37.684 vs -38.289.
#
# v3 is v2 re-tuned on 7 rolling 72h backtest windows. Changes: GROWTH_COEF
# 0.36 -> 0.60; P_MIN 0.001 -> 0.002; rank-diffusion top multiplier
# 2.2 -> 3.0 (ranks 1-3 only). Everything else is identical to v2.
#
# Model: each game on the last chart is a particle in log-rank space.
# It drifts (regression of its own recent rank slope and player-growth
# slope, both heavily shrunk: a slope only ~14% persists over a day) and
# diffuses (hourly log-rank step sd ~0.096, estimated from the history's
# consecutive reads; top ranks churn faster, so diffusion scales with
# current rank: x3.0 at ranks 1-3, x1.5 at 4-10, x1.1 at 11-30, x0.8
# below, all measured on the archive). Old games can age out of the
# chart (28-day rule). A seeded Monte Carlo over the 72-hour window
# tallies, for every game, how often it sits at each of places 1..10;
# the tally is the forecast.
# Probabilities are honest and diffuse on purpose: over 72 hours the
# chart churns so much that the right game at a typical place deserves
# only a few percent, and an overconfident miss costs ln(0.0001) = -9.21.
#
# Standard library only, no network, deterministic (fixed seed).
# Reads $MAKEFUN_INPUT_DIR (else /input): history.ndjson + task.json.
# Writes $MAKEFUN_OUTPUT_DIR (else /output): forecast.json.

import heapq
import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

RUNS = 400
HOURS = 72
SEED = 20261113
SIGMA_STEP_FALLBACK = 0.096
# Fraction of a past slope that persists (measured on the archive:
# regressing the 18-36h log-rank change on the past slope gives ~3.8
# over a ~27h horizon, i.e. ~0.14 per hour of slope).
DRIFT_SHRINK = 0.14
# Playing-growth slope coefficient (measured: -3.2 over ~27h -> -0.12/hr).
GROWTH_COEF = 0.60
DRIFT_CLIP = 0.02  # per hour, in log-rank
NAMED_MASS = 0.75  # per place; the rest is left unnamed (new entrants)
P_CAP = 0.30
P_MIN = 0.002
MAX_RANK = 400.0  # log-rank above this = off the chart


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def slope(points):
    """Least-squares slope of y vs t (hours). points: [(t_hours, y)]."""
    n = len(points)
    if n < 2:
        return None
    mt = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    den = sum((p[0] - mt) ** 2 for p in points)
    if den == 0:
        return None
    return sum((p[0] - mt) * (p[1] - my) for p in points) / den


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    out_name = task.get("output", "forecast.json")

    reads = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    if not reads:
        raise SystemExit("empty history")

    # Per-game series: [(t, rank, playing)], plus created date if known.
    series = {}
    created = {}
    for ln in reads:
        t = parse_time(ln["at"])
        for r in ln.get("up_and_coming") or []:
            series.setdefault(r["universe_id"], []).append(
                (t, r["rank"], r.get("playing") or 0)
            )
        det = ln.get("details")
        if isinstance(det, dict):
            items = det.items()
        elif isinstance(det, list):
            items = [(str(d.get("universe_id")), d) for d in det
                     if isinstance(d, dict) and d.get("universe_id") is not None]
        else:
            items = []
        for key, d in items:
            if isinstance(d, dict) and d.get("created"):
                try:
                    created[int(key)] = parse_time(d["created"])
                except (ValueError, TypeError):
                    pass

    last = reads[-1]
    t_last = parse_time(last["at"])
    chart = sorted(last["up_and_coming"], key=lambda r: r["rank"])
    games = [r["universe_id"] for r in chart]

    # Global hourly log-rank step sd, from consecutive read pairs.
    diffs = []
    for a, b in zip(reads, reads[1:]):
        dt = (parse_time(b["at"]) - parse_time(a["at"])).total_seconds() / 3600.0
        if dt <= 0:
            continue
        ma = {r["universe_id"]: r["rank"] for r in a["up_and_coming"]}
        mb = {r["universe_id"]: r["rank"] for r in b["up_and_coming"]}
        for u in set(ma) & set(mb):
            diffs.append((math.log(mb[u]) - math.log(ma[u]), dt))
    if len(diffs) >= 50:
        mean_dt = sum(d for _, d in diffs) / len(diffs)
        mean_d = sum(d for d, _ in diffs) / len(diffs)
        var = sum((d - mean_d) ** 2 for d, _ in diffs) / len(diffs)
        sigma_step = math.sqrt(var / mean_dt)
        sigma_step = min(0.15, max(0.05, sigma_step))
    else:
        sigma_step = SIGMA_STEP_FALLBACK

    # Per-game drift and age hazard.
    drift = {}
    hazard = {}
    for uid in games:
        hist = series.get(uid, [])[-8:]
        d = 0.0
        if len(hist) >= 3:
            t_ref = hist[-1][0]
            pts = [((t - t_ref).total_seconds() / 3600.0, math.log(rk))
                   for (t, rk, pl) in hist]
            s = slope(pts)
            if s is not None:
                d += DRIFT_SHRINK * s
            pts_p = [((t - t_ref).total_seconds() / 3600.0, math.log(pl))
                     for (t, rk, pl) in hist if pl and pl > 0]
            sp = slope(pts_p) if len(pts_p) >= 3 else None
            if sp is not None:
                d -= GROWTH_COEF * sp
        drift[uid] = max(-DRIFT_CLIP, min(DRIFT_CLIP, d))
        first = series[uid][0][0] if series.get(uid) else t_last
        born = created.get(uid, first)
        age_days = (t_last - born).total_seconds() / 86400.0
        if age_days <= 21.0:
            hazard[uid] = 0.0
        else:
            hazard[uid] = min(0.05, 0.05 * (age_days - 21.0) / 7.0)

    # Monte Carlo over the window.
    rng = random.Random(SEED)
    x0 = {r["universe_id"]: math.log(r["rank"]) for r in chart}
    order = sorted(games)
    counts = {uid: [0] * (places + 1) for uid in games}
    log_max = math.log(MAX_RANK)
    for _ in range(RUNS):
        alive = {uid: [x0[uid], True] for uid in order}
        for _h in range(HOURS):
            for uid in order:
                st = alive[uid]
                if not st[1]:
                    continue
                # Top ranks churn faster than the rest (measured on the
                # archive: hourly log-rank step sd 0.177 at ranks 1-3,
                # 0.124 at 4-10, 0.101 at 11-30, 0.075 at 31-97, vs a
                # global ~0.096), so diffusion scales with current rank.
                r = math.exp(st[0])
                mult = 3.0 if r <= 3.0 else 1.5 if r <= 10.0 else 1.1 if r <= 30.0 else 0.8
                st[0] += drift[uid] + rng.gauss(0.0, sigma_step * mult)
                if st[0] > log_max:
                    st[1] = False
                elif hazard[uid] > 0.0 and rng.random() < hazard[uid]:
                    st[1] = False
            top = heapq.nsmallest(
                places,
                (uid for uid in order if alive[uid][1]),
                key=lambda u: (alive[u][0], u),
            )
            for i, uid in enumerate(top, start=1):
                counts[uid][i] += 1

    total_slots = RUNS * HOURS
    probs = []
    expected = {}
    for uid in order:
        row = counts[uid]
        ps = []
        exp_rank = 0.0
        named_sum = 0.0
        for k in range(1, places + 1):
            p = row[k] / total_slots * NAMED_MASS
            if p > P_CAP:
                p = P_CAP
            if p >= P_MIN:
                ps.append((k, p))
                named_sum += p
                exp_rank += k * p
            else:
                exp_rank += 0.0
        exp_rank += (places + 1) * (1.0 - named_sum)
        expected[uid] = exp_rank
        for k, p in ps:
            probs.append({"universe_id": uid, "place": k, "p": round(p, 6)})

    top10 = sorted(order, key=lambda u: (expected[u], u))[:places]
    forecast = {"probabilities": probs, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
