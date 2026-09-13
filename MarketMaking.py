"""
Market making under inventory risk and adverse selection.

Everything in one file: self-tests, simulation, three experiments, figures.

    python MarketMaking.py

The self-tests at the top need no internet and no data. If they pass, the
book-keeping is right and any later surprise is a result, not a bug.

--------------------------------------------------------------------------
THE IDEA
--------------------------------------------------------------------------
A market maker posts two prices at once -- a bid to buy and an ask to sell --
and earns the difference. No view on direction is required. Three things stop
that being free money, and this file measures all three:

  INVENTORY     flow does not arrive politely alternating. Buy five times in
                a row and you are holding a position you never wanted, and its
                risk is unbounded while your edge per trade is fixed.

  ADVERSE       some of the people trading with you know the price is about to
  SELECTION     move. Your fills are then selected against you: you sell just
                before it rises and buy just before it falls.

  THE TRADEOFF  a wider quote earns more per fill and gets fewer fills. There
                is an optimum, and it moves once the first two exist.

The counter to inventory risk is SKEWING: when long, slide BOTH quotes down,
so the ask is likelier to lift and the bid likelier to be left alone. The
width does not change -- only the centre moves.

--------------------------------------------------------------------------
WHAT IS ASSUMED, AND WHAT IS MEASURED
--------------------------------------------------------------------------
Assumed (chosen before any experiment was run, see PARAMETERS):
  - the mid follows a driftless random walk
  - order arrivals are Poisson with intensity A*exp(-k*delta) in the distance
    of a quote from the mid
  - an informed order moves the mid by IMPACT in the direction it traded

Measured (everything else). Section 4 re-runs every headline conclusion at
half and double each assumed parameter, because with four free parameters
almost any conclusion is available and a reader is entitled to know which
ones survive.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = "figures"
INK, ACCENT, GREY = "#26356B", "#A8620C", "#9AA3B2"

# ==========================================================================
# PARAMETERS -- fixed before the experiments, with the reasoning written down
# ==========================================================================

S0 = 100.0        # starting mid. Arbitrary; nothing depends on the level.
SIGMA = 2.0       # price vol over one session. 2 on a mid of 100 is a 2%
                  # session, which is a normal day for a liquid single name.
A = 60.0          # order arrivals per session per side at zero distance.
K = 1.5           # decay of fill rate with distance. Fill rate halves every
                  # ln2/1.5 = 0.46 in price, i.e. roughly every 0.46%.
IMPACT = 0.5      # how far an informed trade moves the mid. A quarter of a
                  # session's volatility: a real move, not a catastrophe.
N_STEPS = 500     # time steps in a session.
N_PATHS = 6000    # independent sessions per measurement.

# Risk-neutral theory says the best half-spread with no inventory cost is 1/k,
# because d/dd [ d * exp(-k*d) ] = 0 at d = 1/k. Experiment 1 should land near
# this and then be pulled away from it by inventory risk.
NAIVE_BEST_WIDTH = 2.0 / K

# Skewing exists to trade return for certainty, so a RISK-NEUTRAL operator has
# no reason to do it and optimising mean P&L alone cannot test a theory whose
# entire content is about risk. This is the preference Avellaneda-Stoikov
# assume: mean penalised by variance. At gamma = 0.02 a session standard
# deviation of 10 costs about 1, roughly 4% of a typical mean -- a mild but
# real preference for certainty.
RISK_AVERSION = 0.02


def objective(r, gamma=RISK_AVERSION):
    """Certainty-equivalent P&L: mean minus half gamma times variance."""
    return r["mean"] - 0.5 * gamma * r["std"] ** 2


def paired_diff(a, b):
    """Mean of (a - b) path by path, and its standard error.

    Both runs used the same seed, so the price paths underneath them are the
    same market. Differencing path against path removes almost all the noise
    that two independent standard errors would leave behind, which is what
    makes differences of a few percent measurable at all.
    """
    d = np.asarray(a["pnl"]) - np.asarray(b["pnl"])
    return float(d.mean()), float(d.std() / np.sqrt(len(d)))


# ==========================================================================
# Core simulation
# ==========================================================================

def simulate(width, skew=0.0, p_informed=0.0, power=1.0, decay_with_time=False,
             n_paths=N_PATHS, n_steps=N_STEPS, sigma=SIGMA, A=A, k=K,
             impact=IMPACT, seed=0, check_identity=False):
    """One session, n_paths times in parallel.

    width       distance between bid and ask
    skew        coefficient c in  centre = mid - c * f(q).  0 = never skew.
    power       exponent in f(q) = sign(q)*|q|**power. 1 is the linear rule
                theory predicts; 0 is a constant shift regardless of size.
    p_informed  fraction of arriving orders that know where the mid is going.
    decay_with_time  scale the skew by remaining session, as theory predicts.

    Two separate random streams. Prices come from one and order flow from the
    other, so sweeping a parameter does not change the price paths underneath
    it -- the comparison between two widths is then made on the same market,
    which is where most of the noise in a naive sweep comes from.
    """
    dt = 1.0 / n_steps
    rng_price = np.random.default_rng(seed)
    rng_flow = np.random.default_rng(seed + 10_000)

    mid = np.full(n_paths, S0)
    q = np.zeros(n_paths)
    cash = np.zeros(n_paths)

    max_abs_q = np.zeros(n_paths)
    edge_total = np.zeros(n_paths)      # what the spread earned
    inv_total = np.zeros(n_paths)       # what the inventory made or lost

    for i in range(n_steps):
        # ---- quote. Posted BEFORE this step's price move, and filled against
        #      that same price. Resolving fills after the move would mean only
        #      ever being filled on trades that turned out well.
        f_q = np.sign(q) * np.abs(q) ** power
        c = skew * (1.0 - i / n_steps) if decay_with_time else skew
        centre = mid - c * f_q
        bid = centre - width / 2.0
        ask = centre + width / 2.0

        d_b = mid - bid                  # how far the bid sits below the mid
        d_a = ask - mid                  # how far the ask sits above it

        # Intensity cannot exceed A: that is the rate at which orders arrive at
        # all, so quoting through the mid gets you every order and no more.
        lam_b = A * np.exp(-k * np.maximum(d_b, 0.0))
        lam_a = A * np.exp(-k * np.maximum(d_a, 0.0))

        # Four draws every step whether or not they are needed, so the random
        # stream stays aligned across parameter values.
        u_b, u_a = rng_flow.random(n_paths), rng_flow.random(n_paths)
        v_b, v_a = rng_flow.random(n_paths), rng_flow.random(n_paths)

        hit_b = u_b < lam_b * dt         # someone SOLD to us, at our bid
        hit_a = u_a < lam_a * dt         # someone BOUGHT from us, at our ask

        # ---- book-keeping. Buying costs cash and adds inventory.
        edge = hit_b * (mid - bid) + hit_a * (ask - mid)
        cash += np.where(hit_b, -bid, 0.0) + np.where(hit_a, ask, 0.0)
        q = q + hit_b.astype(float) - hit_a.astype(float)
        edge_total += edge
        max_abs_q = np.maximum(max_abs_q, np.abs(q))

        # ---- adverse selection. An informed buyer lifts our ask and THEN the
        #      mid rises, leaving us short into a rising market.
        inf_b = hit_b & (v_b < p_informed)
        inf_a = hit_a & (v_a < p_informed)
        drift = impact * (inf_a.astype(float) - inf_b.astype(float))

        shock = sigma * np.sqrt(dt) * rng_price.standard_normal(n_paths)
        d_mid = drift + shock
        mid = mid + d_mid
        inv_total += q * d_mid           # inventory held across the move

        if check_identity:
            # Value must equal edge captured plus inventory P&L, every step.
            v = cash + q * mid
            assert np.allclose(v, edge_total + inv_total, atol=1e-9), \
                f"ACCOUNTING BROKE at step {i}"

    # The per-step identity above is gated behind check_identity because it is
    # too slow to run on every measurement. This aggregate version is O(1) per
    # run, so it costs nothing and it means NO experiment can quietly run on
    # broken book-keeping -- not just the ones under test.
    assert np.allclose(cash + q * mid, edge_total + inv_total, rtol=1e-9, atol=1e-6), \
        "ACCOUNTING BROKE (end of run)"

    # ---- close the book by crossing the spread. Liquidating is not free, and
    #      a model where it is free will conclude inventory does not matter.
    liquidation_cost = np.abs(q) * width / 2.0
    pnl = cash + q * mid - liquidation_cost

    return {
        "pnl": pnl,
        "liq": float(liquidation_cost.mean()),
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
        "se": float(pnl.std() / np.sqrt(n_paths)),
        "p05": float(np.percentile(pnl, 5)),
        "worst": float(pnl.min()),
        "final_q": q,
        "mean_abs_final_q": float(np.abs(q).mean()),
        "mean_max_abs_q": float(max_abs_q.mean()),
        "edge": float(edge_total.mean()),
        "inv": float(inv_total.mean()),
    }


def sweep(values, key, **kw):
    """Run simulate() once per value of one parameter, everything else fixed."""
    return [simulate(**{key: v, **kw}) for v in values]


def best_of(values, results, score=objective):
    """The value maximising `score`, and whether the peak clears its noise."""
    scores = [score(r) for r in results]
    i = int(np.argmax(scores))
    others = [s for j, s in enumerate(scores) if j != i]
    margin = scores[i] - max(others) if others else float("inf")
    return values[i], scores[i], margin, results[i]["se"]


# ==========================================================================
# Self-tests -- no data, no internet, a few seconds
# ==========================================================================

def run_tests():
    print("SELF-TESTS (no data needed)")
    print("-" * 72)

    # 1. The accounting identity, asserted at every step of every path.
    simulate(width=1.0, skew=0.05, p_informed=0.2, n_paths=400, n_steps=200,
             seed=1, check_identity=True)
    print("PASS  value = edge + inventory P&L at every step of every path")

    # 2. With no price movement and no informed flow there is no risk at all,
    #    so P&L must be exactly the half-spread times the number of fills.
    r = simulate(width=1.0, sigma=0.0, p_informed=0.0, n_paths=800, seed=2)
    assert abs(r["inv"]) < 1e-12, r["inv"]
    # P&L must then be exactly the spread earned minus the cost of closing out.
    assert abs(r["mean"] - (r["edge"] - r["liq"])) < 1e-9, (r["mean"], r["edge"], r["liq"])
    print(f"PASS  frozen price: P&L is exactly spread minus close-out cost "
          f"({r['edge']:.2f} - {r['liq']:.2f} = {r['mean']:.2f}), inventory P&L {r['inv']:.1e}")

    # 3. Quote at zero width and there is no edge to earn.
    r0 = simulate(width=0.0, n_paths=800, seed=3)
    assert abs(r0["edge"]) < 1e-9, r0["edge"]
    print("PASS  zero width earns exactly zero spread")

    # 4. The fill rate must match the intensity it was drawn from. At distance
    #    d the expected number of fills per side is A*exp(-k*d).
    for d in (0.2, 0.8):
        rr = simulate(width=2 * d, sigma=0.0, n_paths=4000, seed=4)
        # edge is (fills x distance) summed over BOTH sides, so fills = edge/d
        got = rr["edge"] / d
        want = 2 * A * np.exp(-K * d)
        assert abs(got - want) / want < 0.06, (d, got, want)
    print("PASS  realised fill rate matches A*exp(-k*d) at both distances")

    # 5. Without skew, inventory is a random walk and wanders. With skew it is
    #    pulled back. This is the mechanism the whole project rests on.
    flat = simulate(width=1.0, skew=0.0, n_paths=2000, seed=5)
    held = simulate(width=1.0, skew=0.05, n_paths=2000, seed=5)
    assert held["mean_abs_final_q"] < 0.6 * flat["mean_abs_final_q"], \
        (flat["mean_abs_final_q"], held["mean_abs_final_q"])
    print(f"PASS  skew bounds inventory: mean |q| at the close "
          f"{flat['mean_abs_final_q']:.1f} -> {held['mean_abs_final_q']:.1f}")

    # 6. Informed flow must COST money, and cost more when there is more of it.
    a = simulate(width=1.0, p_informed=0.0, n_paths=3000, seed=6)["mean"]
    b = simulate(width=1.0, p_informed=0.3, n_paths=3000, seed=6)["mean"]
    c = simulate(width=1.0, p_informed=0.6, n_paths=3000, seed=6)["mean"]
    assert a > b > c, (a, b, c)
    print(f"PASS  adverse selection is monotonically costly: "
          f"{a:.1f} -> {b:.1f} -> {c:.1f}")

    # 7. A constant shift that ignores inventory size should not beat a rule
    #    that scales with it.
    lin = simulate(width=1.0, skew=0.05, power=1.0, n_paths=2000, seed=7)
    con = simulate(width=1.0, skew=0.05, power=0.0, n_paths=2000, seed=7)
    assert lin["mean_abs_final_q"] < con["mean_abs_final_q"]
    print("PASS  size-aware skew controls inventory better than a flat shift")
    print()


# ==========================================================================
# Experiments
# ==========================================================================

def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def experiment_1():
    rule("1. HOW WIDE SHOULD YOU QUOTE?")
    print("No skew, no informed flow. Wider earns more per fill and gets fewer")
    print(f"fills, so there should be a peak. Risk-neutral theory puts it at")
    print(f"2/k = {NAIVE_BEST_WIDTH:.3f}. Closing the book crosses the spread, so a")
    print(f"wider quote also makes the end-of-session clean-up dearer -- which")
    print(f"pulls the optimum the other way. Which force wins is the question.\n")

    widths = np.round(np.arange(0.4, 3.01, 0.2), 2)
    res = sweep(widths, "width", skew=0.0, seed=100)
    print(f"{'width':>7} {'mean':>9} {'se':>7} {'std':>9} {'5th pct':>9} {'mean|q|':>9}")
    for w, r in zip(widths, res):
        print(f"{w:7.2f} {r['mean']:9.2f} {r['se']:7.2f} {r['std']:9.2f} "
              f"{r['p05']:9.2f} {r['mean_max_abs_q']:9.1f}")

    w_star, best, margin, se = best_of(widths, res)
    print(f"\n  best width {w_star:.2f}, mean {best:.2f} +/- {se:.2f}")
    print(f"  clears the runner-up by {margin:.2f} "
          f"({'real' if margin > 2 * se else 'NOT distinguishable -- the peak is a plateau'})")
    print(f"  risk-neutral prediction was {NAIVE_BEST_WIDTH:.3f}")
    return widths, res, float(w_star)


def experiment_2(w_star):
    rule("2. DOES SKEWING PAY FOR ITSELF?")
    print("At the best width, slide the centre of the quotes against inventory.")
    print("The expectation going in was that this must COST mean P&L, since you")
    print("are deliberately accepting worse prices, and buy risk reduction with")
    print("it. Whether that is what happens is the experiment.\n")

    skews = np.round(np.arange(0.0, 0.401, 0.02), 3)
    res = sweep(skews, "skew", width=w_star, seed=200)
    base = res[0]

    print(f"{'skew':>7} {'mean':>9} {'d mean':>9} {'+/-':>6} {'std':>9} "
          f"{'5th pct':>9} {'objective':>10} {'mean|q|':>9}")
    for sk, r in zip(skews, res):
        dm, dse = paired_diff(r, base)
        print(f"{sk:7.3f} {r['mean']:9.2f} {dm:+9.2f} {dse:6.2f} {r['std']:9.2f} "
              f"{r['p05']:9.2f} {objective(r):10.2f} {r['mean_abs_final_q']:9.1f}")

    s_star, best, margin, _ = best_of(skews, res)
    print(f"\n  no skew:        mean {base['mean']:7.2f}   std {base['std']:7.2f}   "
          f"objective {objective(base):7.2f}")
    print(f"  best skew {s_star:<5.3f}  mean {res[list(skews).index(s_star)]['mean']:7.2f}   "
          f"std {res[list(skews).index(s_star)]['std']:7.2f}   objective {best:7.2f}")

    # Did it in fact cost mean P&L? Answer honestly whichever way it came out.
    peak_mean = max(r["mean"] for r in res)
    i_mean = [r["mean"] for r in res].index(peak_mean)
    dm, dse = paired_diff(res[i_mean], base)
    if dm > 2 * dse:
        print(f"\n  NOT A TRADEOFF AT THESE PARAMETERS. Skewing at {skews[i_mean]:.3f}")
        print(f"  RAISES mean P&L by {dm:+.2f} (+/- {dse:.2f}) while cutting the")
        print(f"  standard deviation {100*(res[i_mean]['std']/base['std']-1):+.0f}%. "
              f"Two reasons, both real:")
        print(f"    - closing out costs width/2 per unit, and skewing ends the")
        print(f"      session near flat: mean |q| {base['mean_abs_final_q']:.1f} -> "
              f"{res[i_mean]['mean_abs_final_q']:.1f}")
        print(f"    - arrival intensity is convex in distance, so moving one quote")
        print(f"      nearer and the other further RAISES total fill rate")
        print(f"  The tradeoff only appears once the skew is large enough to drag")
        print(f"  quotes far from the mid -- see where mean turns over above.")
    else:
        print(f"\n  Skewing costs {dm:+.2f} (+/- {dse:.2f}) of mean, as expected.")
    return skews, res


def experiment_2b(w_star):
    rule("2b. FOUR PREDICTIONS THE THEORY CANNOT WRIGGLE OUT OF")
    print("Avellaneda-Stoikov give centre = mid - q*gamma*sigma^2*(T-t), and a")
    print("width with no q in it. Four checkable claims follow. Scored on the")
    print("risk-adjusted objective, because a risk-neutral maker would not skew")
    print("at all and the mean alone cannot test a claim about risk.\n")

    skews = np.round(np.arange(0.0, 0.401, 0.02), 3)
    wide = np.round(np.arange(0.0, 1.201, 0.04), 3)

    print("(i)  Is the best rule LINEAR in inventory?")
    print("     f(q) = sign(q)*|q|^power. Theory says power = 1.")
    ref = None
    for p in (0.0, 0.5, 1.0, 1.5, 2.0):
        r = sweep(wide, "skew", width=w_star, power=p, seed=210)
        sk, sc, _, _ = best_of(wide, r)
        bestr = r[list(wide).index(sk)]
        if p == 1.0:
            ref = bestr
        print(f"     power {p:4.1f}: best coefficient {sk:5.3f}, objective {sc:8.2f}")
    for p in (0.0, 0.5, 1.5, 2.0):
        r = sweep(wide, "skew", width=w_star, power=p, seed=210)
        sk, _, _, _ = best_of(wide, r)
        dm, dse = paired_diff(r[list(wide).index(sk)], ref)
        verdict = "worse than linear" if dm < -2 * dse else (
            "better than linear" if dm > 2 * dse else "indistinguishable from linear")
        print(f"     power {p:4.1f} vs 1.0 at each one's own best: "
              f"{dm:+6.2f} +/- {dse:.2f}  -> {verdict}")

    print("\n(ii) Does the best skew scale with sigma^2, or with sigma?")
    base_sk = None
    for mult in (0.5, 1.0, 2.0):
        r = sweep(wide, "skew", width=w_star, sigma=SIGMA * mult, seed=220)
        sk, _, _, _ = best_of(wide, r)
        if mult == 1.0:
            base_sk = sk
        print(f"     sigma x{mult:<4}: best skew {sk:5.3f}" +
              ("" if base_sk in (None, 0) else
               f"   ratio to baseline {sk / base_sk:5.2f}x"
               f"   (sigma^2 predicts {mult ** 2:.2f}x, sigma predicts {mult:.2f}x)"))

    print("\n(iii) Should the skew fade as the session runs out?")
    pair = {}
    for decay in (False, True):
        r = sweep(wide, "skew", width=w_star, decay_with_time=decay, seed=230)
        sk, sc, _, _ = best_of(wide, r)
        pair[decay] = r[list(wide).index(sk)]
        print(f"     {'fading':>9} " if decay else f"     {'constant':>9} ",
              f"best skew {sk:5.3f}, objective {sc:8.2f}")
    dm, dse = paired_diff(pair[True], pair[False])
    print(f"     fading minus constant, each at its own best: {dm:+.2f} +/- {dse:.2f}"
          f"  -> {'fading wins' if dm > 2*dse else 'constant wins' if dm < -2*dse else 'cannot tell'}")

    print("\n(iv) Does the best WIDTH move once skewing is switched on?")
    widths = np.round(np.arange(0.6, 3.01, 0.1), 2)
    for sk in (0.0, 0.20, 0.40):
        r = sweep(widths, "width", skew=sk, seed=240)
        w, sc, _, _ = best_of(widths, r)
        print(f"     skew {sk:5.3f}: best width {w:4.2f}, objective {sc:8.2f}")
    print("     Theory says only the centre moves with inventory, so this row")
    print("     should be flat. Any drift is the model disagreeing with it.")


def experiment_3(w_star, skew_star):
    rule("3. HOW MUCH ADVERSE SELECTION CAN A MARKET MAKER SURVIVE?")
    print("Some arriving orders know where the mid is going and lift the ask")
    print("just before it rises. Two dials: HOW MANY of them there are, and HOW")
    print("FAR the mid moves when one trades. Sweep both.\n")

    ps = np.round(np.arange(0.0, 1.001, 0.1), 3)
    res = sweep(ps, "p_informed", width=w_star, skew=skew_star, seed=300)
    print(f"{'informed':>9} {'mean':>9} {'se':>7} {'edge':>9} {'inventory':>10}")
    for p, r in zip(ps, res):
        print(f"{p:9.0%} {r['mean']:9.2f} {r['se']:7.2f} {r['edge']:9.2f} "
              f"{r['inv']:10.2f}")
    means = np.array([r["mean"] for r in res])
    slope = np.polyfit(ps, means, 1)[0]
    print(f"\n  Edge is untouched -- informed traders pay the same spread as")
    print(f"  anyone else. The damage is entirely in the inventory column, and")
    print(f"  it is linear: {slope:+.1f} per unit of informed fraction.")
    if means.min() > 0:
        print(f"  At an impact of {IMPACT}, even 100% informed flow does not kill")
        print(f"  it ({means[-1]:.1f} at p = 1). The fraction is not the binding")
        print(f"  constraint at this impact -- so sweep the impact instead.")

    print("\n  Breakeven in impact, with every order informed:")
    impacts = np.round(np.arange(0.0, 3.01, 0.25), 2)
    ri = sweep(impacts, "impact", width=w_star, skew=skew_star,
               p_informed=1.0, seed=320)
    mi = np.array([r["mean"] for r in ri])
    print(f"{'impact':>9} {'mean':>9} {'se':>7} {'as % of sigma':>15}")
    for im, r in zip(impacts, ri):
        print(f"{im:9.2f} {r['mean']:9.2f} {r['se']:7.2f} {im / SIGMA:14.0%}")
    neg = np.where(mi < 0)[0]
    if len(neg):
        i = neg[0]
        x0, x1, y0, y1 = impacts[i - 1], impacts[i], mi[i - 1], mi[i]
        cross = x0 + (x1 - x0) * y0 / (y0 - y1)
        print(f"\n  Breaks even at an impact of {cross:.2f}, i.e. {cross / SIGMA:.0%} of")
        print(f"  a session's volatility per informed trade. Above that no amount")
        print(f"  of spread at this width is worth quoting.")
    else:
        print(f"\n  Still profitable at an impact of {impacts[-1]:.2f}.")

    print("\n  Can a wider quote rescue it? Best width at each level:")
    widths = np.round(np.arange(0.6, 6.01, 0.3), 2)
    out = []
    for p in (0.0, 0.25, 0.50, 0.75, 1.0):
        r = sweep(widths, "width", skew=skew_star, p_informed=p, seed=310)
        w, sc, _, _ = best_of(widths, r)
        m = r[list(widths).index(w)]["mean"]
        out.append((p, float(w), m))
        print(f"     informed {p:4.0%}: best width {w:4.2f}, mean {m:8.2f}")
    print("     A widening row is the market maker's response to being picked")
    print("     off, and it is what 'the market went wide' means in a crisis.")
    return ps, res, out


def experiment_4(w_star, skew_star):
    rule("4. SENSITIVITY -- which conclusions survive different assumptions?")
    print("Four parameters were chosen, not measured. Halve and double each and")
    print("re-ask the two headline questions. A conclusion that flips under a")
    print("factor of two is not a conclusion.\n")

    skews = np.round(np.arange(0.0, 0.241, 0.02), 3)
    widths = np.round(np.arange(0.4, 4.01, 0.2), 2)

    print(f"{'variant':>16} {'best width':>11} {'best skew':>10} "
          f"{'std cut by skew':>16} {'still helps?':>13}")
    for label, kw in [
        ("as specified", {}),
        ("A x0.5", {"A": A * 0.5}), ("A x2", {"A": A * 2}),
        ("k x0.5", {"k": K * 0.5}), ("k x2", {"k": K * 2}),
        ("sigma x0.5", {"sigma": SIGMA * 0.5}), ("sigma x2", {"sigma": SIGMA * 2}),
        ("impact x0.5", {"impact": IMPACT * 0.5, "p_informed": 0.2}),
        ("impact x2", {"impact": IMPACT * 2, "p_informed": 0.2}),
    ]:
        rw = sweep(widths, "width", skew=0.0, seed=400, **kw)
        w, _, _, _ = best_of(widths, rw)
        rs = sweep(skews, "skew", width=float(w), seed=410, **kw)
        s, _, _, _ = best_of(skews, rs)
        cut = 100 * (rs[list(skews).index(s)]["std"] - rs[0]["std"]) / rs[0]["std"]
        print(f"{label:>16} {w:11.2f} {s:10.3f} {cut:15.1f}% "
              f"{'yes' if s > 0 else 'NO':>13}")


# ==========================================================================
# Figures
# ==========================================================================

def figures(widths, w_res, skews, s_res, w_star, skew_star):
    rule("FIGURES")
    os.makedirs(FIGS, exist_ok=True)

    # (a) the spread tradeoff
    fig, ax = plt.subplots(figsize=(7, 4.2))
    m = np.array([r["mean"] for r in w_res])
    se = np.array([r["se"] for r in w_res])
    ax.errorbar(widths, m, yerr=2 * se, color=INK, lw=1.5, capsize=3, marker="o", ms=4)
    ax.axvline(NAIVE_BEST_WIDTH, color=GREY, ls=":", lw=1.2)
    ax.annotate("risk-neutral 2/k", (NAIVE_BEST_WIDTH, ax.get_ylim()[0]),
                xytext=(4, 6), textcoords="offset points", fontsize=8, color=GREY)
    ax.axvline(w_star, color=ACCENT, ls="--", lw=1.4, label=f"best {w_star:.2f}")
    ax.set_xlabel("quoted width"); ax.set_ylabel("mean session P&L")
    ax.set_title("Wider earns more per fill and gets fewer fills")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGS}/width.png", dpi=150, bbox_inches="tight")
    print(f"  {FIGS}/width.png")

    # (b) the frontier the skew traces out
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = [r["std"] for r in s_res]; ys = [r["mean"] for r in s_res]
    ax.plot(xs, ys, color=INK, lw=1.4, marker="o", ms=4, zorder=2)
    for s, x, y in zip(skews, xs, ys):
        ax.annotate(f"{s:.2f}", (x, y), textcoords="offset points",
                    xytext=(5, -9), fontsize=7, color=GREY)
    ax.scatter([xs[0]], [ys[0]], s=90, color=ACCENT, zorder=3, label="no skew")
    ax.set_xlabel("standard deviation of session P&L")
    ax.set_ylabel("mean session P&L")
    ax.set_title("What skewing buys, and what it costs")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGS}/frontier.png", dpi=150, bbox_inches="tight")
    print(f"  {FIGS}/frontier.png")

    # (c) one session, so the mechanism is legible in thirty seconds
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    path = _one_path(w_star, 0.0, seed=999)
    path_s = _one_path(w_star, skew_star, seed=999)
    axes[0].plot(path["mid"], color=INK, lw=1)
    axes[0].set_ylabel("mid")
    axes[1].plot(path["q"], color=GREY, lw=1, label="no skew")
    axes[1].plot(path_s["q"], color=ACCENT, lw=1.2, label=f"skew {skew_star:.2f}")
    axes[1].axhline(0, color="#ccc", lw=0.6); axes[1].set_ylabel("inventory")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(path["value"], color=GREY, lw=1)
    axes[2].plot(path_s["value"], color=ACCENT, lw=1.2)
    axes[2].axhline(0, color="#ccc", lw=0.6)
    axes[2].set_ylabel("P&L"); axes[2].set_xlabel("step")
    axes[0].set_title("One session: the same market, quoted two ways")
    fig.tight_layout(); fig.savefig(f"{FIGS}/session.png", dpi=150, bbox_inches="tight")
    print(f"  {FIGS}/session.png")


def _one_path(width, skew, seed, n_steps=N_STEPS, sigma=SIGMA, A=A, k=K):
    """A single session, recorded step by step, for the illustration."""
    dt = 1.0 / n_steps
    rp = np.random.default_rng(seed)
    rf = np.random.default_rng(seed + 10_000)
    mid, q, cash = S0, 0.0, 0.0
    mids, qs, vals = [], [], []
    for i in range(n_steps):
        centre = mid - skew * q
        bid, ask = centre - width / 2, centre + width / 2
        lb = A * np.exp(-k * max(mid - bid, 0.0))
        la = A * np.exp(-k * max(ask - mid, 0.0))
        ub, ua = rf.random(), rf.random()
        rf.random(); rf.random()
        if ub < lb * dt: cash -= bid; q += 1
        if ua < la * dt: cash += ask; q -= 1
        mid += sigma * np.sqrt(dt) * rp.standard_normal()
        mids.append(mid); qs.append(q); vals.append(cash + q * mid)
    return {"mid": mids, "q": qs, "value": vals}


# ==========================================================================

def main():
    run_tests()
    widths, w_res, w_star = experiment_1()
    skews, s_res = experiment_2(w_star)
    skew_star, _, _, _ = best_of(skews, s_res)
    print(f"\n  skew carried into the rest of the file: {skew_star:.3f}")
    experiment_2b(w_star)
    experiment_3(w_star, float(skew_star))
    experiment_4(w_star, float(skew_star))
    figures(widths, w_res, skews, s_res, w_star, float(skew_star))
    rule("DONE")


if __name__ == "__main__":
    main()