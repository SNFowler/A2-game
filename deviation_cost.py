"""
Do equilibrium deviations cost EV?  (AKQ vs A-J vs A-2)

A folklore claim (Chen & Ankenman and others): in *small* games like
rock-paper-scissors and the AKQ game, deviating from the equilibrium does
NOT cost you EV -- provided the opponent doesn't adjust to exploit you. In
richer games (the 13-card A..2 game, and real poker) it DOES.

Why: against a fixed opponent you can only lose EV at an information set
where the opponent's strategy makes you *strictly* prefer one action. The
diagnostic is the **action-EV gap** at each of your cards, measured against
the fixed equilibrium opponent:

    IP  card:  gap = EV(bet)  - EV(check)
    OOP card:  gap = EV(call) - EV(fold)

    gap == 0  ->  indifferent: deviating there is FREE (RPS-like)
    gap != 0  ->  strict: playing the other action COSTS |gap|

Run:  python deviation_cost.py
"""

from __future__ import annotations

from solver import HalfStreetGame


def equilibrium(labels, pot=1.0, bet=1.0):
    """Return (game, ip_betprob, oop_callprob) for the canonical equilibrium."""
    if len(labels) == 3:                       # pot-sized AKQ is degenerate
        base = HalfStreetGame(labels, pot, bet)
        ip, oop, _ = base.solve_canonical(iterations=600000, delta=0.01)
        g = HalfStreetGame(labels, pot, bet)   # payoffs for gap evaluation
        return g, ip, oop
    g = HalfStreetGame(labels, pot, bet, explore=1e-4)
    g.train(200000)
    ip, oop = g.canonical_strategies()
    return g, ip, oop


def action_gaps(g, ip, oop):
    """Per-card action-EV gaps against the fixed equilibrium opponent."""
    N, lab = g.N, g.labels_high_to_low
    pb = {N - 1 - lab.index(c): ip[c] for c in lab}
    pc = {N - 1 - lab.index(c): oop[c] for c in lab}
    ip_gap, oop_gap = {}, {}
    for c in range(N):
        eb = ec = 0.0; n = 0
        for o in range(N):
            if o == c:
                continue
            n += 1
            eb += pc[o] * g._u_bet_call(c, o) + (1 - pc[o]) * g._u_bet_fold
            ec += g._u_check(c, o)
        ip_gap[g.label_of[c]] = (eb - ec) / n
        d = 0.0; n = 0
        for i in range(N):
            if i == c:
                continue
            n += 1
            d += pb[i] * ((-g._u_bet_call(i, c)) - (-g._u_bet_fold))
        oop_gap[g.label_of[c]] = d / n
    return ip_gap, oop_gap


def _fmt(gap, order):
    return "  ".join(
        f"{c}:{'free' if abs(gap[c]) < 5e-3 else f'{gap[c]:+.2f}'}" for c in order)


def report(name, labels):
    g, ip, oop = equilibrium(labels)
    ig, og = action_gaps(g, ip, oop)
    lab = g.labels_high_to_low                 # high -> low
    print("=" * 68)
    print(f"{name} game  (pot = bet = 1)")
    print("  IP  bet-vs-check gap per card  (free = deviating is costless):")
    print("    " + _fmt(ig, lab))
    print("  OOP call-vs-fold gap per card:")
    print("    " + _fmt(og, lab))
    # headline: cost of NOT value-betting your best hand (checking the top card)
    top = lab[0]
    print(f"  --> cost of CHECKING your best hand ({top}) instead of "
          f"value-betting: {abs(ig[top]):.3f}")
    return abs(ig[top])


def main():
    print("Deviation cost = EV you lose by playing the wrong action at a card,")
    print("against a FIXED equilibrium opponent. 'free' (gap 0) = no cost.\n")
    print("(Rock-paper-scissors: every action ties the uniform opponent, so the")
    print(" gap is 0 everywhere -- ALL deviations are free.)\n")
    costs = []
    for name, labels in [("AKQ", "AKQ"), ("A-J", "AKQJ"),
                         ("A-2", "AKQJT98765432")]:
        costs.append((name, report(name, labels)))
    print("=" * 68)
    print("Cost of failing to value-bet your best hand, by game size:")
    for name, c in costs:
        print(f"    {name:>4}:  {c:.3f}")
    print("AKQ ~ 0 (the claim holds: value-betting is FREE because a pot-sized")
    print("bet folds out every bluff-catcher). It becomes strictly costly the")
    print("moment the deck is big enough that the opponent must call with real")
    print("bluff-catchers -- i.e. in A-J already, and strongly in A-2 / real poker.")


if __name__ == "__main__":
    main()
