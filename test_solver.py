"""
Self-contained checks for the half-street AKQ solver.

Run directly:   python test_solver.py
Or via pytest:  pytest test_solver.py
"""

from solver import HalfStreetGame


def test_akq_halfpot_matches_closed_form():
    """Non-degenerate AKQ: reproduces the analytic GTO frequencies.

    For a bet B into pot P, alpha = B/(P+B):
        IP value-bets A, checks K, bluffs Q at  B/(P+B)
        OOP calls A, folds Q, calls K (bluff-catcher) at (P-B)/(P+B)
    """
    g = HalfStreetGame("AKQ", pot=1.0, bet=0.5)
    ip, oop = g.train(120000)
    alpha = g.bet / (g.pot + g.bet)            # 1/3

    assert abs(ip["A"] - 1.0) < 0.01
    assert ip["K"] < 0.01
    assert abs(ip["Q"] - alpha) < 0.02         # bluff freq
    assert abs(oop["A"] - 1.0) < 0.01
    assert abs(oop["K"] - (1 - 2 * alpha)) < 0.02   # (P-B)/(P+B) = 1-2alpha
    assert oop["Q"] < 0.01
    assert g.exploitability() < 5e-3


def test_akq_potsized_is_unexploitable_value_zero():
    """Pot-sized AKQ is degenerate but the returned strategy is a valid,
    unexploitable equilibrium with game value 0."""
    g = HalfStreetGame("AKQ", pot=1.0, bet=1.0)
    g.train(60000)
    assert abs(g.game_value()) < 1e-3
    assert g.exploitability() < 5e-3
    ip, oop = g.average_strategies()
    assert abs(ip["A"] - 1.0) < 0.01           # A always value-bets
    assert abs(oop["A"] - 1.0) < 0.01           # nuts always call (on/off path)


def test_full_deck_structure_and_equilibrium():
    """A..2 (13 ranks), pot-sized: polar betting structure + low exploitability.

    IP should value-bet the very top, check the middle, and bluff the very
    bottom; OOP's call frequency should be monotonically non-increasing as its
    card weakens.
    """
    deck = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    g = HalfStreetGame(deck, pot=1.0, bet=1.0)
    ip, oop = g.train(60000)

    assert ip["A"] > 0.99 and ip["K"] > 0.99    # strong value bets
    assert ip["7"] < 0.05                        # a middle card checks
    assert ip["2"] > 0.5                         # bottom card bluffs
    assert oop["A"] > 0.99                        # nuts always call

    calls = [oop[c] for c in deck]
    assert all(calls[i] >= calls[i + 1] - 1e-2 for i in range(len(calls) - 1))
    assert g.exploitability() < 5e-3
    assert g.game_value() > 0                     # positional/betting edge


def test_pot_sized_akq_canonical_bluff_is_one_half():
    """The pot-sized AKQ game is degenerate (a continuum of unexploitable
    equilibria). The canonical selector must return the balanced solution:
    IP bluffs Q at ~1/2, OOP folds K -- not the b=0 endpoint that raw CFR
    happens to land on."""
    g = HalfStreetGame("AKQ", pot=1.0, bet=1.0)
    ip, oop, _ = g.solve_canonical(iterations=400000, delta=0.01)
    assert abs(ip["A"] - 1.0) < 0.01
    assert ip["K"] < 0.01
    assert abs(ip["Q"] - 0.5) < 0.03          # canonical bluff ~ 1/2
    assert oop["K"] < 0.03                      # bluff-catcher folds vs pot bet


def test_canonical_threshold_form():
    """The canonical OOP strategy is a clean threshold: a run of 100% calls,
    at most one mixed boundary card, then all folds -- and it preserves the
    equilibrium value of the raw (smeared) solution."""
    deck = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    g = HalfStreetGame(deck, pot=1.0, bet=1.0, explore=1e-4)
    g.train(150000)
    _, oop = g.canonical_strategies()
    calls = [oop[c] for c in deck]

    # monotone non-increasing
    assert all(calls[i] >= calls[i + 1] - 1e-9 for i in range(len(calls) - 1))
    # at most one strictly-mixed (interior) card
    interior = [c for c in calls if 1e-3 < c < 1 - 1e-3]
    assert len(interior) <= 1
    # same game value as the raw solution (still an equilibrium)
    raw_val = g.game_value()
    ci = {g.N - 1 - deck.index(deck[i]): calls[i] for i in range(len(deck))}
    pb = g._ip_betprob()
    can_val = sum(
        g.w * (pb[i] * (ci[o] * g._u_bet_call(i, o) + (1 - ci[o]) * g._u_bet_fold)
               + (1 - pb[i]) * g._u_check(i, o))
        for o in range(g.N) for i in range(g.N) if i != o)
    assert abs(raw_val - can_val) < 2e-3


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll checks passed.")


if __name__ == "__main__":
    _run_all()
