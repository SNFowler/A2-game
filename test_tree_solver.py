"""
Checks for the general betting-tree solver (donks, raises, re-raises).

Run directly:   python test_tree_solver.py
Or via pytest:  pytest test_tree_solver.py
"""

from functools import lru_cache

from solver import HalfStreetGame
from tree_solver import TreeGame, CHECK, BET, CALL, RAISE


@lru_cache(maxsize=None)                         # solve each config once per run
def _solve(labels, pot, bet, cap, donk, explore, iters):
    g = TreeGame(labels, pot, bet, cap=cap, allow_donk=donk, explore=explore)
    avg = g.train(iters)
    return g, avg


def _akq_full():
    return _solve("AKQ", 1.0, 1.0, 3, True, 1e-3, 120000)


def _ip_bet(avg, g):
    """IP P(bet) after OOP's forced check, by card label."""
    return {g.label_of[c]: avg.get((1, c, (CHECK,)), {}).get(BET, 0.0)
            for c in range(g.N)}


def _oop_call(avg, g):
    """OOP P(call) facing IP's bet after a check, by card label."""
    return {g.label_of[c]: avg.get((0, c, (CHECK, BET)), {}).get(CALL, 0.0)
            for c in range(g.N)}


def test_reduces_to_half_street():
    """cap=1 + no donk must reproduce HalfStreetGame's AKQ half-pot solution
    (value-bet A, bluff Q at 1/3, check K; OOP calls A, calls K at 1/3, folds
    Q) and the same game value -- the general solver generalises the special
    one."""
    g = TreeGame("AKQ", 1.0, 0.5, cap=1, allow_donk=False)
    avg = g.train(120000)
    ip, oop = _ip_bet(avg, g), _oop_call(avg, g)

    ref = HalfStreetGame("AKQ", 1.0, 0.5)
    rip, roop = ref.train(120000)

    for c in ("A", "K", "Q"):
        assert abs(ip[c] - rip[c]) < 0.03,  (c, ip[c], rip[c])
        assert abs(oop[c] - roop[c]) < 0.03, (c, oop[c], roop[c])
    assert abs(g.game_value() - ref.game_value()) < 5e-3
    assert g.exploitability() < 5e-3


def test_a2_reduces_to_half_street():
    """The general solver extends to the 13-rank A..2 deck: cap=1 + no donk
    reproduces the half-street A..2 solution -- value ~ +0.0545, IP value-bets
    A/K/Q and bluffs the bottom (2 always, 3 at ~1/2)."""
    deck = "AKQJT98765432"
    g, _ = _solve(deck, 1.0, 1.0, 1, False, 1e-4, 50000)
    avg = g.average_strategies()
    ip = _ip_bet(avg, g)

    ref = HalfStreetGame(deck, 1.0, 1.0)
    ref.train(60000)

    assert abs(g.game_value() - ref.game_value()) < 5e-3
    assert g.exploitability() < 5e-3
    for c in ("A", "K", "Q"):
        assert ip[c] > 0.95                       # value-bet the top
    assert ip["2"] > 0.9                          # always bluff the worst hand
    assert abs(ip["3"] - 0.5) < 0.15              # half-bluff the next-worst
    assert ip["7"] < 0.05                         # a middle card never bets


def test_a2_raises_converge_and_tighten_value_bets():
    """A..2 with a raise allowed (cap=2): OOP may check-raise. The solver still
    converges (low exploitability), the positional edge shrinks (OOP can fight
    back), and IP's value bets tighten -- Q drops out of the betting range."""
    deck = "AKQJT98765432"
    g, _ = _solve(deck, 1.0, 1.0, 2, False, 1e-4, 50000)
    avg = g.average_strategies()
    ip = _ip_bet(avg, g)
    assert g.exploitability() < 5e-3
    assert ip["A"] > 0.95 and ip["K"] > 0.95     # premium value bets remain
    assert ip["Q"] < 0.2                          # thin value drops vs raises
    # value is positive but below the half-street (raises let OOP fight back)
    assert 0.0 < g.game_value() < 0.054


def test_full_tree_is_an_equilibrium():
    """With donk + bet/raise/re-raise the average strategy is still an
    equilibrium: a best-response walk over the whole tree finds ~0 to exploit."""
    g, _ = _akq_full()
    assert g.exploitability() < 1e-2
    # pot-sized AKQ is value-0 (as in the half-street game)
    assert abs(g.game_value()) < 1e-2


def test_oop_does_not_donk_in_symmetric_akq():
    """A genuine GTO finding (not hard-coded): out of position, leading is
    dominated -- OOP opens ~0% with every card and checks to keep position
    value. The solver discovers this from regret minimisation alone."""
    g, avg = _akq_full()
    for c in range(g.N):
        assert avg[(0, c, ())][BET] < 0.05, (g.label_of[c], avg[(0, c, ())])


def test_nuts_value_raise_and_bluffcatcher_folds():
    """Facing a pot-sized bet, the nuts (A) check-raises for value while the
    bluff-catcher (K) folds -- the polar structure now spans a raise level."""
    g, avg = _akq_full()
    a_idx = g.N - 1                              # A = highest internal index
    k_idx = g.N - 2
    facing = (CHECK, BET)
    assert avg[(0, a_idx, facing)][RAISE] > 0.9   # value check-raise the nuts
    assert avg[(0, k_idx, facing)].get(CALL, 0) < 0.1   # fold the bluff-catcher


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll checks passed.")


if __name__ == "__main__":
    _run_all()
