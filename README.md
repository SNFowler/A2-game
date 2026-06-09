# a2-game

A small, dependency-free (pure-Python) CFR solver for the classic
half-street "AKQ" poker toy game and its `A..2` (13-rank) generalisation.

```
solver.py          # the CFR engine (HalfStreetGame) -- solves any deck
run.py             # formatted report: AKQ, half-pot AKQ, A..2
deviation_cost.py  # do equilibrium deviations cost EV? AKQ vs A-J vs A-2
test_solver.py     # self-checks vs the closed-form solution
VERIFICATION.md    # literature cross-check (Chen & Ankenman, GTO Wizard)
```

## The game

Two players each get one card from a deck of `N` distinct ranks, dealt
without replacement (higher rank wins at showdown):

1. **OOP** (out of position) is forced to **check** — the "half street".
2. **IP** (in position) may **bet** `bet` into a pot of `pot`, or **check** behind.
   - IP checks → showdown for the pot.
   - IP bets → OOP may **call** or **fold**.

"OOP always checks" is exactly this structure: OOP never leads, but it
still has a call/fold decision when facing a bet (otherwise IP would bet
every hand and there would be no game).

Payoffs are zero-sum, reported as net chip swing to IP. The pot is assumed
already in the middle, anted equally (`pot/2` each), so:

| terminal            | payoff to IP        |
|---------------------|---------------------|
| both check          | ±`pot/2`            |
| bet, fold           | +`pot/2`           |
| bet, call           | ±(`pot/2` + `bet`) |

## Usage

```bash
python run.py            # solve AKQ + half-pot AKQ + the A..2 deck
python test_solver.py    # self-checks vs the closed-form solution
```

```python
from solver import HalfStreetGame

g = HalfStreetGame("AKQ", pot=1.0, bet=1.0)   # labels high->low
ip, oop = g.train(60000)                      # average strategies
print(ip)    # {'A': P(bet), 'K': ..., 'Q': ...}
print(oop)   # {'A': P(call|bet), ...}
print(g.game_value(), g.exploitability())     # ~0 exploitability = equilibrium
```

The solver uses plain CFR (regret matching) with a small exploration
"tremble" on IP's bet frequency so that off-path information sets (e.g.
"OOP holds the nuts facing a bet" when IP never value-bets into it) stay
trained — otherwise the degenerate pot-sized AKQ game leaves them frozen
and exploitable. `exploitability()` best-responds to the average
strategies and returns ≈0 at equilibrium.

### Canonical (threshold) strategies

In games where many of OOP's cards are *identical* bluff-catchers (they
beat exactly the bluffs and lose to exactly the value bets), only the
**total** call frequency is pinned by the equilibrium — how it is split
across those cards is a free choice, so raw CFR returns an arbitrary
"smeared" mixture (e.g. call J 95%, T 80%, 9 52%, …). That is a valid
equilibrium but not the canonical one. `canonical_strategies()` re-packs
the indifferent block **top-down** — call the strongest bluff-catchers at
100%, mix at most one boundary card, fold the rest — preserving the exact
game value while giving the clean threshold form you would actually play
(you never fold a stronger bluff-catcher to call a weaker one). This is
the blocker-/future-street-robust representative.

## Results

### AKQ, bet \$1 into \$1 pot (as requested)

IP value-bets A and checks K/Q; OOP folds K/Q. **Game value = 0** (a
pot-sized bet extracts no value in the 3-card game; IP's EV-maximizing
size is ≈41.4% pot). With a pot-sized bet the game is *degenerate* — a
continuum of equilibria all worth 0 — so the **canonical** solution is
the limit from just below pot: **IP bluffs Q at 50%** (one bluff per two
value bets), **OOP folds K**. Raw CFR returns a different (Q-bluff 0%) but
equally-valued, unexploitable representative; `run.py` prints the
canonical answer explicitly. See [VERIFICATION.md](VERIFICATION.md) for
the literature cross-check.

### AKQ, half-pot bet (\$0.50 into \$1) — validation vs closed form

Uniquely determined; matches theory (`alpha = B/(P+B) = 1/3`):

| card | IP P(bet) | theory | OOP P(call) | theory |
|------|-----------|--------|-------------|--------|
| A    | 100%      | value  | 100%        | call   |
| K    | 0%        | check  | 33%         | 1/3    |
| Q    | 33%       | 1/3    | 0%          | fold   |

Game value to IP ≈ **+0.0278**.

### A..2 game (13 ranks), bet \$1 into \$1 pot

The solver recovers the classic *polar* structure: IP value-bets the very
top, checks the middle, and bluffs the very bottom; OOP defends with a
monotonically decreasing call frequency as its bluff-catcher weakens.

- IP bets **A, K, Q** for value (100%), checks **J…4**, bluffs **2 (100%)
  and 3 (exactly 50%)**.
- OOP defends, in canonical threshold form, by calling **A, K, Q, J, T**
  at 100%, mixing **9 at ~50%**, and folding **8 and below**. (Raw CFR
  returns an equivalent smeared mixture across J…4 worth the same total;
  `canonical_strategies()` re-packs it top-down.)
- Game value to IP ≈ **+0.0545**, exploitability ≈ 8e-5.
