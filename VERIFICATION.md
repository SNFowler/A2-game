# Verification against the literature

Cross-checked the solver's output against the standard references for the
half-street AKQ game (Chen & Ankenman, *The Mathematics of Poker*, ch. on
half-street games / "half-street Kuhn poker"; and modern restatements such
as GTO Wizard's toy-game articles and pokerqz's "Theoretical Poker 1").

## What the literature says

The AKQ game is *the* canonical half-street toy game: OOP checks, IP bets
or checks behind, OOP then calls or folds. The published GTO solution:

- IP plays a **polarized** range: value-bets the **A**, bluffs the **Q**,
  checks the **K** (medium hand keeps showdown value).
- IP's **bluff frequency = `alpha = bet/(pot+bet)`**. For a **pot-sized**
  bet that is **50%** — "one bluff for every two value bets" (the bet lays
  the caller 2:1).
- The bluff-catcher (**K**) calls just often enough to make IP indifferent
  to bluffing; as the bet grows it calls less, and **folds entirely to a
  pot-sized bet**.
- IP's EV-maximizing bet size is **≈41.4% of pot (`√2 − 1`)**, not pot.

## Solver vs literature

| Quantity | Literature | Solver | Verdict |
|---|---|---|---|
| Betting structure | value A / bluff Q / check K | same | ✅ |
| Bluff freq, half-pot bet | `alpha = 1/3` | 0.332 | ✅ |
| Call freq (K), half-pot bet | `(P−B)/(P+B) = 1/3` | 0.336 | ✅ |
| Bluff freq, **pot-sized** bet | **50%** | see note | ⚠️ see below |
| Call freq (K), pot-sized bet | **0% (fold)** | 0% | ✅ |
| EV-max bet size | `√2 − 1 ≈ 0.414` pot | 0.41 (independently found) | ✅ |
| Game value, pot-sized bet | 0 | 0.00001 | ✅ |

## The real correction — and a subtlety worth stating precisely

The textbook answer for a pot-sized bet is **IP bluffs Q at 50%**. Raw CFR
returned **Q-bluff 0%**, and `exploitability()` reported ≈0 — which is
*true but misleading*. Here is the precise situation:

- At *exactly* pot-sized the game is **degenerate**: the profiles
  "IP bluffs Q at any `b ∈ [0, 1/2]`, OOP folds K" are **all exact,
  unexploitable Nash equilibria**, every one worth 0. We verified the
  `b = 0` endpoint has exploitability **exactly 0** (both players' best
  responses tie it).
- So this is **not** the rock-paper-scissors "always rock" situation.
  "Always rock" ties the equilibrium mix but *loses to a best responder*,
  i.e. it is exploitable. `b = 0` here is genuinely unexploitable. The
  difference is that RPS has a **unique** equilibrium, whereas pot-sized
  AKQ has a **continuum**.
- **But the user's underlying point stands:** unexploitability /
  "ties against the equilibrium" **cannot select** the canonical
  equilibrium from a continuum — every member ties every other. A solver
  needs an explicit **selection rule**; relying on an exploitability check
  gives false comfort. Our original code had no selection rule, so it
  reported a non-canonical endpoint.

The **canonical** equilibrium is `b = 1/2`, selected by the standard rule:
the **limit of the unique sub-pot game**. For any `bet < pot` the
bluff-catcher (K) *strictly mixes* and thereby **pins** the bluff at
`alpha = bet/(pot+bet)`; CFR converges to it cleanly:

| bet (×pot) | `alpha` | solver Q-bluff | solver K-call |
|---|---|---|---|
| 0.70 | 0.4118 | 0.4118 | 0.176 |
| 0.80 | 0.4444 | 0.4440 | 0.111 |
| 0.90 | 0.4737 | 0.4730 | 0.053 |
| 0.99 | 0.4975 | 0.4961 | 0.005 |
| 1.00 | 0.5000 | **→ 0.500** | **→ 0** |

`HalfStreetGame.solve_canonical()` implements this selection (solve at
`bet*(1-delta)`), and `run.py` now leads with it for the pot-sized case.

## The A..2 (13-rank) generalization

No single published numeric value was located for the exact 13-card
half-street game, but the solver reproduces the **known qualitative
structure** of the multi-card generalization (the discrete analogue of
Chen & Ankenman's `[0,1]` game): IP value-bets a top region, checks the
middle, bluffs a bottom region; OOP defends a top region of
bluff-catchers, tightening monotonically. For a pot-sized bet the solver
gives value **≈ +0.0545** to IP with exploitability ≈ 8e-5 (the bettor
*does* profit here, unlike the 3-card pot-sized case, because medium value
bets now get called by worse hands). The aggregate bluff:value ratio is
1.5 : 3 = 1 : 2, matching `alpha = 1/2`.

## Sources

- The Mathematics of Poker, Chen & Ankenman — half-street AKQ / Kuhn poker.
- GTO Wizard, "How to solve toy games" and "When is bluffing profitable?".
- pokerqz, "Learn Poker Theory 1: optimal value bet sizing in the AKQ game".
