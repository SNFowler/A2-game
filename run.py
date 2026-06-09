"""
Solve and pretty-print the half-street AKQ game and its A..2 generalisation.

Run:  python run.py
"""

from __future__ import annotations

from solver import HalfStreetGame

FULL_DECK = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]


def _fmt_pct(p):
    return f"{100 * p:5.1f}%"


def classify_ip(label, bet_prob, game):
    """Label an IP card's role: value-bet, check, or bluff."""
    if bet_prob < 0.005:
        return "check"
    if bet_prob > 0.995:
        return "bet (pure)"
    return "mix"


def report(game: HalfStreetGame, iterations=40000, title=""):
    ip, oop_raw = game.train(iterations)
    _, oop = game.canonical_strategies()   # clean threshold form for display
    pot, bet = game.pot, game.bet
    alpha = bet / (pot + bet)

    print("=" * 64)
    print(title or f"{''.join(game.labels_high_to_low)} game")
    print(f"pot = {pot:g}, bet = {bet:g}   (alpha = bet/(pot+bet) = {alpha:.4f})")
    print("-" * 64)

    print("IP (in position) -- P(bet) by card; checks otherwise:")
    print("   card   P(bet)   role")
    for lab in game.labels_high_to_low:
        b = ip[lab]
        role = "value" if b > 0.005 and _is_strong(lab, game) else (
            "bluff" if b > 0.005 else "check")
        print(f"    {lab:>2}   {_fmt_pct(b)}   {role}")

    print()
    print("OOP (out of position) -- P(call) by card facing a bet (canonical")
    print("threshold form: call strongest bluff-catchers, one mixed boundary):")
    print("   card   P(call)")
    for lab in game.labels_high_to_low:
        print(f"    {lab:>2}   {_fmt_pct(oop[lab])}")

    print("-" * 64)
    val = game.game_value()
    expl = game.exploitability()
    print(f"Game value to IP : {val:+.5f}  (chips, per hand)")
    print(f"Exploitability   : {expl:.2e}   (~0 confirms equilibrium)")

    # aggregate value:bluff structure for IP
    n = game.N
    value_bets = sum(1 for lab in game.labels_high_to_low
                     if ip[lab] > 0.005 and _is_strong(lab, game))
    bluffs = sum(ip[lab] for lab in game.labels_high_to_low
                 if ip[lab] > 0.005 and not _is_strong(lab, game))
    print(f"IP bets ~{value_bets} value card(s) and ~{bluffs:.2f} bluff "
          f"card-equivalents  (alpha-ratio target bluff/value = {alpha/(1-alpha):.3f})")
    print("=" * 64)
    print()
    return ip, oop, val, expl


def _is_strong(label, game):
    """Heuristic: a card is a 'value' bet if it is in the top half."""
    idx_high_to_low = game.labels_high_to_low.index(label)
    return idx_high_to_low < game.N / 2.0


def main():
    # ---- the classic AKQ game --------------------------------------------
    akq = HalfStreetGame("AKQ", pot=1.0, bet=1.0)
    report(akq, iterations=40000,
           title="AKQ game  (OOP always checks; IP bets $1 into $1 pot or checks)")

    print("WHY 0%? A pot-sized bet makes AKQ DEGENERATE: a whole CONTINUUM of")
    print("profiles (IP Q-bluff anywhere in [0, 1/2], OOP folding K) are ALL")
    print("exact, unexploitable Nash equilibria worth 0 -- each ties against the")
    print("others, so 'unexploitable' cannot pick the right one. Raw CFR lands on")
    print("the b=0 endpoint. The CANONICAL solution needs an explicit selection")
    print("rule -- the limit of the unique sub-pot game (bet -> pot from below),")
    print("which pins IP's bluff at alpha = B/(P+B):")
    ip_c, oop_c, gc = akq.solve_canonical(iterations=800000, delta=0.01)
    print(f"   canonical (solver, bet=0.99 pot): "
          f"IP bluff Q = {ip_c['Q']:.3f} -> 1/2,   OOP call K = {oop_c['K']:.3f} -> 0")
    print("   => textbook AKQ: value-bet A (100%), BLUFF Q at 50%, check K;")
    print("      OOP calls A (100%), FOLDS K.  Game value to IP = 0.")
    print("   (Confirmed by Chen & Ankenman / GTO Wizard -- see VERIFICATION.md)")
    print()

    # ---- AKQ with a HALF-POT bet: unique, non-degenerate equilibrium -----
    akq_half = HalfStreetGame("AKQ", pot=1.0, bet=0.5)
    report(akq_half, iterations=120000,
           title="AKQ game (HALF-pot bet $0.50 into $1) -- validates vs closed form")
    print("Closed form (bet B into pot P, alpha = B/(P+B) = 1/3):")
    print("   IP bluffs Q at  B/(P+B)     = 0.333   (value-bets A, checks K)")
    print("   OOP calls K at  (P-B)/(P+B) = 0.333   (calls A, folds Q)")
    print()

    # ---- the A..2 generalisation -----------------------------------------
    # non-degenerate, so use a tiny exploration floor + long run for sharp
    # convergence (IP bluffs card 3 at exactly 1/2; OOP indifference -> 0).
    full = HalfStreetGame(FULL_DECK, pot=1.0, bet=1.0, explore=1e-4)
    report(full, iterations=150000,
           title="A..2 game (13 ranks)  (OOP always checks; IP bets $1 into $1 or checks)")


if __name__ == "__main__":
    main()
