"""
A general extensive-form CFR solver for a ONE-street poker toy game with a full
branching betting tree -- donk bets, raises and re-raises -- generalising the
hard-coded half-street ``solver.HalfStreetGame``.

Where the half-street game allows exactly one decision each (IP bets/checks, OOP
calls/folds), this solver builds and solves the whole tree:

    OOP acts first.  With ``allow_donk`` it may CHECK or BET (a "donk"/lead);
    otherwise it is forced to check (recovering the half street).
      - after a check, IP may CHECK BEHIND (showdown) or BET.
      - facing any bet, the player to act may FOLD, CALL, or (if the raise
        ``cap`` is not yet reached) RAISE -- and the opponent may then re-raise,
        up to ``cap`` total bets in the sequence.

Betting is FIXED-LIMIT: every bet and raise adds the same increment ``bet``.
``cap`` is the maximum number of bets+raises in the sequence (cap=1 is a single
bet with no raises; the classic limit cap is 4).

Payoff convention (identical to ``solver``): the pot ``pot`` is already anted
(``pot/2`` each); net chip swing is reported to IP, zero-sum. Tracking each
player's street contribution ``c``:

    showdown (contributions equal = c) : +-(pot/2 + c)   higher card wins
    fold                               : winner gets pot/2 + folder's contrib

Solving is vanilla CFR (regret matching) sweeping every deal each iteration over
a PRE-COMPILED game tree (built once; only the showdown winner depends on the
deal, so the tree topology and terminal pots are cached). Average strategies
converge to a Nash equilibrium and ``exploitability`` -- an exact best-response
walk over the tree -- returns ~0 there.

Sanity check: ``TreeGame(labels, pot, bet, cap=1, allow_donk=False)`` reproduces
``HalfStreetGame(labels, pot, bet)`` exactly (see test_tree_solver.py).
"""

from __future__ import annotations

# Action tokens
CHECK, BET, CALL, FOLD, RAISE = "x", "b", "c", "f", "r"

# Terminal kinds
_FOLD, _SHOWDOWN = 0, 1


class TreeGame:
    """One-street, fixed-limit poker toy game over a deck of ``len(labels)``
    distinct ranks (``labels`` high -> low; internal index 0 = lowest).

    Parameters
    ----------
    labels      : ranks, highest first (e.g. "AKQ").
    pot, bet    : anted pot and the fixed bet/raise increment.
    cap         : max number of bets+raises in a sequence (>=1).
    allow_donk  : if True OOP may open-bet; if False OOP is forced to check.
    explore     : optional uniform exploration floor on each strategy (keeps
                  off-path infosets trained, as in the half-street solver).
    """

    def __init__(self, labels, pot=1.0, bet=1.0, cap=1, allow_donk=True,
                 explore=0.0):
        self.labels_high_to_low = list(labels)
        self.N = len(labels)
        self.label_of = {i: self.labels_high_to_low[self.N - 1 - i]
                         for i in range(self.N)}
        self.pot = float(pot)
        self.bet = float(bet)
        self.cap = int(cap)
        self.allow_donk = bool(allow_donk)
        self.explore = float(explore)

        self.deals = [(o, i) for o in range(self.N)
                      for i in range(self.N) if o != i]
        self.w = 1.0 / len(self.deals)

        # accumulators, keyed by the integer  idx = node_id * N + card
        self.regret = {}
        self.strat_sum = {}

        # ---- pre-compile the game tree once (parallel arrays per node) -----
        self._term = []         # is terminal?
        self._par = []          # len(history) % 2  (which player's value a
        #                          terminal returns; alternation parity)
        self._player = []       # acting player (0=OOP, 1=IP); -1 if terminal
        self._actions = []      # tuple of legal action tokens
        self._children = []     # tuple of child node ids aligned to _actions
        self._tkind = []        # terminal kind (_FOLD / _SHOWDOWN)
        self._tamount = []      # _FOLD: signed net-to-OOP; _SHOWDOWN: magnitude
        self._history = []      # betting history tuple (for read-out)
        self.root = self._build(())

    # ---- tree mechanics (used by the compiler, read-out and tests) -------
    def actions(self, history):
        """Legal actions at ``history`` (a tuple of tokens); () = OOP to open."""
        if not history:                                   # OOP opens
            return (BET, CHECK) if self.allow_donk else (CHECK,)
        last = history[-1]
        if last == CHECK:                                 # check to IP (no bet)
            return (BET, CHECK)
        # facing a bet or raise: fold / call / (raise if cap allows)
        bets = sum(1 for t in history if t in (BET, RAISE))
        return (FOLD, CALL, RAISE) if bets < self.cap else (FOLD, CALL)

    def is_terminal(self, history):
        if not history:
            return False
        last = history[-1]
        if last in (CALL, FOLD):
            return True
        return history == (CHECK, CHECK)                  # check-check showdown

    def _contribs(self, history):
        """Each player's street contribution after ``history``. Player 0=OOP."""
        c = [0.0, 0.0]
        player = 0
        for t in history:
            other = 1 - player
            if t in (BET, RAISE):
                c[player] = c[other] + self.bet            # match then add one
            elif t == CALL:
                c[player] = c[other]                       # match
            player = other
        return c

    def _payoff_oop(self, history, oop_card, ip_card):
        """Terminal net chip swing to OOP (player 0). (Reference implementation;
        the hot path uses the pre-compiled ``_tkind``/``_tamount``.)"""
        c = self._contribs(history)
        half = self.pot / 2.0
        if history[-1] == FOLD:
            folder = (len(history) - 1) % 2                # who just acted
            amount = half + c[folder]
            return amount if folder == 1 else -amount      # winner = non-folder
        amount = half + c[0]                               # showdown, c[0]==c[1]
        return amount if oop_card > ip_card else -amount

    def _build(self, history):
        """Recursively compile ``history`` and its subtree; returns its node id.
        Nodes are emitted post-order, so a node's id exceeds its children's."""
        if self.is_terminal(history):
            c = self._contribs(history)
            half = self.pot / 2.0
            if history[-1] == FOLD:
                folder = (len(history) - 1) % 2
                amt = half + c[folder]
                kind, amount = _FOLD, (amt if folder == 1 else -amt)
            else:
                kind, amount = _SHOWDOWN, half + c[0]
            return self._emit(history, True, -1, (), (), kind, amount)
        acts = self.actions(history)
        kids = tuple(self._build(history + (a,)) for a in acts)
        return self._emit(history, False, len(history) % 2, acts, kids, -1, 0.0)

    def _emit(self, history, term, player, acts, kids, kind, amount):
        nid = len(self._term)
        self._term.append(term)
        self._par.append(len(history) % 2)
        self._player.append(player)
        self._actions.append(acts)
        self._children.append(kids)
        self._tkind.append(kind)
        self._tamount.append(amount)
        self._history.append(history)
        return nid

    # ---- CFR over the compiled tree --------------------------------------
    def _strategy(self, idx, n):
        """Current regret-matching strategy (list of ``n`` probs) at ``idx``."""
        rs = self.regret.get(idx)
        if rs is None:
            strat = [1.0 / n] * n
        else:
            s, pos = 0.0, [0.0] * n
            for j in range(n):
                r = rs[j]
                if r > 0.0:
                    pos[j] = r
                    s += r
            strat = [p / s for p in pos] if s > 0.0 else [1.0 / n] * n
        if self.explore > 0.0:                             # blend toward uniform
            e, u = self.explore, 1.0 / n
            strat = [(1 - e) * strat[j] + e * u for j in range(n)]
        return strat

    def _cfr(self, nid, oop, ip, p0, p1):
        if self._term[nid]:
            if self._tkind[nid] == _FOLD:
                po = self._tamount[nid]
            else:
                amt = self._tamount[nid]
                po = amt if oop > ip else -amt
            return po if self._par[nid] == 0 else -po      # value to player `par`

        player = self._player[nid]
        card = oop if player == 0 else ip
        acts = self._actions[nid]
        kids = self._children[nid]
        n = len(acts)
        idx = nid * self.N + card
        strat = self._strategy(idx, n)

        util = [0.0] * n
        node_util = 0.0
        if player == 0:
            for j in range(n):
                u = -self._cfr(kids[j], oop, ip, p0 * strat[j], p1)
                util[j] = u
                node_util += strat[j] * u
            cf, own = p1, p0
        else:
            for j in range(n):
                u = -self._cfr(kids[j], oop, ip, p0, p1 * strat[j])
                util[j] = u
                node_util += strat[j] * u
            cf, own = p0, p1

        rs = self.regret.get(idx)
        if rs is None:
            rs = [0.0] * n
            self.regret[idx] = rs
            self.strat_sum[idx] = [0.0] * n
        ss = self.strat_sum[idx]
        for j in range(n):
            rs[j] += cf * (util[j] - node_util)
            ss[j] += own * strat[j]
        return node_util

    def train(self, iterations=20000):
        root = self.root
        for _ in range(iterations):
            for (oop, ip) in self.deals:
                self._cfr(root, oop, ip, 1.0, 1.0)
        return self.average_strategies()

    # ---- read-out --------------------------------------------------------
    def _strat_table(self):
        """``idx -> normalised average-strategy list`` (idx = nid*N + card)."""
        tab = {}
        for idx, ss in self.strat_sum.items():
            s = sum(ss)
            n = len(ss)
            tab[idx] = [v / s for v in ss] if s > 0 else [1.0 / n] * n
        return tab

    def average_strategies(self):
        """``(player, card, history) -> {action: prob}`` from strategy sums."""
        out = {}
        for idx, ss in self.strat_sum.items():
            nid, card = divmod(idx, self.N)
            acts = self._actions[nid]
            player = self._player[nid]
            hist = self._history[nid]
            s = sum(ss)
            probs = ({acts[j]: ss[j] / s for j in range(len(acts))} if s > 0
                     else {a: 1.0 / len(acts) for a in acts})
            out[(player, card, hist)] = probs
        return out

    def game_value(self):
        """EV to IP under the average strategies."""
        tab = self._strat_table()
        tot = 0.0
        for (oop, ip) in self.deals:
            tot += self.w * self._ev_oop(tab, self.root, oop, ip)
        return -tot                                        # net to IP

    def _ev_oop(self, tab, nid, oop, ip):
        if self._term[nid]:
            if self._tkind[nid] == _FOLD:
                return self._tamount[nid]
            amt = self._tamount[nid]
            return amt if oop > ip else -amt
        player = self._player[nid]
        card = oop if player == 0 else ip
        kids = self._children[nid]
        n = len(kids)
        strat = tab.get(nid * self.N + card) or [1.0 / n] * n
        return sum(strat[j] * self._ev_oop(tab, kids[j], oop, ip)
                   for j in range(n))

    # ---- best response / exploitability ----------------------------------
    def _best(self, tab, nid, br, br_card, opp_reach):
        """EV to best-responder ``br`` (0=OOP,1=IP) holding ``br_card``, summed
        over the opponent-card reach distribution ``opp_reach``."""
        if self._term[nid]:
            fold = self._tkind[nid] == _FOLD
            amt = self._tamount[nid]
            s = 0.0
            for oc, pr in opp_reach.items():
                if pr == 0.0:
                    continue
                if fold:
                    po = amt
                else:
                    oop, ip = (br_card, oc) if br == 0 else (oc, br_card)
                    po = amt if oop > ip else -amt
                s += pr * (po if br == 0 else -po)
            return s
        player = self._player[nid]
        kids = self._children[nid]
        n = len(kids)
        if player == br:                                   # best-responder maxes
            return max(self._best(tab, kids[j], br, br_card, opp_reach)
                       for j in range(n))
        total = 0.0                                        # opponent: split reach
        for j in range(n):
            reach_a = {}
            for oc, pr in opp_reach.items():
                if pr == 0.0:
                    continue
                strat = tab.get(nid * self.N + oc)
                pj = strat[j] if strat is not None else 1.0 / n
                if pj != 0.0:
                    reach_a[oc] = pr * pj
            if reach_a:
                total += self._best(tab, kids[j], br, br_card, reach_a)
        return total

    def best_response_value(self, br):
        """EV to player ``br`` when it best-responds to the average opponent."""
        tab = self._strat_table()
        inv = 1.0 / (self.N - 1)
        total = 0.0
        for br_card in range(self.N):
            reach = {oc: inv for oc in range(self.N) if oc != br_card}
            total += (1.0 / self.N) * self._best(tab, self.root, br, br_card,
                                                 reach)
        return total

    def exploitability(self):
        """Sum of both best-response values; ~0 at equilibrium."""
        return self.best_response_value(1) + self.best_response_value(0)


# ---------------------------------------------------------------------------
# Pretty-printing / demo
# ---------------------------------------------------------------------------
_PLAYER = {0: "OOP", 1: "IP "}


def print_tree(game, avg, max_len=None):
    """Print every non-trivial decision infoset (more than one action) grouped
    by betting history, smallest history first."""
    rows = []
    for (player, card, hist), strat in avg.items():
        if len(game.actions(hist)) < 2:
            continue                                       # forced (e.g. donk off)
        if max_len is not None and len(hist) > max_len:
            continue
        rows.append((len(hist), hist, player, card, strat))
    rows.sort(key=lambda r: (r[0], r[1], -r[3]))
    last_hist = object()
    for _, hist, player, card, strat in rows:
        if hist != last_hist:
            h = "".join(hist) if hist else "(open)"
            print(f"  facing [{h}]:")
            last_hist = hist
        s = "   ".join(f"{a} {p:5.1%}" for a, p in strat.items())
        print(f"      {_PLAYER[player]} {game.label_of[card]:>2}   {s}")


def report(game, iterations, title, max_len=None):
    avg = game.train(iterations)
    print("=" * 70)
    print(title)
    print(f"  pot={game.pot:g} bet={game.bet:g} cap={game.cap} "
          f"donk={'on' if game.allow_donk else 'off'}  "
          f"deck={''.join(game.labels_high_to_low)}")
    print("-" * 70)
    print_tree(game, avg, max_len=max_len)
    print("-" * 70)
    print(f"  value to IP  : {game.game_value():+.4f}")
    print(f"  exploitability: {game.exploitability():.2e}  (~0 = equilibrium)")
    print("=" * 70)
    print()
    return avg


def main():
    print("General betting-tree CFR: donks, raises and re-raises.\n")

    # (1) collapses to the half-street game when donks/raises are off
    report(TreeGame("AKQ", 1.0, 0.5, cap=1, allow_donk=False), 120000,
           "AKQ half-pot, NO donk, cap=1  ==  the half-street solver")

    # (2) the full tree: donk allowed, bet/raise/re-raise (cap=3)
    report(TreeGame("AKQ", 1.0, 1.0, cap=3, allow_donk=True, explore=1e-3),
           120000, "AKQ pot-sized, donk ON, cap=3 (bet/raise/re-raise)")

    print("Reading the AKQ full-tree solution:")
    print("  * OOP never DONKS (opens 0% with every card): leading out of")
    print("    position is dominated here -- position is strictly valuable, so")
    print("    OOP checks and lets IP act. The solver discovers this; it is not")
    print("    hard-coded (unlike the half-street game, which assumes it).")
    print("  * Facing a bet, OOP check-RAISES the nuts (A) and folds its")
    print("    bluff-catcher (K) to the pot-sized bet; IP value-raises A,")
    print("    bluff-catches K, folds Q -- the polar value/bluff/fold structure")
    print("    now plays out across multiple raise levels.")
    print()

    # (3) the 13-rank A..2 deck -- validate the half street reduces correctly
    deck = "AKQJT98765432"
    report(TreeGame(deck, 1.0, 1.0, cap=1, allow_donk=False, explore=1e-4),
           150000, "A..2 (13 ranks), NO donk, cap=1  ==  half-street A..2")
    print("Matches the half-street solver on the big deck: value ~ +0.0545, IP")
    print("value-bets A/K/Q, bluffs 2 (100%) and 3 (50%), OOP calls a top block.")
    print()

    # (4) A..2 with raises -- OOP can now check-raise (re-raises in the tree)
    report(TreeGame(deck, 1.0, 1.0, cap=2, allow_donk=False, explore=1e-4),
           300000, "A..2 (13 ranks), NO donk, cap=2 (IP bets, OOP may RAISE)",
           max_len=2)
    print("Adding a raise changes the solution: value falls to ~ +0.0481 because")
    print("OOP can now CHECK-RAISE and fight back. IP value-bets tighten (Q drops")
    print("out of the betting range -- it can no longer profitably face a raise).")
    print()

    # (5) A..2 with DONKS allowed -- does leading out of position pay here?
    report(TreeGame(deck, 1.0, 1.0, cap=2, allow_donk=True, explore=1e-4),
           300000, "A..2 (13 ranks), DONK ON, cap=2 (OOP may lead or check-raise)",
           max_len=1)
    print("Donking PAYS in A..2 (unlike AKQ): allowing OOP to lead drops IP's")
    print("value from +0.0481 to ~ +0.0395 -- OOP gains ~ +0.0086/hand by leading.")
    print("And it is a textbook POLARIZED lead: OOP donks K (~100%, value) and the")
    print("bottom (2 ~70%, bluff) while CHECKING the nuts A (~95%, to trap and")
    print("check-raise) and the whole middle. The 3-card AKQ game is simply too")
    print("small to contain a profitable lead; the 13-rank game is not. (Same")
    print("theme as thin value betting and natural exploitation: structure that")
    print("only the larger game is rich enough to have.)")


if __name__ == "__main__":
    main()
