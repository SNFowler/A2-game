"""
A generic CFR solver for the half-street "AKQ"-style poker toy game.

Game definition (half street)
-----------------------------
Two players each get one card, dealt without replacement from a deck of
``N`` distinct ranks (higher rank wins at showdown).

    1. OOP (out of position) is forced to CHECK  -- the "half street".
    2. IP (in position) may BET ``bet`` into a pot of ``pot``, or CHECK behind.
         - IP CHECK  -> showdown for the pot.
         - IP BET    -> OOP may CALL or FOLD.
              - FOLD  -> IP takes the pot uncontested.
              - CALL  -> showdown for the (now larger) pot.

OOP "always checks" is exactly this structure: OOP never leads, but it still
gets a call/fold decision when facing a bet (otherwise IP would bet every
hand and there would be no game).

Payoff convention (zero-sum, reported from IP's perspective)
------------------------------------------------------------
The pot ``pot`` is assumed already in the middle, contributed equally
(``pot/2`` each as an ante). Net chip swing to IP:

    both check / bet+fold : +-pot/2   (winner of pot gains its own ante back
                                        plus the opponent's ante)
    bet + call            : +-(pot/2 + bet)

The strategies (bet/bluff/call frequencies) are invariant to this scaling;
only the reported EV scale depends on it.

Solving
-------
Plain external-sampling-free CFR: every iteration sweeps all ``N*(N-1)``
deals as chance events and updates regret/strategy sums for the two kinds
of information set (IP's card; OOP's card when facing a bet). Average
strategies converge to a Nash equilibrium.

``exploitability`` gives a rigorous correctness check by best-responding to
the average strategies.
"""

from __future__ import annotations

# Action indices
BET, CHECK = 0, 1          # IP actions
CALL, FOLD = 0, 1          # OOP actions (facing a bet)


def regret_match(regret):
    """Turn a 2-vector of cumulative regrets into a strategy (positive part,
    normalised; uniform if all non-positive)."""
    r0 = regret[0] if regret[0] > 0 else 0.0
    r1 = regret[1] if regret[1] > 0 else 0.0
    s = r0 + r1
    if s > 0:
        return (r0 / s, r1 / s)
    return (0.5, 0.5)


class HalfStreetGame:
    """Half-street AKQ-style game over a deck of ``len(labels)`` ranks.

    ``labels`` are ordered from HIGHEST to LOWEST (e.g. ``"AKQ"`` or
    ``["A","K","Q","J","T","9",...,"2"]``); internally rank index 0 is the
    LOWEST card so that ``a > b`` means card ``a`` wins the showdown.
    """

    def __init__(self, labels, pot=1.0, bet=1.0, explore=3e-3):
        self.labels_high_to_low = list(labels)
        self.N = len(labels)
        # internal index: 0 = lowest ... N-1 = highest
        self.label_of = {i: self.labels_high_to_low[self.N - 1 - i]
                         for i in range(self.N)}
        self.pot = float(pot)
        self.bet = float(bet)
        # Exploration "tremble": when reading IP's bet frequency for OOP's
        # counterfactual updates we floor it at ``explore``. This keeps every
        # OOP information set trained even when it is OFF-PATH at equilibrium
        # (e.g. "OOP holds the nuts facing a bet" when IP never value-bets into
        # it), curing the otherwise-frozen, exploitable off-path strategies
        # that plague the degenerate pot-sized AKQ game.
        self.explore = float(explore)

        # all ordered, distinct deals (oop_card, ip_card)
        self.deals = [(o, i) for o in range(self.N)
                      for i in range(self.N) if o != i]
        self.w = 1.0 / len(self.deals)            # prob of each deal
        self.w_cond = 1.0 / (self.N - 1)          # P(opp card | your card)

        # regret / strategy accumulators, keyed by the acting player's card
        self.ip_regret = {c: [0.0, 0.0] for c in range(self.N)}
        self.ip_strat = {c: [0.0, 0.0] for c in range(self.N)}
        self.oop_regret = {c: [0.0, 0.0] for c in range(self.N)}
        self.oop_strat = {c: [0.0, 0.0] for c in range(self.N)}

    # ---- payoffs (to IP) -------------------------------------------------
    def _show(self, ip, oop, amount):
        """Signed showdown payoff to IP of magnitude ``amount``."""
        return amount if ip > oop else -amount

    def _u_check(self, ip, oop):
        return self._show(ip, oop, self.pot / 2)

    def _u_bet_call(self, ip, oop):
        return self._show(ip, oop, self.pot / 2 + self.bet)

    @property
    def _u_bet_fold(self):
        return self.pot / 2

    # ---- CFR -------------------------------------------------------------
    def _cfr_iteration(self):
        w = self.w
        for (oop, ip) in self.deals:
            sig_ip = regret_match(self.ip_regret[ip])
            sig_oop = regret_match(self.oop_regret[oop])
            p_bet, p_check = sig_ip
            p_call, p_fold = sig_oop

            u_bet_call = self._u_bet_call(ip, oop)
            u_bet_fold = self._u_bet_fold
            u_bet = p_call * u_bet_call + p_fold * u_bet_fold
            u_check = self._u_check(ip, oop)
            node_util = p_bet * u_bet + p_check * u_check

            # IP regrets: counterfactual reach = chance only (OOP acts later)
            self.ip_regret[ip][BET] += w * (u_bet - node_util)
            self.ip_regret[ip][CHECK] += w * (u_check - node_util)
            self.ip_strat[ip][BET] += w * p_bet
            self.ip_strat[ip][CHECK] += w * p_check

            # OOP node only reached when IP bets: cf reach = chance * p_bet.
            # OOP maximises its own payoff = -IP payoff.
            oop_u_call = -u_bet_call
            oop_u_fold = -u_bet_fold
            oop_node = p_call * oop_u_call + p_fold * oop_u_fold
            # floor IP's bet reach so off-path OOP infosets keep learning
            cf = w * (p_bet if p_bet > self.explore else self.explore)
            self.oop_regret[oop][CALL] += cf * (oop_u_call - oop_node)
            self.oop_regret[oop][FOLD] += cf * (oop_u_fold - oop_node)
            self.oop_strat[oop][CALL] += cf * p_call
            self.oop_strat[oop][FOLD] += cf * p_fold

    def train(self, iterations=20000):
        for _ in range(iterations):
            self._cfr_iteration()
        return self.average_strategies()

    def solve_canonical(self, iterations=800000, delta=0.01):
        """Select the *canonical* (balanced) equilibrium.

        At a knife-edge bet size the game can be DEGENERATE: a whole
        continuum of strategy profiles are all unexploitable Nash equilibria
        (e.g. pot-sized AKQ, where IP's Q-bluff frequency is free over
        ``[0, 1/2]``). Plain CFR -- and an exploitability check -- cannot pick
        among them: every one ties against the others. An explicit selection
        rule is required.

        The standard rule is the **limit of the non-degenerate game**: shrink
        the bet to ``bet*(1-delta)``, where the equilibrium is unique (the
        opponent's bluff-catcher strictly mixes and thereby *pins* the
        bettor's bluff frequency at ``alpha = bet/(pot+bet)``), solve that, and
        take ``delta -> 0``. This returns the balanced solution that makes the
        opponent indifferent -- the textbook answer. For already-unique games
        the perturbation is negligible.

        Returns ``(ip, oop, solved_game)`` in canonical threshold form.
        """
        g = HalfStreetGame(self.labels_high_to_low, self.pot,
                           self.bet * (1 - delta), explore=1e-5)
        g.train(iterations)
        ip, oop = g.canonical_strategies()
        return ip, oop, g

    # ---- read-out --------------------------------------------------------
    @staticmethod
    def _normalise(pair):
        s = pair[0] + pair[1]
        return (pair[0] / s, pair[1] / s) if s > 0 else (0.5, 0.5)

    def average_strategies(self):
        """Return ``(ip, oop)`` dicts mapping card-label -> action prob.

        ``ip[label]``  = P(bet)   for that IP card.
        ``oop[label]`` = P(call)  for that OOP card facing a bet.
        """
        ip = {self.label_of[c]: self._normalise(self.ip_strat[c])[BET]
              for c in range(self.N)}
        oop = {self.label_of[c]: self._normalise(self.oop_strat[c])[CALL]
               for c in range(self.N)}
        return ip, oop

    def _oop_call_indifference(self, oop):
        """EV(call) - EV(fold) for OOP holding internal card ``oop`` against
        IP's average betting strategy. Zero => indifferent bluff-catcher."""
        pb = self._ip_betprob()
        d = 0.0
        for ip in range(self.N):
            if ip == oop:
                continue
            # vs a bettor that beats us, calling costs (lose bet) rather than
            # folding (lose ante); vs a worse bettor, calling wins the pot+bet.
            gain = (self._u_bet_fold - self._u_bet_call(ip, oop))  # OOP's call-minus-fold
            d += self.w_cond * pb[ip] * gain
        return d

    def canonical_strategies(self, tol=1e-3):
        """Return ``(ip, oop)`` in canonical *threshold* form.

        OOP's bluff-catchers that are strictly +EV to call are set to 100%,
        strictly -EV to 0%, and the genuinely indifferent block is re-packed
        TOP-DOWN: the strongest cards call 100%, one boundary card mixes, the
        rest fold -- preserving the total defense (so it stays an exact
        equilibrium) while removing the solver's arbitrary smear. This is the
        canonical, blocker-/future-street-robust representative: you never fold
        a stronger bluff-catcher while calling a weaker one.
        """
        ip, _ = self.average_strategies()
        pc = self._oop_callprob()

        strict_call, strict_fold, block = [], [], []
        for c in range(self.N):
            d = self._oop_call_indifference(c)
            if d > tol:
                strict_call.append(c)
            elif d < -tol:
                strict_fold.append(c)
            else:
                block.append(c)

        oop_call = {c: 1.0 for c in strict_call}
        oop_call.update({c: 0.0 for c in strict_fold})

        # total defense mass to preserve across the indifferent block
        budget = sum(pc[c] for c in block)
        # re-pack top-down: strongest (highest internal index) first
        for c in sorted(block, reverse=True):
            take = min(1.0, max(0.0, budget))
            oop_call[c] = take
            budget -= take

        oop = {self.label_of[c]: oop_call[c] for c in range(self.N)}
        return ip, oop

    # ---- value & exploitability -----------------------------------------
    def _ip_betprob(self):
        return {c: self._normalise(self.ip_strat[c])[BET] for c in range(self.N)}

    def _oop_callprob(self):
        return {c: self._normalise(self.oop_strat[c])[CALL] for c in range(self.N)}

    def game_value(self):
        """EV to IP under the current average strategies."""
        pb = self._ip_betprob()
        pc = self._oop_callprob()
        v = 0.0
        for (oop, ip) in self.deals:
            p_bet = pb[ip]
            p_call = pc[oop]
            u_bet = p_call * self._u_bet_call(ip, oop) + (1 - p_call) * self._u_bet_fold
            u_check = self._u_check(ip, oop)
            v += self.w * (p_bet * u_bet + (1 - p_bet) * u_check)
        return v

    def best_response_value_ip(self):
        """Value to IP when IP best-responds to OOP's average strategy."""
        pc = self._oop_callprob()
        total = 0.0
        for ip in range(self.N):
            ev_bet = ev_check = 0.0
            n = 0
            for oop in range(self.N):
                if oop == ip:
                    continue
                n += 1
                p_call = pc[oop]
                ev_bet += p_call * self._u_bet_call(ip, oop) + (1 - p_call) * self._u_bet_fold
                ev_check += self._u_check(ip, oop)
            total += (1.0 / self.N) * max(ev_bet, ev_check) / n
        return total

    def best_response_value_oop(self):
        """Value to OOP when OOP best-responds to IP's average strategy."""
        pb = self._ip_betprob()
        total = 0.0
        for oop in range(self.N):
            ev_call = ev_fold = forced = 0.0
            n = 0
            for ip in range(self.N):
                if ip == oop:
                    continue
                n += 1
                p_bet = pb[ip]
                # forced part: IP checks -> showdown (OOP gets -IP payoff)
                forced += (1 - p_bet) * (-self._u_check(ip, oop))
                # decision part: IP bets
                ev_call += p_bet * (-self._u_bet_call(ip, oop))
                ev_fold += p_bet * (-self._u_bet_fold)
            total += (1.0 / self.N) * (forced + max(ev_call, ev_fold)) / n
        return total

    def exploitability(self):
        """Sum of both players' best-response gains; ~0 at equilibrium."""
        return self.best_response_value_ip() + self.best_response_value_oop()
