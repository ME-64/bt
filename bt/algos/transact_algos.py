"""
A collection of algos for buying and selling securities or strategies
"""

from __future__ import division

import abc
import random
import re

import numpy as np
import pandas as pd
import sklearn.covariance
from future.utils import iteritems

import bt
from bt.core import Algo, AlgoStack, SecurityBase, is_zero

class Rebalance(Algo):# {{{

    """
    Rebalances capital based on temp['weights']

    Rebalances capital based on temp['weights']. Also closes
    positions if open but not in target_weights. This is typically
    the last Algo called once the target weights have been set.

    Requires:
        * weights
        * cash (optional): You can set a 'cash' value on temp. This should be a
          number between 0-1 and determines the amount of cash to set aside.
          For example, if cash=0.3, the strategy will allocate 70% of its
          value to the provided weights, and the remaining 30% will be kept
          in cash. If this value is not provided (default), the full value
          of the strategy is allocated to securities.
        * notional_value (optional): Required only for fixed_income targets. This is the base
          balue of total notional that will apply to the weights.
    """

    def __init__(self):
        super(Rebalance, self).__init__()

    def __call__(self, target):
        if "weights" not in target.temp:
            return True

        targets = target.temp["weights"]

        # save value because it will change after each call to allocate
        # use it as base in rebalance calls
        # call it before de-allocation so that notional_value is correct
        if target.fixed_income:
            if "notional_value" in target.temp:
                base = target.temp["notional_value"]
            else:
                base = target.notional_value
        else:
            base = target.value

        # de-allocate children that are not in targets and have non-zero value
        # (open positions)
        for cname in target.children:
            # if this child is in our targets, we don't want to close it out
            if cname in targets:
                continue

            # get child and value
            c = target.children[cname]
            if target.fixed_income:
                v = c.notional_value
            else:
                v = c.value
            # if non-zero and non-null, we need to close it out
            if v != 0.0 and not np.isnan(v):
                target.close(cname, update=False)

        # If cash is set (it should be a value between 0-1 representing the
        # proportion of cash to keep), calculate the new 'base'
        if "cash" in target.temp and not target.fixed_income:
            base = base * (1 - target.temp["cash"])

        # Turn off updating while we rebalance each child
        for item in iteritems(targets):
            target.rebalance(item[1], child=item[0], base=base, update=False)

        # Now update
        target.root.update(target.now)

        return True# }}}

class RebalanceOverTime(Algo):# {{{

    """
    Similar to Rebalance but rebalances to target
    weight over n periods.

    Rebalances towards a target weight over a n periods. Splits up the weight
    delta over n periods.

    This can be useful if we want to make more conservative rebalacing
    assumptions. Some strategies can produce large swings in allocations. It
    might not be reasonable to assume that this rebalancing can occur at the
    end of one specific period. Therefore, this algo can be used to simulate
    rebalancing over n periods.

    This has typically been used in monthly strategies where we want to spread
    out the rebalancing over 5 or 10 days.

    Note:
        This Algo will require the run_always wrapper in the above case. For
        example, the RunMonthly will return True on the first day, and
        RebalanceOverTime will be 'armed'. However, RunMonthly will return
        False the rest days of the month. Therefore, we must specify that we
        want to always run this algo.

    Args:
        * n (int): number of periods over which rebalancing takes place.

    Requires:
        * weights

    """

    def __init__(self, n=10):
        super(RebalanceOverTime, self).__init__()
        self.n = float(n)
        self._rb = Rebalance()
        self._weights = None
        self._days_left = None

    def __call__(self, target):
        # new weights specified - update rebalance data
        if "weights" in target.temp:
            self._weights = target.temp["weights"]
            self._days_left = self.n

        # if _weights are not None, we have some work to do
        if self._weights:
            tgt = {}
            # scale delta relative to # of periods left and set that as the new
            # target
            for t in self._weights:
                curr = target.children[t].weight if t in target.children else 0.0
                dlt = (self._weights[t] - curr) / self._days_left
                tgt[t] = curr + dlt

            # mock weights and call real Rebalance
            target.temp["weights"] = tgt
            self._rb(target)

            # dec _days_left. If 0, set to None & set _weights to None
            self._days_left -= 1

            if self._days_left == 0:
                self._days_left = None
                self._weights = None

        return True# }}}

class PTE_Rebalance(Algo):# {{{
    """
    Triggers a rebalance when PTE from static weights is past a level.

    Args:
        * PTE_volatility_cap: annualized volatility to target
        * target_weights: dataframe of weights that needs to have the same index as the price dataframe
        * lookback (DateOffset): lookback period for estimating volatility
        * lag (DateOffset): amount of time to wait to calculate the covariance
        * covar_method: method of calculating volatility
        * annualization_factor: number of periods to annualize by.
          It is assumed that target volatility is already annualized by this factor.

    """

    def __init__(
        self,
        PTE_volatility_cap,
        target_weights,
        lookback=pd.DateOffset(months=3),
        lag=pd.DateOffset(days=0),
        covar_method="standard",
        annualization_factor=252,
    ):

        super(PTE_Rebalance, self).__init__()
        self.PTE_volatility_cap = PTE_volatility_cap
        self.target_weights = target_weights
        self.lookback = lookback
        self.lag = lag
        self.covar_method = covar_method
        self.annualization_factor = annualization_factor

    def __call__(self, target):

        if target.now is None:
            return False

        if target.positions.shape == (0, 0):
            return True

        positions = target.positions.loc[target.now]
        if positions is None:
            return True
        prices = target.universe.loc[target.now, positions.index]
        if prices is None:
            return True

        current_weights = positions * prices / target.value

        target_weights = self.target_weights.loc[target.now, :]

        cols = list(current_weights.index.copy())
        for c in target_weights.keys():
            if c not in cols:
                cols.append(c)

        weights = pd.Series(np.zeros(len(cols)), index=cols)
        for c in cols:
            if c in current_weights:
                weights[c] = current_weights[c]
            if c in target_weights:
                weights[c] -= target_weights[c]

        t0 = target.now - self.lag
        prc = target.universe.loc[t0 - self.lookback : t0, cols]
        returns = bt.ffn.to_returns(prc)
        returns = prc / prc.shift(1) - 1

        # calc covariance matrix
        if self.covar_method == "ledoit-wolf":
            covar = sklearn.covariance.ledoit_wolf(returns)
        elif self.covar_method == "standard":
            covar = returns.cov()
        else:
            raise NotImplementedError("covar_method not implemented")

        PTE_vol = np.sqrt(
            np.matmul(weights.values.T, np.matmul(covar.values, weights.values))
            * self.annualization_factor
        )

        if pd.isnull(PTE_vol):
            return False
        # vol is too high
        if PTE_vol > self.PTE_volatility_cap:
            return True
        else:
            return False

        return True# }}}

class ReplayTransactions(Algo):# {{{

    """
    Replay a list of transactions that were executed.
    This is useful for taking a blotter of actual trades that occurred,
    and measuring performance against hypothetical strategies.
    In particular, one can replay the outputs of backtest.Result.get_transactions

    Note that this allows the timestamps and prices of the reported transactions
    to be completely arbitrary, so while the strategy may track performance
    on a daily basis, it will accurately account for the actual PNL of
    the trades based on where they actually traded, and the bidofferpaid
    attribute on the strategy will capture the "slippage" as measured
    against the daily prices.

    Args:
        * transactions (str): name of a MultiIndex dataframe with format
          Date, Security | quantity, price.
          Note this schema follows the output of backtest.Result.get_transactions

    """

    def __init__(self, transactions):
        super(ReplayTransactions, self).__init__()
        self.transactions = transactions

    def __call__(self, target):
        timeline = target.data.index
        index = timeline.get_loc(target.now)
        end = target.now
        if index == 0:
            start = pd.Timestamp.min
        else:
            start = timeline[index - 1]
        # Get the transactions since the last update
        all_transactions = target.get_data(self.transactions)
        timestamps = all_transactions.index.get_level_values("Date")
        transactions = all_transactions[(timestamps > start) & (timestamps <= end)]
        for (_, security), transaction in transactions.iterrows():
            c = target[security]
            c.transact(
                transaction["quantity"], price=transaction["price"], update=False
            )

        # Now update
        target.root.update(target.now)

        return True# }}}

class ClosePositionsAfterDates(Algo):# {{{

    """
    Close positions on securities after a given date.
    This can be used to make sure positions on matured/redeemed securities are
    closed. It can also be used as part of a strategy to, i.e. make sure
    the strategy doesn't hold any securities with time to maturity less than a year

    Note that if placed after a RunPeriod algo in the stack, that the actual
    closing of positions will occur after the provided date. For this to work,
    the "price" of the security (even if matured) must exist up until that date.
    Alternatively, run this with the @run_always decorator to close the positions
    immediately.

    Also note that this algo does not operate using temp['weights'] and Rebalance.
    This is so that hedges (which are excluded from that workflow) will also be
    closed as necessary.

    Args:
        * close_dates (str): the name of a dataframe indexed by security name, with columns
          "date": the date after which we want to close the position ASAP

    Sets:
        * target.perm['closed'] : to keep track of which securities have already closed
    """

    def __init__(self, close_dates):
        super(ClosePositionsAfterDates, self).__init__()
        self.close_dates = close_dates

    def __call__(self, target):
        if "closed" not in target.perm:
            target.perm["closed"] = set()
        close_dates = target.get_data(self.close_dates)["date"]
        # Find securities that are candidate for closing
        sec_names = [
            sec_name
            for sec_name, sec in iteritems(target.children)
            if isinstance(sec, SecurityBase)
            and sec_name in close_dates.index
            and sec_name not in target.perm["closed"]
        ]

        # Check whether closed
        is_closed = close_dates.loc[sec_names] <= target.now

        # Close position
        for sec_name in is_closed[is_closed].index:
            target.close(sec_name, update=False)
            target.perm["closed"].add(sec_name)

        # Now update
        target.root.update(target.now)

        return True# }}}

class RollPositionsAfterDates(Algo):# {{{

    """
    Roll securities based on the provided map.
    This can be used for any securities which have "On-The-Run" and "Off-The-Run"
    versions (treasury bonds, index swaps, etc).

    Also note that this algo does not operate using temp['weights'] and Rebalance.
    This is so that hedges (which are excluded from that workflow) will also be
    rolled as necessary.

    Args:
        * roll_data (str): the name of a dataframe indexed by security name, with columns
            - "date": the first date at which the roll can occur
            - "target": the security name we are rolling into
            - "factor": the conversion factor. One unit of the original security
              rolls into "factor" units of the new one.

    Sets:
        * target.perm['rolled'] : to keep track of which securities have already rolled
    """

    def __init__(self, roll_data):
        super(RollPositionsAfterDates, self).__init__()
        self.roll_data = roll_data

    def __call__(self, target):
        if "rolled" not in target.perm:
            target.perm["rolled"] = set()
        roll_data = target.get_data(self.roll_data)
        transactions = {}
        # Find securities that are candidate for roll
        sec_names = [
            sec_name
            for sec_name, sec in iteritems(target.children)
            if isinstance(sec, SecurityBase)
            and sec_name in roll_data.index
            and sec_name not in target.perm["rolled"]
        ]

        # Calculate new transaction and close old position
        for sec_name, sec_fields in roll_data.loc[sec_names].iterrows():
            if sec_fields["date"] <= target.now:
                target.perm["rolled"].add(sec_name)
                new_quantity = sec_fields["factor"] * target[sec_name].position
                new_sec = sec_fields["target"]
                if new_sec in transactions:
                    transactions[new_sec] += new_quantity
                else:
                    transactions[new_sec] = new_quantity
                target.close(sec_name, update=False)

        # Do all the new transactions at the end, to do any necessary aggregations first
        for new_sec, quantity in iteritems(transactions):
            target.transact(quantity, new_sec, update=False)

        # Now update
        target.root.update(target.now)

        return True# }}}

class SimulateRFQTransactions(Algo):# {{{
    """
    An algo that simulates the outcomes from RFQs (Request for Quote)
    using a "model" that determines which ones becomes transactions and at what price
    those transactions happen. This can be used from the perspective of the sender of the
    RFQ or the receiver.

    Args:
        * rfqs (str): name of a dataframe with columns
          Date, Security | quantity, *additional columns as required by model
        * model (object): a function/callable object with arguments
                - rfqs : data frame of rfqs to respond to
                - target : the strategy object, for access to position and value data
          and which returns a set of transactions, a MultiIndex DataFrame with:
                Date, Security | quantity, price
    """

    def __init__(self, rfqs, model):
        super(SimulateRFQTransactions, self).__init__()
        self.rfqs = rfqs
        self.model = model

    def __call__(self, target):
        timeline = target.data.index
        index = timeline.get_loc(target.now)
        end = target.now
        if index == 0:
            start = pd.Timestamp.min
        else:
            start = timeline[index - 1]
        # Get the RFQs since the last update
        all_rfqs = target.get_data(self.rfqs)
        timestamps = all_rfqs.index.get_level_values("Date")
        rfqs = all_rfqs[(timestamps > start) & (timestamps <= end)]

        # Turn the RFQs into transactions
        transactions = self.model(rfqs, target)

        for (_, security), transaction in transactions.iterrows():
            c = target[security]
            c.transact(
                transaction["quantity"], price=transaction["price"], update=False
            )

        # Now update
        target.root.update(target.now)

        return True# }}}

class HedgeRisks(Algo):# {{{
    """
    Hedges risk measures with selected instruments.

    Make sure that the UpdateRisk algo has been called beforehand.

    Args:
        * measures (list): the names of the risk measures to hedge
        * pseudo (bool): whether to use the pseudo-inverse to compute
          the inverse Jacobian. If False, will fail if the number
          of selected instruments is not equal to the number of
          measures, or if the Jacobian is singular
        * strategy (StrategyBase): If provided, will hedge the risk
          from this strategy in addition to the risk from target.
          This is to allow separate tracking of hedged and unhedged
          performance. Note that risk_strategy must occur earlier than
          'target' in a depth-first traversal of the children of the root,
          otherwise hedging will occur before positions of risk_strategy are
          updated.
        * throw_nan (bool): Whether to throw on nan hedge notionals, rather
          than simply not hedging.

    Requires:
        * selected
    """

    def __init__(self, measures, pseudo=False, strategy=None, throw_nan=True):
        super(HedgeRisks, self).__init__()
        if len(measures) == 0:
            raise ValueError("Must pass in at least one measure to hedge")
        self.measures = measures
        self.pseudo = pseudo
        self.strategy = strategy
        self.throw_nan = throw_nan

    def _get_target_risk(self, target, measure):
        if not hasattr(target, "risk"):
            raise ValueError("risk not set up on target %s" % target.name)
        if measure not in target.risk:
            raise ValueError("measure %s not set on target %s" % (measure, target.name))
        return target.risk[measure]

    def __call__(self, target):
        securities = target.temp["selected"]

        # Get target risk
        target_risk = np.array(
            [self._get_target_risk(target, m) for m in self.measures]
        )
        if self.strategy is not None:
            # Add the target risk of the strategy to the risk of the target
            # (which contains existing hedges)
            target_risk += np.array(
                [self._get_target_risk(self.strategy, m) for m in self.measures]
            )
        # Turn target_risk into a column array
        target_risk = target_risk.reshape(len(self.measures), 1)

        # Get hedge risk as a Jacobian matrix
        data = []
        for m in self.measures:
            d = target.get_data("unit_risk").get(m)
            if d is None:
                raise ValueError(
                    "unit_risk for %s not present in temp on %s"
                    % (self.measure, target.name)
                )
            i = d.index.get_loc(target.now)
            data.append((i, d))

        hedge_risk = np.array(
            [[_get_unit_risk(s, d, i) for (i, d) in data] for s in securities]
        )

        # Get hedge ratios
        if self.pseudo:
            inv = np.linalg.pinv(hedge_risk).T
        else:
            inv = np.linalg.inv(hedge_risk).T
        notionals = np.matmul(inv, -target_risk).flatten()

        # Hedge
        for notional, security in zip(notionals, securities):
            if np.isnan(notional) and self.throw_nan:
                raise ValueError("%s has nan hedge notional" % security)
            target.transact(notional, security)
        return True# }}}

class SetNotional(Algo):# {{{

    """
    Sets the notional_value to use as the base for rebalancing for
    :class:`FixedIncomeStrategy <bt.core.FixedIncomeStrategy>` targets

    Args:
        * notional_value (str): Name of a pd.Series object containing the
          target notional values of the strategy over time.

    Sets:
        * notional_value
    """

    def __init__(self, notional_value):
        self.notional_value = notional_value
        super(SetNotional, self).__init__()

    def __call__(self, target):
        notional_value = target.get_data(self.notional_value)

        if target.now in notional_value.index:
            target.temp["notional_value"] = notional_value.loc[target.now]

            return True
        else:
            return False# }}}

class CloseDead(Algo):# {{{

    """
    Closes all positions for which prices are equal to zero (we assume
    that these stocks are dead) and removes them from temp['weights'] if
    they enter it by any chance.
    To be called before Rebalance().

    In a normal workflow it is not needed, as those securities will not
    be selected by SelectAll(include_no_data=False) or similar method, and
    Rebalance() closes positions that are not in temp['weights'] anyway.
    However in case when for some reasons include_no_data=False could not
    be used or some modified weighting method is used, CloseDead() will
    allow to avoid errors.

    Requires:
        * weights

    """

    def __init__(self):
        super(CloseDead, self).__init__()

    def __call__(self, target):
        if "weights" not in target.temp:
            return True

        targets = target.temp["weights"]
        for c in target.children:
            if target.universe[c].loc[target.now] <= 0:
                target.close(c)
                if c in targets:
                    del targets[c]

        return True# }}}

class CapitalFlow(Algo):# {{{

    """
    Used to model capital flows. Flows can either be inflows or outflows.

    This Algo can be used to model capital flows. For example, a pension
    fund might have inflows every month or year due to contributions. This
    Algo will affect the capital of the target node without affecting returns
    for the node.

    Since this is modeled as an adjustment, the capital will remain in the
    strategy until a re-allocation/rebalancement is made.

    Args:
        * amount (float): Amount of adjustment

    """

    def __init__(self, amount):
        """
        CapitalFlow constructor.

        Args:
            * amount (float): Amount to adjust by
        """
        super(CapitalFlow, self).__init__()
        self.amount = float(amount)

    def __call__(self, target):
        target.adjust(self.amount)
        return True# }}}
