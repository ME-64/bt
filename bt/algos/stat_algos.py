"""
A collection of algos for creating statistics and signals for use elsewhere
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

def _get_unit_risk(security, data, index=None):# {{{
    try:
        unit_risks = data[security]
        unit_risk = unit_risks.values[index]
    except Exception:
        # No risk data, assume zero
        unit_risk = 0.0
    return unit_risk# }}}

class UpdateRisk(Algo):# {{{

    """
    Tracks a risk measure on all nodes of the strategy. To use this node, the
    ``additional_data`` argument on :class:`Backtest <bt.backtest.Backtest>` must
    have a "unit_risk" key. The value should be a dictionary, keyed
    by risk measure, of DataFrames with a column per security that is sensitive to that measure.

    Args:
        * name (str): the name of the risk measure (IR01, PVBP, IsIndustials, etc).
          The name must coincide with the keys of the dictionary passed to additional_data as the
          "unit_risk" argument.
        * history (int): The level of depth in the tree at which to track the time series of risk numbers.
          i.e. 0=no tracking, 1=first level only, etc. More levels is more expensive.

    Modifies:
        * The "risk" attribute on the target and all its children
        * If history==True, the "risks" attribute on the target and all its children

    """

    def __init__(self, measure, history=0):
        super(UpdateRisk, self).__init__(name="UpdateRisk>%s" % measure)
        self.measure = measure
        self.history = history

    def _setup_risk(self, target, set_history):
        """ Setup risk attributes on the node in question """
        target.risk = {}
        if set_history:
            target.risks = pd.DataFrame(index=target.data.index)

    def _setup_measure(self, target, set_history):
        """ Setup a risk measure within the risk attributes on the node in question """
        target.risk[self.measure] = np.NaN
        if set_history:
            target.risks[self.measure] = np.NaN

    def _set_risk_recursive(self, target, depth, unit_risk_frame):
        set_history = depth < self.history
        # General setup of risk on nodes
        if not hasattr(target, "risk"):
            self._setup_risk(target, set_history)
        if self.measure not in target.risk:
            self._setup_measure(target, set_history)

        if isinstance(target, bt.core.SecurityBase):
            # Use target.root.now as non-traded securities may not have been updated yet
            # and there is no need to update them here as we only use position
            index = unit_risk_frame.index.get_loc(target.root.now)
            unit_risk = _get_unit_risk(target.name, unit_risk_frame, index)
            if is_zero(target.position):
                risk = 0.0
            else:
                risk = unit_risk * target.position * target.multiplier
        else:
            risk = 0.0
            for child in target.children.values():
                self._set_risk_recursive(child, depth + 1, unit_risk_frame)
                risk += child.risk[self.measure]

        target.risk[self.measure] = risk
        if depth < self.history:
            target.risks.loc[target.now, self.measure] = risk

    def __call__(self, target):
        unit_risk_frame = target.get_data("unit_risk")[self.measure]
        self._set_risk_recursive(target, 0, unit_risk_frame)
        return True# }}}

class SetStat(Algo):# {{{

    """
    Sets temp['stat'] for use by downstream algos (such as SelectN).

    Args:
        * stat (str|DataFrame): A dataframe of the same dimension as target.universe
          If a string is passed, frame is accessed using target.get_data
          This is the preferred way of using the algo.
    Sets:
        * stat
    """

    def __init__(self, stat):
        if isinstance(stat, pd.DataFrame):
            self.stat_name = None
            self.stat = stat
        else:
            self.stat_name = stat
            self.stat = None

    def __call__(self, target):
        if self.stat_name is None:
            stat = self.stat
        else:
            stat = target.get_data(self.stat_name)
        target.temp["stat"] = stat.loc[target.now]
        return True# }}}

class StatTotalReturn(Algo):# {{{

    """
    Sets temp['total_return'] with total returns over a given period.

    Sets the 'stat' based on the total return of each element in
    temp['selected'] over a given lookback period. The total return
    is determined by ffn's calc_total_return.

    Args:
        * lookback (int): lookback period.
        * lag (int): Lag interval. Total return is calculated in
          the inteval [now - lookback, now - lag]

    Sets:
        * total_return

    Requires:
        * selected

    """

    def __init__(self, lookback=252, lag=1):
        super(StatTotalReturn, self).__init__()
        self.lookback = lookback
        self.lag = lag

    def __call__(self, target):
        selected = (target.temp["selected"]['long'] + target.temp['selected']['short'])
        # t0 = target.now - self.lag

        uni = target.universe
        uni = uni.loc[uni.index <= target.now, list(set(selected))]
        uni = uni.iloc[-self.lookback:-self.lag]

        target.temp['total_return'] = uni.iloc[-1] / uni.iloc[0] - 1
        return True# }}}

class StatSharpeRatio(Algo):# {{{

    """
    Sets temp['sharpe_ratio'] with total returns over a given period.

    Sets the 'stat' based on the total return of each element in
    temp['selected'] over a given lookback period. The total return
    is determined by ffn's calc_total_return.

    Args:
        * lookback (int): lookback period.
        * lag (int): Lag interval. Total return is calculated in
          the inteval [now - lookback, now - lag]

    Sets:
        * sharpe_ratio

    Requires:
        * selected

    """

    def __init__(self, lookback=252, lag=1):
        super(StatTotalReturn, self).__init__()
        self.lookback = lookback
        self.lag = lag

    def __call__(self, target):
        selected = (target.temp["selected"]['long'] + target.temp['selected']['short'])
        # t0 = target.now - self.lag

        uni = target.universe
        uni = uni.loc[uni.index <= target.now, selected]
        uni = uni.iloc[-self.lookback:-self.lag]

        target.temp['sharpe_ratio'] = uni.std()
        return True# }}}

class StatRSI(Algo):# {{{

    """
    Sets temp['rsi'] with the rsi using `n` lookback

    Sets the 'rsi' based on the relative strength indicator for the
    instruments over the `n` lookback period.

    Args:
        * lookback (int): lookback period.
        * lag (int): Lag interval. Total return is calculated in
          the inteval [now - lookback, now - lag]

    Sets:
        * total_return

    Requires:
        * selected

    """

    def __init__(self, lookback=14, lag=1):
        super(StatRSI, self).__init__()
        self.lookback = lookback
        self.lag = lag

    def __call__(self, target):
        selected = (target.temp["selected"]['long'] + target.temp['selected']['short'])

        uni = target.universe
        uni = uni.loc[uni.index <= target.now, list(set(selected))]
        uni = uni.iloc[-self.lookback:-self.lag]

        delta = uni.diff().copy()
        du, dd = delta.copy(), delta.copy()
        du[du < 0] = 0
        dd[dd > 0] = 0
        ru = du.mean()
        rd = dd.mean().abs()
        rs = ru / rd
        rsi = 100 - (100 / (1 + rs))
        target.temp['rsi'] = rsi
        return True# }}}
