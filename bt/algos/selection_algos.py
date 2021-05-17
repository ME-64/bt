"""
A collection of Algos used to select securities from the universe
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


class SelectNotNegative(Algo):# {{{
    """
    Filters temp['selected'] to exclude securities with negative prices at target.now

    Requires:
        * selected

    Sets:
        * selected
    """


    def __init__(self):
        super(SelectNotNegative, self).__init__()


    def __call__(self, target):
        universe = target.universe.loc[target.now]
        negs = list(universe.loc[universe < 0].index)

        if 'selected' not in target.temp.keys():
            target.temp['selected'] = {'long': [], 'short': []}
            return True
        tmp = {}
        tmp['long'] = [x for x in target.temp['selected'].get('long', []) if x not in negs]
        tmp['short'] = [x for x in target.temp['selected'].get('short', []) if x not in negs]
        target.temp['selected'] = tmp
        return True# }}}

class SelectNotNa(Algo):# {{{
    """
    Filters temp['selected'] to exclude securities with Na prices at target.now

    Requires:
        * selected

    Sets:
        * selected
    """


    def __init__(self):
        super(SelectNotNa, self).__init__()


    def __call__(self, target):
        if 'selected' not in target.temp.keys():
            target.temp['selected'] = {'long': [], 'short': []}
            return True

        universe = target.universe.loc[target.now]
        nas = list(universe.loc[universe.isnull()].index)
        # print(nas)

        tmp = {}
        tmp['long'] = [x for x in target.temp['selected'].get('long', []) if x not in nas]
        tmp['short'] = [x for x in target.temp['selected'].get('short', []) if x not in nas]
        target.temp['selected'] = tmp
        return True# }}}

class SelectAll(Algo):# {{{

    """
    Sets temp['selected'] with all securities as longs (based on universe).

    Can set the securities as longs and/or shorts simulteanously

    Args:
        * long (bool): Whether to select the securities as longs
        * short (bool): Whether to select the securities as shorts

    Sets:
        * selected

    """

    def __init__(self, longs=True, shorts=False):
        super(SelectAll, self).__init__()
        self.longs = longs
        self.shorts = shorts

    def __call__(self, target):
        target.temp['selected'] = {}
        universe = target.universe.loc[target.now].index
        universe = list(universe)

        if self.longs:
            target.temp['selected']['long'] = universe.copy()
        if self.shorts:
            target.temp['selected']['short'] = universe.copy()
        if not self.longs:
            target.temp['selected']['long'] = []
        if not self.shorts:
            target.temp['selected']['short'] = []

        return True# }}}

class SelectHasData(Algo):# {{{

    """
    Sets temp['selected'] based on all items in universe that meet
    data requirements.

    This is a more advanced version of SelectAll. Useful for selecting
    tickers that need a certain amount of data for future algos to run
    properly.

    If there is nothing already in selected (long or short) it will start
    with the whole universe and assign everything to temp['selected']['long']

    For example, if we need the items with 3 months of data or more,
    we could use this Algo with a lookback period of 3 months.

    When providing a lookback period, it is also wise to provide a min_count.
    This is basically the number of data points needed within the lookback
    period for a series to be considered valid. For example, in our 3 month
    lookback above, we might want to specify the min_count as being
    57 -> a typical trading month has give or take 20 trading days. If we
    factor in some holidays, we can use 57 or 58. It's really up to you.

    If you don't specify min_count, min_count will default to ffn's
    get_num_days_required.

    Args:
        * lookback (DateOffset): A DateOffset that determines the lookback
          period.
    Requires:
        * selected
    Sets:
        * selected

    """

    def __init__(self, lookback=30):
        super(SelectHasData, self).__init__()
        self.lookback = lookback

    def __call__(self, target):

        if 'selected' not in target.temp.keys():
            return True

        selected = target.temp['selected']['long'] + target.temp['selected']['short']
        tmp = {}

        univ = target.universe
        univ = univ.loc[univ.index <= target.now]
        univ = univ.iloc[-self.lookback:]
        filt = univ.count()
        cnt = list(filt[filt >= self.lookback].index)
        tmp['long'] = [x for x in target.temp['selected']['long'] if x in cnt]
        tmp['short'] = [x for x in target.temp['selected']['short'] if x in cnt]

        target.temp['selected'] = tmp

        return True# }}}

class SelectThese(Algo):# {{{

    """
    Sets temp['selected'] with a set list of tickers for long and short

    Args:
        * long_tickers (list): List of tickers to select long
        * short_tickers (list): List of tickers to select short

    Requires:
        * selected
    Sets:
        * selected

    """

    def __init__(self, long_tickers, short_tickers):
        super(SelectThese, self).__init__()
        self.long_tickers = long_tickers
        self.short_tickers = short_tickers

    def __call__(self, target):
        if 'selected' not in target.temp.keys():
            target.temp['selected'] = {'long': [], 'short': []}
            return True

        tmp = {}
        tmp['long'] = [x for x in self.long_tickers if x in target.temp['selected']['long']]
        tmp['short'] = [x for x in self.short_tickers if x in target.temp['selected']['short']]
        target.temp['selected'] = tmp
        return True# }}}

class SelectN(Algo):# {{{

    """
    Sets temp['selected'] based on ranking temp[`stat_name`].

    Selects the top or botton N items based on temp[`stat_name`].
    If long_short is True, then the Top and Bottom N will be selected.
    This is usually some kind of metric that will be computed in a
    previous Algo and will be used for ranking purposes. Can select
    top or bottom N based on sort_descending parameter.

    Args:
        * n (int): select top n items.
        * stat_name (string): the name of the stat to use for ranking
        * sort_descending (bool): Should the stat be sorted in descending order
          before selecting the first n items?
        * longs (bool): include the top `n` as longs
        * shorts (bool): include the bottom `n` as shorts
        * all_or_none (bool): If true, only populates temp['selected'] if we
          have n items. If we have less than n, then temp['selected'] = [].

    Sets:
        * selected

    Requires:
        * selected
        * stat_name (string) the name of the stat stored in temp to be selected

    """

    def __init__(
        self, n, stat_name='stat', longs=True, shorts=False, sort_descending=True, all_or_none=False):
        super(SelectN, self).__init__()
        if n < 0:
            raise ValueError("n cannot be negative")
        self.n = n
        self.stat_name = stat_name
        self.ascending = not sort_descending
        self.all_or_none = all_or_none
        self.longs = longs
        self.shorts = shorts

    def __call__(self, target):
        stat = target.temp[self.stat_name].dropna()
        stat_long = stat.loc[stat.index.intersection(target.temp['selected']['long'])]
        stat_short = stat.loc[stat.index.intersection(target.temp['selected']['short'])]

        stat_long.sort_values(ascending=self.ascending, inplace=True)
        stat_short.sort_values(ascending=self.ascending, inplace=True)

        # handle percent n
        keep_n = self.n
        if self.n < 1:
            keep_n = int(self.n * len(set(stat_long + stat_short)))

        sel_long = list(stat[:keep_n].index)
        sel_short = list(stat[-keep_n:].index)

        if self.longs and self.shorts:
            fin_sel_long = [x for x in sel_long if x not in sel_short]
            fin_sel_short = [x for x in sel_short if x not in sel_long]
        elif self.longs:
            fin_sel_long = sel_long
            fin_sel_short = []
        elif self.shorts:
            fin_sel_long = []
            fin_sel_short = sel_short


        if self.all_or_none and (len(fin_sel_long) < keep_n or len(fin_sel_short) < keep_n):
            fin_sel_long = []
            fin_sel_short = []

        target.temp["selected"]['long'] = fin_sel_long
        target.temp["selected"]['short'] = fin_sel_short

        return True# }}}

class SelectWhere(Algo):# {{{

    """
    Selects securities based on an indicator DataFrame.

    Selects securities where the value is True on the current date
    (target.now) only if current date is present in signal DataFrame.

    For example, this could be the result of a pandas boolean comparison such
    as data > 100.

    Args:
        * signal (str|DataFrame): Boolean DataFrame containing selection logic.
          If a string is passed, frame is accessed using target.get_data
          This is the preferred way of using the algo.

    Requires:
        * selected

    Sets:
        * selected

    """

    def __init__(self, signal):
        super(SelectWhere, self).__init__()
        if isinstance(signal, pd.DataFrame):
            self.signal_name = None
            self.signal = signal
        else:
            self.signal_name = signal
            self.signal = None

    def __call__(self, target):
        if 'selected' not in target.temp.keys():
            target.temp['selected'] = {'long': [], 'short': []}
            return True
        longs = target.temp['selected']['long']
        shorts = target.temp['selected']['short']

        # get signal Series at target.now
        if self.signal_name is None:
            signal = self.signal
        else:
            signal = target.get_data(self.signal_name)

        if target.now in signal.index:
            sig = signal.loc[target.now]
            # get tickers where True
            # selected = sig.index[sig]
            long_signal = list(sig[sig == True].index)  # noqa: E712
            short_signal = list(sig[sig == False].index) # noqa: E712

            long_signal = [x for x in long_signal if x in longs]
            short_signal = [x for x in short_signal if x in shorts]

            target.temp['selected']['long'] = long_signal
            target.temp['selected']['short'] = short_signal

        else:
            target.temp['selected']['long'] = []
            target.temp['selected']['short'] = []
        return True# }}}

class SelectActive(Algo):# {{{

    """
    Sets temp['selected'] based on filtering temp['selected'] to exclude
    those securities that have been closed or rolled after a certain date
    using ClosePositionsAfterDates or RollPositionsAfterDates. This makes sure
    not to select them again for weighting (even if they have prices).

    Requires:
        * selected
        * perm['closed'] or perm['rolled']

    Sets:
        * selected

    """

    def __call__(self, target):
        longs = target.temp['selected']['long']
        shorts = target.temp['selected']['short']
        selected = longs + shorts
        rolled = target.perm.get("rolled", set())
        closed = target.perm.get("closed", set())
        selected = [s for s in selected if s not in set.union(rolled, closed)]
        target.temp['selected']['long'] = [x for x in longs if x in selected]
        target.temp['selected']['short'] = [x for x in shorts if x in selected]
        return True# }}}

class SelectRandomly(Algo):# {{{

    """
    Sets temp['selected'] based on a random subset of
    the items currently in temp['selected'].

    Selects n random elements from the list stored in temp['selected'].
    This is useful for benchmarking against a strategy where we believe
    the selection algorithm is adding value.

    For example, if we are testing a momentum strategy and we want to see if
    selecting securities based on momentum is better than just selecting
    securities randomly, we could use this Algo to create a random Strategy
    used for random benchmarking.

    Note:
        Another selection algorithm should be use prior to this Algo to
        populate temp['selected']. This will typically be SelectAll.

    Args:
        * n (int): Select N elements randomly.

    Sets:
        * selected

    Requires:
        * selected

    """

    def __init__(self, n=None):
        super(SelectRandomly, self).__init__()
        self.n = n

    def __call__(self, target):
        if 'selected' not in target.temp.keys():
            target.temp['selected'] = {'long': [], 'short': []}
            return True
        longs = target.temp['selected']['long']
        shorts = target.temp['selected']['shorts']

        if self.n is not None:
            n_long = self.n if self.n < len(longs) else len(longs)
            sel_long = random.sample(longs, int(n))
            n_shorts = self.n if self.n < len(shorts) else len(shorts)
            sel_short = random.sample(shorts, int(n))

        target.temp['selected']['long'] = sel_long
        target.temp['selected']['short'] = sel_short
        return True# }}}

class SelectRegex(Algo):# {{{

    """
    Sets temp['selected'] based on a regex on their names.
    Useful when working with a large universe of different kinds of securities

    Args:
        * regex (str): regular expression on the name
        * longs (bool): apply this to longs
        * shorts (bool): apply this to shorts

    Sets:
        * selected

    Requires:
        * selected
    """

    def __init__(self, regex, longs=True, shorts=True):
        super(SelectRegex, self).__init__()
        self.regex = re.compile(regex)
        self.longs = longs
        self.shorts = shorts

    def __call__(self, target):
        if 'selected' not in target.temp.keys():
            target.temp['selected'] = {'long': [], 'short': []}
            return True
        if self.longs:
            selected = target.temp["selected"]['long']
            selected = [s for s in selected if self.regex.search(s)]
            target.temp["selected"]['long'] = selected
        if self.shorts:
            selected = target.temp['selected']['short']
            selected = [s for s in selected if self.regex.search(s)]
            target.temp['selected']['short'] = selected
        return True# }}}

class SelectMomentum(AlgoStack):# {{{

    """
    Sets temp['selected'] based on a simple momentum filter.

    Selects the top n securities based on the total return over
    a given lookback period. This is just a wrapper around an
    AlgoStack with two algos: StatTotalReturn and SelectN.

    Note, that SelectAll() or similar should be called before
    SelectMomentum(), as StatTotalReturn uses values of temp['selected']

    Args:
        * n (int): select first N elements
        * lookback (int): lookback period for total return
          calculation
        * lag (int): Lag interval for total return calculation
        * sort_descending (bool): Sort descending (highest return is best)
        * all_or_none (bool): If true, only populates temp['selected'] if we
          have n items. If we have less than n, then temp['selected'] = [].
        * long_short (bool): if True, will also populate temp['selected']['short']
          with the lowest values by stat
        * longs (bool) if True, will populate top N in temp['selected']['long']
        * shorts (bool) if True, will populate top N in temp['selected']['shorts']

    Sets:
        * selected

    Requires:
        * selected

    """

    def __init__(
        self,
        n,
        sort_descending=True,
        longs=True,
        shorts=False,
        all_or_none=False,
        lookback=252,
        lag=1,
    ):
        from .stat_algos import StatTotalReturn
        super(SelectMomentum, self).__init__(
            StatTotalReturn(lookback=lookback, lag=lag),
            SelectN(n=n, stat_name='total_return',
                sort_descending=sort_descending, all_or_none=all_or_none,
                longs=longs, shorts=shorts),
        )# }}}

class SelectTypes(Algo):# {{{
    """
    Sets temp['selected'] based on node type.
    If temp['selected'] is already set, it will filter the existing
    selection.

    Args:
        * include_types (list): Types of nodes to include
        * exclude_types (list): Types of nodes to exclude

    Requires:
        * selected

    Sets:
        * selected
    """

    def __init__(self, include_types=(bt.core.Node,), exclude_types=()):
        super(SelectTypes, self).__init__()
        self.include_types = include_types
        self.exclude_types = exclude_types or (type(None),)

    def __call__(self, target):
        selected = [
            sec_name
            for sec_name, sec in target.children.items()
            if isinstance(sec, self.include_types)
            and not isinstance(sec, self.exclude_types)
        ]

        if 'selected' not in target.temp.keys():
            target.temp['selected'] = {'long': [], 'short': []}
            return True

        longs = [s for s in selected if s in target.temp["selected"]['long']]
        shorts = [s for s in selected if s in target.temp["selected"]['short']]
        target.temp['selected']['long'] = longs
        target.temp['selected']['shorts'] = longs
        return True# }}}

class ResolveOnTheRun(Algo):# {{{

    """
    Looks at securities set in temp['selected'] and searches for names that
    match the names of "aliases" for on-the-run securities in the provided
    data. Then replaces the alias with the name of the underlying security
    appropriate for the given date, and sets it back on temp['selected']

    Args:
        * on_the_run (str): Name of a Data frame with
            - columns set to "on the run" ticker names
            - index set to the timeline for the backtest
            - values are the actual security name to use for the given date
        * include_no_data (bool): Include securities that do not have data?
        * include_negative (bool): Include securities that have negative
          or zero prices?

    Requires:
        * selected

    Sets:
        * selected

    """

    def __init__(self, on_the_run, include_no_data=False, include_negative=False):
        super(ResolveOnTheRun, self).__init__()
        self.on_the_run = on_the_run
        self.include_no_data = include_no_data
        self.include_negative = include_negative

    def __call__(self, target):
        # Resolve real tickers based on OTR
        on_the_run = target.get_data(self.on_the_run)
        selected = target.temp["selected"]['long']
        aliases = [s for s in selected if s in on_the_run.columns]
        resolved = on_the_run.loc[target.now, aliases].tolist()
        target.temp["selected"]['long'] = resolved + [
            s for s in selected if s not in on_the_run.columns
        ]
        selected = target.temp["selected"]['short']
        aliases = [s for s in selected if s in on_the_run.columns]
        resolved = on_the_run.loc[target.now, aliases].tolist()
        target.temp["selected"]['short'] = resolved + [
            s for s in selected if s not in on_the_run.columns
        ]
        return True# }}}

class SelectApplyToStat(Algo):# {{{
    """
    Applies a function to `stat_name` (stored in temp) and then
    selects long anything that returns True, and short anything that
    returns false

    Args:
        * stat_name (str): name of the stat stored in temp
        * func (callable): the function to be applied to the stat
        * longs (bool): include securities where the function evaluates to True as longs
        * shorts (bool): include securities where the function evaluates to True as longs

    Requires:
        * selected

    Sets:
        * selected
    """

    def __init__(self, stat_name, func, longs=True, shorts=False):
        self.longs = longs
        self.shorts = shorts
        self.stat_name = stat_name
        self.func = func

    def __call__(self, target):

        if 'selected' not in target.temp.keys():
            return True

        stat = target.temp[self.stat_name]

        res = self.func(stat)

        longs = list((res.loc[res == True]).index)
        shorts = list((res.loc[res == False]).index)

        if self.longs:
            target.temp['selected']['long'] = [x for x in target.temp['selected']['long'] if x in longs]
        else:
            target.temp['selected']['long'] = []
        if self.shorts:
            target.temp['selected']['short'] = [x for x in target.temp['selected']['short'] if x in shorts]
        else:
            target.temp['selected']['short'] = []

        return True# }}}


