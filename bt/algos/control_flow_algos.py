"""
A collection of Algos used to control the flow
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

def run_always(f):# {{{
    """
    Run always decorator to be used with Algo
    to ensure stack runs the decorated Algo
    on each pass, regardless of failures in the stack.
    """
    f.run_always = True
    return f# }}}

class Require(Algo):# {{{

    """
    Flow control Algo.

    This algo returns the value of a predicate
    on an temp entry. Useful for controlling
    flow.

    For example, we might want to make sure we have some items selected.
    We could pass a lambda function that checks the len of 'selected':

        pred=lambda x: len(x) == 0
        item='selected'

    Args:
        * pred (Algo): Function that returns a Bool given the strategy. This
          is the definition of an Algo. However, this is typically used
          with a simple lambda function.
        * item (str): An item within temp.
        * if_none (bool): Result if the item required is not in temp or if it's
          value if None

    """

    def __init__(self, pred, item, if_none=False):
        super(Require, self).__init__()
        self.item = item
        self.pred = pred
        self.if_none = if_none

    def __call__(self, target):
        if self.item not in target.temp:
            return self.if_none

        item = target.temp[self.item]

        if item is None:
            return self.if_none

        return self.pred(item)# }}}

class Not(Algo):# {{{
    """
    Flow control Algo

    It is usful for "inverting" other flow control algos,
    For example Not( RunAfterDate(...) ), Not( RunAfterDays(...) ), etc

    Args:
        * list_of_algos (Algo): The algo to run and invert the return value of
    """

    def __init__(self, algo):
        super(Not, self).__init__()
        self._algo = algo

    def __call__(self, target):
        return not self._algo(target)# }}}

class Or(Algo):# {{{
    """
    Flow control Algo

    It useful for combining multiple signals into one signal.
    For example, we might want two different rebalance signals to work together:

        runOnDateAlgo = bt.algos.RunOnDate(pdf.index[0]) # where pdf.index[0] is the first date in our time series
        runMonthlyAlgo = bt.algos.RunMonthly()
        orAlgo = Or([runMonthlyAlgo,runOnDateAlgo])

    orAlgo will return True if it is the first date or if it is 1st of the month

    Args:
        * list_of_algos: Iterable list of algos.
          Runs each algo and
          returns true if any algo returns true.
    """

    def __init__(self, list_of_algos):
        super(Or, self).__init__()
        self._list_of_algos = list_of_algos
        return

    def __call__(self, target):
        res = False
        for algo in self._list_of_algos:
            tempRes = algo(target)
            res = res | tempRes

        return res# }}}

class RunOnce(Algo):# {{{

    """
    Returns True on first run then returns False.

    Args:
        * run_on_first_call: bool which determines if it runs the first time the algo is called

    As the name says, the algo only runs once. Useful in situations
    where we want to run the logic once (buy and hold for example).

    """

    def __init__(self):
        super(RunOnce, self).__init__()
        self.has_run = False

    def __call__(self, target):
        # if it hasn't run then we will
        # run it and set flag
        if not self.has_run:
            self.has_run = True
            return True

        # return false to stop future execution
        return False# }}}

class RunPeriod(Algo):# {{{
    def __init__(
        self, run_on_first_date=True, run_on_end_of_period=False, run_on_last_date=False
    ):
        super(RunPeriod, self).__init__()
        self._run_on_first_date = run_on_first_date
        self._run_on_end_of_period = run_on_end_of_period
        self._run_on_last_date = run_on_last_date

    def __call__(self, target):
        # get last date
        now = target.now

        # if none nothing to do - return false
        if now is None:
            return False

        # not a known date in our universe
        if now not in target.data.index:
            return False

        # get index of the current date
        index = target.data.index.get_loc(target.now)

        result = False

        # index 0 is a date added by the Backtest Constructor
        if index == 0:
            return False
        # first date
        if index == 1:
            if self._run_on_first_date:
                result = True
        # last date
        elif index == (len(target.data.index) - 1):
            if self._run_on_last_date:
                result = True
        else:

            # create pandas.Timestamp for useful .week,.quarter properties
            now = pd.Timestamp(now)

            index_offset = -1
            if self._run_on_end_of_period:
                index_offset = 1

            date_to_compare = target.data.index[index + index_offset]
            date_to_compare = pd.Timestamp(date_to_compare)

            result = self.compare_dates(now, date_to_compare)

        return result

    @abc.abstractmethod
    def compare_dates(self, now, date_to_compare):
        raise (NotImplementedError("RunPeriod Algo is an abstract class!"))# }}}

class RunDaily(RunPeriod):# {{{

    """
    Returns True on day change.

    Args:
        * run_on_first_date (bool): determines if it runs the first time the algo is called
        * run_on_end_of_period (bool): determines if it should run at the end of the period
          or the beginning
        * run_on_last_date (bool): determines if it runs on the last time the algo is called

    Returns True if the target.now's day has changed
    compared to the last(or next if run_on_end_of_period) date, if not returns False.
    Useful for daily rebalancing strategies.

    """

    def compare_dates(self, now, date_to_compare):
        if now.date() != date_to_compare.date():
            return True
        return False# }}}

class RunWeekly(RunPeriod):# {{{

    """
    Returns True on week change.

    Args:
        * run_on_first_date (bool): determines if it runs the first time the algo is called
        * run_on_end_of_period (bool): determines if it should run at the end of the period
          or the beginning
        * run_on_last_date (bool): determines if it runs on the last time the algo is called

    Returns True if the target.now's week has changed
    since relative to the last(or next) date, if not returns False. Useful for
    weekly rebalancing strategies.

    """

    def compare_dates(self, now, date_to_compare):
        if now.year != date_to_compare.year or now.week != date_to_compare.week:
            return True
        return False# }}}

class RunMonthly(RunPeriod):# {{{

    """
    Returns True on month change.

    Args:
        * run_on_first_date (bool): determines if it runs the first time the algo is called
        * run_on_end_of_period (bool): determines if it should run at the end of the period
          or the beginning
        * run_on_last_date (bool): determines if it runs on the last time the algo is called

    Returns True if the target.now's month has changed
    since relative to the last(or next) date, if not returns False. Useful for
    monthly rebalancing strategies.

    """

    def compare_dates(self, now, date_to_compare):
        if now.year != date_to_compare.year or now.month != date_to_compare.month:
            return True
        return False# }}}

class RunQuarterly(RunPeriod):# {{{

    """
    Returns True on quarter change.

    Args:
        * run_on_first_date (bool): determines if it runs the first time the algo is called
        * run_on_end_of_period (bool): determines if it should run at the end of the period
          or the beginning
        * run_on_last_date (bool): determines if it runs on the last time the algo is called

    Returns True if the target.now's quarter has changed
    since relative to the last(or next) date, if not returns False. Useful for
    quarterly rebalancing strategies.

    """

    def compare_dates(self, now, date_to_compare):
        if now.year != date_to_compare.year or now.quarter != date_to_compare.quarter:
            return True
        return False# }}}

class RunYearly(RunPeriod):# {{{

    """
    Returns True on year change.

    Args:
        * run_on_first_date (bool): determines if it runs the first time the algo is called
        * run_on_end_of_period (bool): determines if it should run at the end of the period
          or the beginning
        * run_on_last_date (bool): determines if it runs on the last time the algo is called

    Returns True if the target.now's year has changed
    since relative to the last(or next) date, if not returns False. Useful for
    yearly rebalancing strategies.

    """

    def compare_dates(self, now, date_to_compare):
        if now.year != date_to_compare.year:
            return True
        return False# }}}

class RunOnDate(Algo):# {{{

    """
    Returns True on a specific set of dates.

    Args:
        * dates (list): List of dates to run Algo on.

    """

    def __init__(self, *dates):
        """
        Args:
            * dates (*args): A list of dates. Dates will be parsed
              by pandas.to_datetime so pass anything that it can
              parse. Typically, you will pass a string 'yyyy-mm-dd'.
        """
        super(RunOnDate, self).__init__()
        # parse dates and save
        self.dates = [pd.to_datetime(d) for d in dates]

    def __call__(self, target):
        return target.now in self.dates# }}}

class RunAfterDate(Algo):# {{{

    """
    Returns True after a date has passed

    Args:
        * date: Date after which to start trading

    Note:
        This is useful for algos that rely on trailing averages where you
        don't want to start trading until some amount of data has been built up

    """

    def __init__(self, date):
        """
        Args:
            * date: Date after which to start trading
        """
        super(RunAfterDate, self).__init__()
        # parse dates and save
        self.date = pd.to_datetime(date)

    def __call__(self, target):
        return target.now > self.date# }}}

class RunAfterDays(Algo):# {{{

    """
    Returns True after a specific number of 'warmup' trading days have passed

    Args:
        * days (int): Number of trading days to wait before starting

    Note:
        This is useful for algos that rely on trailing averages where you
        don't want to start trading until some amount of data has been built up

    """

    def __init__(self, days):
        """
        Args:
            * days (int): Number of trading days to wait before starting
        """
        super(RunAfterDays, self).__init__()
        self.days = days

    def __call__(self, target):
        if self.days > 0:
            self.days -= 1
            return False
        return True# }}}

class RunIfOutOfBounds(Algo):# {{{

    """
    This algo returns true if any of the target weights deviate by an amount greater
    than tolerance. For example, it will be run if the tolerance is set to 0.5 and
    a security grows from a target weight of 0.2 to greater than 0.3.

    A strategy where rebalancing is performed quarterly or whenever any
    security's weight deviates by more than 20% could be implemented by:

        Or([runQuarterlyAlgo,runIfOutOfBoundsAlgo(0.2)])

    Args:
        * tolerance (float): Allowed deviation of each security weight.

    Requires:
        * Weights

    """

    def __init__(self, tolerance):
        self.tolerance = float(tolerance)
        super(RunIfOutOfBounds, self).__init__()

    def __call__(self, target):
        if "weights" not in target.temp:
            return True

        targets = target.temp["weights"]

        for cname in target.children:
            if cname in targets:
                c = target.children[cname]
                deviation = abs((c.weight - targets[cname]) / targets[cname])
                if deviation > self.tolerance:
                    return True

        if "cash" in target.temp:
            cash_deviation = abs(
                (target.capital - targets.value) / targets.value - target.temp["cash"]
            )
            if cash_deviation > self.tolerance:
                return True

        return False# }}}

class RunEveryNPeriods(Algo):# {{{

    """
    This algo runs every n periods.

    Args:
        * n (int): Run each n periods
        * offset (int): Applies to the first run. If 0, this algo will run the
          first time it is called.

    This Algo can be useful for the following type of strategy:
        Each month, select the top 5 performers. Hold them for 3 months.

    You could then create 3 strategies with different offsets and create a
    master strategy that would allocate equal amounts of capital to each.

    """

    def __init__(self, n, offset=0):
        super(RunEveryNPeriods, self).__init__()
        self.n = n
        self.offset = offset
        self.idx = n - offset - 1
        self.lcall = 0

    def __call__(self, target):
        # ignore multiple calls on same period
        if self.lcall == target.now:
            return False
        else:
            self.lcall = target.now
            # run when idx == (n-1)
            if self.idx == (self.n - 1):
                self.idx = 0
                return True
            else:
                self.idx += 1
                return False# }}}
