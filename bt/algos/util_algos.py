"""
A collection of algos that provide utility functions such as printing and debugging
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

class PrintDate(Algo):# {{{

    """
    This Algo simply print's the current date.

    Can be useful for debugging purposes.
    """

    def __call__(self, target):
        print(target.now)
        return True# }}}

class PrintTempData(Algo):# {{{

    """
    This Algo prints the temp data.

    Useful for debugging.

    Args:
        * fmt_string (str): A string that will later be formatted with the
          target's temp dict. Therefore, you should provide
          what you want to examine within curly braces ( { } )
    """

    def __init__(self, fmt_string=None):
        super(PrintTempData, self).__init__()
        self.fmt_string = fmt_string

    def __call__(self, target):
        if self.fmt_string:
            print(self.fmt_string.format(**target.temp))
        else:
            print(target.temp)
        return True# }}}

class PrintInfo(Algo):# {{{

    """
    Prints out info associated with the target strategy. Useful for debugging
    purposes.

    Args:
        * fmt_string (str): A string that will later be formatted with the
          target object's __dict__ attribute. Therefore, you should provide
          what you want to examine within curly braces ( { } )

    Ex:
        PrintInfo('Strategy {name} : {now}')


    This will print out the name and the date (now) on each call.
    Basically, you provide a string that will be formatted with target.__dict__

    """

    def __init__(self, fmt_string="{name} {now}"):
        super(PrintInfo, self).__init__()
        self.fmt_string = fmt_string

    def __call__(self, target):
        print(self.fmt_string.format(**target.__dict__))
        return True# }}}

class Debug(Algo):# {{{

    """
    Utility Algo that calls pdb.set_trace when triggered.

    In the debug session, 'target' is available and can be examined through the
    StrategyBase interface.
    """

    def __call__(self, target):
        import pdb

        pdb.set_trace()
        return True# }}}

class SetCash(Algo):# {{{

    """
    algo to set a cash in percentage. This is used in `bt.algos.Rebalance` as
    the amount of cash to set aside.
    """

    def __init__(self, cash=0.0):
        self.cash = cash

    def __call__(self, target):
        target.temp['cash'] = self.cash
        return True# }}}

class PrintRisk(Algo):# {{{

    """
    This Algo prints the risk data.

    Args:
        * fmt_string (str): A string that will later be formatted with the
          target object's risk attributes. Therefore, you should provide
          what you want to examine within curly braces ( { } )
          If not provided, will print the entire dictionary with no formatting.
    """

    def __init__(self, fmt_string=""):
        super(PrintRisk, self).__init__()
        self.fmt_string = fmt_string

    def __call__(self, target):
        if hasattr(target, "risk"):
            if self.fmt_string:
                print(self.fmt_string.format(**target.risk))
            else:
                print(target.risk)
        return True# }}}

class DebugPortfolioLevel(Algo):# {{{
    """
    Print portfolio level information relevant to this strategy
    """
    def __call__( self, target ):
        flows = target.flows.loc[ target.now ]
        if flows:
            fmt_str = '{now} {name}: Price = {price:>6.2f}, Value = {value:>10,.0f}, Flows = {flows:>8,.0f}'
        else:
            fmt_str = '{now} {name}: Price = {price:>6.2f}, Value = {value:>10,.0f}'
        print( fmt_str.format(
            now = target.now,
            name = target.name,
            price = target.price,
            value = target.value,
            flows = flows
            ) )
        return True
        # }}}

class DebugTradeLevel(Algo):# {{{
    """
    Print trade level information
    """
    def __call__( self, target ):
        flows = target.flows.loc[ target.now ]
        # Check that sub-strategy is active (and not paper trading, which is always active)
        #if (target.capital > 0 or flows != 0) and target.parent != target:
        #     if flows:
        #         fmt_str = '{name:>33}: Price = {price:>6.2f}, Value = {value:>10,.0f}, Flows = {flows:>8,.0f}'
        #     else:
        fmt_str = '{name:>33}: Price = {price:>6.2f}, Value = {value:>10,.0f}'
        print(fmt_str.format(
            now = target.now,
            name = target.name,
            price = target.price,
            value = target.value,
            ))
        return True# }}}
