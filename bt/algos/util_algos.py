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
