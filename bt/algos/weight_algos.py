"""
A collection of algos used to define security weights
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

class WeighEqually(Algo):# {{{

    """
    Sets temp['weights'] by calculating equal weights for all items in
    selected.

    Equal weight Algo. Sets the 'weights' to 1/n for each item in 'selected'.

    Sets:
        * weights

    Requires:
        * selected

    """

    def __init__(self):
        super(WeighEqually, self).__init__()

    def __call__(self, target):
        longs = target.temp['selected']['long']
        shorts = target.temp['selected']['short']
        selected = longs + shorts
        if len(selected) != len(set(shorts + longs)):
            raise ValueError(f'duplicates in selected {target.temp["selected"]}')
        # print(target.temp['selected'])
        n = len(selected)

        if n == 0:
            target.temp["weights"] = {}
        else:
            w = 1.0 / n
            target.temp['weights'] = {x: w for x in longs}
            # target.temp['weights'] = {x: -w for x in selected if x in shorts}
            target.temp['weights'].update({x: -w for x in shorts})
        return True# }}}

class WeighSpecified(Algo):# {{{

    """
    Sets temp['weights'] based on a provided dict of ticker:weights.

    Sets the weights based on pre-specified targets.

    Args:
        * weights (dict): target weights -> ticker: weight

    Sets:
        * weights

    """

    def __init__(self, **weights):
        super(WeighSpecified, self).__init__()
        self.weights = weights

    def __call__(self, target):
        # added copy to make sure these are not overwritten
        target.temp["weights"] = self.weights.copy()
        return True# }}}

class ScaleWeights(Algo):# {{{

    """
    Sets temp['weights'] based on a scaled version of itself.
    Useful for going short, or scaling up/down when using
    :class:`FixedIncomeStrategy <bt.core.FixedIncomeStrategy>`.

    Args:
        * scale (float): the scaling factor

    Sets:
        * weights

    Requires:
        * weights

    """

    def __init__(self, scale):
        super(ScaleWeights, self).__init__()
        self.scale = scale

    def __call__(self, target):
        target.temp["weights"] = {
            k: self.scale * w for k, w in iteritems(target.temp["weights"])
        }
        return True# }}}

class WeighTarget(Algo):# {{{

    """
    Sets target weights based on a target weight DataFrame.

    If the target weight dataFrame is  of same dimension
    as the target.universe, the portfolio will effectively be rebalanced on
    each period. For example, if we have daily data and the target DataFrame
    is of the same shape, we will have daily rebalancing.

    However, if we provide a target weight dataframe that has only month end
    dates, then rebalancing only occurs monthly.

    Basically, if a weight is provided on a given date, the target weights are
    set and the algo moves on (presumably to a Rebalance algo). If not, not
    target weights are set.

    Args:
        * weights (str|DataFrame): DataFrame containing the target weights
          If a string is passed, frame is accessed using target.get_data
          This is the preferred way of using the algo.

    Sets:
        * weights

    """

    def __init__(self, weights):
        super(WeighTarget, self).__init__()
        if isinstance(weights, pd.DataFrame):
            self.weights_name = None
            self.weights = weights
        else:
            self.weights_name = weights
            self.weights = None

    def __call__(self, target):
        # get current target weights
        if self.weights_name is None:
            weights = self.weights
        else:
            weights = target.get_data(self.weights_name)

        if target.now in weights.index:
            w = weights.loc[target.now]

            # dropna and save
            target.temp["weights"] = w.dropna()

            return True
        else:
            return False# }}}

class WeighInvVol(Algo):# {{{

    """
    Sets temp['weights'] based on the inverse volatility Algo.

    Sets the target weights proportionally to the securities volatility,
    resulting in a portfolio where each position has the same level of
    volatility.

    This is a commonly used technique for risk parity portfolios. The least
    volatile elements receive the highest weight under this scheme. Weights
    are proportional to the inverse of their volatility.

    Args:
        * lookback (DateOffset): lookback period for estimating volatility

    Sets:
        * weights

    Requires:
        * selected

    """

    def __init__(self, lookback=63, lag=1):
        super(WeighInvVol, self).__init__()
        self.lookback = lookback
        self.lag = lag

    def __call__(self, target):
        longs = target.temp['selected']['long']
        shorts = target.temp['selected']['short']
        selected = longs + shorts
        if len(set(longs + shorts)) != len(selected):
            raise ValueError(f'duplicates in selected {target.temp["selected"]}')

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True

        if len(selected) == 1:
            if len(longs) == 1:
                target.temp["weights"] = {selected[0]: 1.0}
            if len(shorts) == 1:
                target.temp["weights"] = {selected[0]: -1.0}
            return True

        univ = target.universe
        univ = univ.loc[univ.index <= target.now]
        univ = univ.iloc[-self.lookback:-self.lag]

        rets = univ.pct_change().dropna(how='all')

        vol = np.divide(1.0, np.std(rets, ddof=1))
        vol[np.isinf(vol)] = np.NaN
        volsum = vol.sum()
        tw = np.divide(vol, volsum)
        target.temp['weights'] = {}

        for k, v in tw.items():
            if k in longs:
                target.temp['weights'][k] = v
            elif k in shorts:
                target.temp['weights'][k] = -v

        return True# }}}

class WeighERC(Algo):# {{{

    """
    Sets temp['weights'] based on equal risk contribution algorithm.

    Sets the target weights based on ffn's calc_erc_weights. This
    is an extension of the inverse volatility risk parity portfolio in
    which the correlation of asset returns is incorporated into the
    calculation of risk contribution of each asset.

    The resulting portfolio is similar to a minimum variance portfolio
    subject to a diversification constraint on the weights of its components
    and its volatility is located between those of the minimum variance and
    equally-weighted portfolios (Maillard 2008).

    See:
        https://en.wikipedia.org/wiki/Risk_parity

    Args:
        * lookback (DateOffset): lookback period for estimating covariance
        * initial_weights (list): Starting asset weights [default inverse vol].
        * risk_weights (list): Risk target weights [default equal weight].
        * covar_method (str): method used to estimate the covariance. See ffn's
          calc_erc_weights for more details. (default ledoit-wolf).
        * risk_parity_method (str): Risk parity estimation method. see ffn's
          calc_erc_weights for more details. (default ccd).
        * maximum_iterations (int): Maximum iterations in iterative solutions
          (default 100).
        * tolerance (float): Tolerance level in iterative solutions (default 1E-8).


    Sets:
        * weights

    Requires:
        * selected

    """

    def __init__(
        self,
        lookback=pd.DateOffset(months=3),
        initial_weights=None,
        risk_weights=None,
        covar_method="ledoit-wolf",
        risk_parity_method="ccd",
        maximum_iterations=100,
        tolerance=1e-8,
        lag=pd.DateOffset(days=0),
    ):

        super(WeighERC, self).__init__()
        self.lookback = lookback
        self.initial_weights = initial_weights
        self.risk_weights = risk_weights
        self.covar_method = covar_method
        self.risk_parity_method = risk_parity_method
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance
        self.lag = lag

    def __call__(self, target):
        selected = target.temp["selected"]

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True

        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        t0 = target.now - self.lag
        prc = target.universe.loc[t0 - self.lookback : t0, selected]
        tw = bt.ffn.calc_erc_weights(
            prc.to_returns().dropna(),
            initial_weights=self.initial_weights,
            risk_weights=self.risk_weights,
            covar_method=self.covar_method,
            risk_parity_method=self.risk_parity_method,
            maximum_iterations=self.maximum_iterations,
            tolerance=self.tolerance,
        )

        target.temp["weights"] = tw.dropna()
        return True# }}}

class WeighMeanVar(Algo):# {{{

    """
    Sets temp['weights'] based on mean-variance optimization.

    Sets the target weights based on ffn's calc_mean_var_weights. This is a
    Python implementation of Markowitz's mean-variance optimization.

    See:
        http://en.wikipedia.org/wiki/Modern_portfolio_theory#The_efficient_frontier_with_no_risk-free_asset

    Args:
        * lookback (DateOffset): lookback period for estimating volatility
        * bounds ((min, max)): tuple specifying the min and max weights for
          each asset in the optimization.
        * covar_method (str): method used to estimate the covariance. See ffn's
          calc_mean_var_weights for more details.
        * rf (float): risk-free rate used in optimization.

    Sets:
        * weights

    Requires:
        * selected

    """

    def __init__(
        self,
        lookback=pd.DateOffset(months=3),
        bounds=(0.0, 1.0),
        covar_method="ledoit-wolf",
        rf=0.0,
        lag=pd.DateOffset(days=0),
    ):
        super(WeighMeanVar, self).__init__()
        self.lookback = lookback
        self.lag = lag
        self.bounds = bounds
        self.covar_method = covar_method
        self.rf = rf

    def __call__(self, target):
        selected = target.temp["selected"]

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True

        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        t0 = target.now - self.lag
        prc = target.universe.loc[t0 - self.lookback : t0, selected]
        tw = bt.ffn.calc_mean_var_weights(
            prc.to_returns().dropna(),
            weight_bounds=self.bounds,
            covar_method=self.covar_method,
            rf=self.rf,
        )

        target.temp["weights"] = tw.dropna()
        return True# }}}

class WeighRandomly(Algo):# {{{

    """
    Sets temp['weights'] based on a random weight vector.

    Sets random target weights for each security in 'selected'.
    This is useful for benchmarking against a strategy where we believe
    the weighing algorithm is adding value.

    For example, if we are testing a low-vol strategy and we want to see if
    our weighing strategy is better than just weighing
    securities randomly, we could use this Algo to create a random Strategy
    used for random benchmarking.

    This is an Algo wrapper around ffn's random_weights function.

    Args:
        * bounds ((low, high)): Tuple including low and high bounds for each
          security
        * weight_sum (float): What should the weights sum up to?

    Sets:
        * weights

    Requires:
        * selected

    """

    def __init__(self, bounds=(0.0, 1.0), weight_sum=1):
        super(WeighRandomly, self).__init__()
        self.bounds = bounds
        self.weight_sum = weight_sum

    def __call__(self, target):
        sel = target.temp["selected"]
        n = len(sel)

        w = {}
        try:
            rw = bt.ffn.random_weights(n, self.bounds, self.weight_sum)
            w = dict(zip(sel, rw))
        except ValueError:
            pass

        target.temp["weights"] = w
        return True# }}}

class LimitDeltas(Algo):# {{{

    """
    Modifies temp['weights'] based on weight delta limits.

    Basically, this can be used if we want to restrict how much a security's
    target weight can change from day to day. Useful when we want to be more
    conservative about how much we could actually trade on a given day without
    affecting the market.

    For example, if we have a strategy that is currently long 100% one
    security, and the weighing Algo sets the new weight to 0%, but we
    use this Algo with a limit of 0.1, the new target weight will
    be 90% instead of 0%.

    Args:
        * limit (float, dict): Weight delta limit. If float, this will be a
          global limit for all securities. If dict, you may specify by-ticker
          limit.

    Sets:
        * weights

    Requires:
        * weights

    """

    def __init__(self, limit=0.1):
        super(LimitDeltas, self).__init__()
        self.limit = limit
        # determine if global or specific
        self.global_limit = True
        if isinstance(limit, dict):
            self.global_limit = False

    def __call__(self, target):
        tw = target.temp["weights"]
        all_keys = set(list(target.children.keys()) + list(tw.keys()))

        for k in all_keys:
            tgt = tw[k] if k in tw else 0.0
            cur = target.children[k].weight if k in target.children else 0.0
            delta = tgt - cur

            # check if we need to limit
            if self.global_limit:
                if abs(delta) > self.limit:
                    tw[k] = cur + (self.limit * np.sign(delta))
            else:
                # make sure we have a limit defined in case of limit dict
                if k in self.limit:
                    lmt = self.limit[k]
                    if abs(delta) > lmt:
                        tw[k] = cur + (lmt * np.sign(delta))

        return True# }}}

class LimitWeights(Algo):# {{{

    """
    Modifies temp['weights'] based on weight limits.

    This is an Algo wrapper around ffn's limit_weights. The purpose of this
    Algo is to limit the weight of any one specifc asset. For example, some
    Algos will set some rather extreme weights that may not be acceptable.
    Therefore, we can use this Algo to limit the extreme weights. The excess
    weight is then redistributed to the other assets, proportionally to
    their current weights.

    See ffn's limit_weights for more information.

    Args:
        * limit (float): Weight limit.

    Sets:
        * weights

    Requires:
        * weights

    """

    def __init__(self, limit=0.1):
        super(LimitWeights, self).__init__()
        self.limit = limit

    def __call__(self, target):
        if "weights" not in target.temp:
            return True

        tw = target.temp["weights"]
        if len(tw) == 0:
            return True

        # if the limit < equal weight then set weights to 0
        if self.limit < 1.0 / len(tw):
            tw = {}
        else:
            tw = bt.ffn.limit_weights(tw, self.limit)
        target.temp["weights"] = tw

        return True# }}}

class TargetVol(Algo):# {{{
    """
    Updates temp['weights'] based on the target annualized volatility desired.

    Args:
        * target_volatility: annualized volatility to target
        * lookback (DateOffset): lookback period for estimating volatility
        * lag (DateOffset): amount of time to wait to calculate the covariance
        * covar_method: method of calculating volatility
        * annualization_factor: number of periods to annualize by.
          It is assumed that target volatility is already annualized by this factor.

    Updates:
        * weights

    Requires:
        * temp['weights']


    """

    def __init__(
        self,
        target_volatility,
        lookback=63,
        lag=1,
        covar_method="standard",
        annualization_factor=252,
    ):

        super(TargetVol, self).__init__()
        self.target_volatility = target_volatility
        self.lookback = lookback
        self.lag = lag
        self.covar_method = covar_method
        self.annualization_factor = annualization_factor

    def __call__(self, target):

        current_weights = target.temp["weights"]
        selected = current_weights.keys()
        # import pdb; pdb.set_trace();

        # if there were no weights already set then skip
        if len(selected) == 0:
            return True

        uni = target.universe
        univ = uni.loc[uni.index <= target.now, list(selected)]
        uni = uni.iloc[-self.lookback:-self.lag]
        returns = uni.to_returns()

        # calc covariance matrix
        if self.covar_method == "ledoit-wolf":
            covar = sklearn.covariance.ledoit_wolf(returns)
        elif self.covar_method == "standard":
            covar = returns.cov()
        else:
            raise NotImplementedError("covar_method not implemented")

        covcol = covar.columns.tolist()
        covcol = [x for x in covcol if x in selected]
        covar = covar[covcol]
        covar = covar.loc[covar.index.isin(covcol)]
        weights = pd.Series(
            [current_weights[x] for x in covcol], index=covcol
        )

        vol = np.sqrt(
            np.matmul(weights.values.T, np.matmul(covar.values, weights.values))
            * self.annualization_factor
        )

        # vol is too high
        if vol > self.target_volatility:
            mult = self.target_volatility / vol
        # vol is too low
        elif vol < self.target_volatility:
            mult = self.target_volatility / vol
        else:
            mult = 1

        for k in target.temp["weights"].keys():
            target.temp["weights"][k] = target.temp["weights"][k] * mult

        return True# }}}

class ReScaleWeights(Algo):# {{{

    """
    modifies temp['weights'] by taking the current weight for all items
    selected and rescaling them to add up to the specified total weight

    Args:
        * total_weight (float): the new overall weight of the portfolio

    Sets:
        * weights

    Requires:
        * weights

    """

    def __init__(self, total_weight):
        self.scale = total_weight
        super(ReScaleWeights, self).__init__()

    def __call__(self, target):
        selected = target.temp['weights'].keys()
        n = len(selected)
        # where there are one or zero selected assets
        if n == 0:
            target.temp['weights'] = {}
            return True
        elif n == 1:
            target.temp['weights'] = {selected[0]: self.scale}
            return True


        weights = target.temp['weights'].copy()

        vals = list(weights.values)
        vals = [abs(x) for x in vals]
        total = sum(vals)

        factor = self.scale / total

        # target.temp['weights'] = {x: w * factor for x, w in weights}
        target.temp['weights'] = {
                k: w * factor for k, w in iteritems(weights)}
        return True# }}}
