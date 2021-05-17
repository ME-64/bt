"""
Contains backtesting logic and objects.
"""
from __future__ import division
from copy import deepcopy
import bt
import ffn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pyprind
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = 'plotly_dark'
pd.options.plotting.backend = 'plotly'
import olimutils
import plotly.express as px



def run(*backtests):# {{{
    """
    Runs a series of backtests and returns a Result
    object containing the results of the backtests.

    Args:
        * backtest (*list): List of backtests.

    Returns:
        Result

    """
    # run each backtest
    for bkt in backtests:
        bkt.run()

    return Result(*backtests)# }}}


def benchmark_random(backtest, random_strategy, nsim=100):# {{{
    """
    Given a backtest and a random strategy, compare backtest to
    a number of random portfolios.

    The idea here is to benchmark your strategy vs a bunch of
    random strategies that have a similar structure but execute
    some part of the logic randomly - basically you are trying to
    determine if your strategy has any merit - does it beat
    randomly picking weight? Or randomly picking the selected
    securities?

    Args:
        * backtest (Backtest): A backtest you want to benchmark
        * random_strategy (Strategy): A strategy you want to benchmark
          against. The strategy should have a random component to
          emulate skilless behavior.
        * nsim (int): number of random strategies to create.

    Returns:
        RandomBenchmarkResult

    """
    # save name for future use
    if backtest.name is None:
        backtest.name = "original"

    # run if necessary
    if not backtest.has_run:
        backtest.run()

    bts = []
    bts.append(backtest)
    data = backtest.data

    # create and run random backtests
    for i in range(nsim):
        random_strategy.name = "random_%s" % i
        rbt = bt.Backtest(random_strategy, data)
        rbt.run()

        bts.append(rbt)

    # now create new RandomBenchmarkResult
    res = RandomBenchmarkResult(*bts)

    return res # }}}

class Backtest(object):
    """# {{{
    A Backtest combines a Strategy with data to
    produce a Result.

    A backtest is basically testing a strategy over a data set.

    Note:
        The Strategy will be deepcopied so it is re-usable in other
        backtests. To access the backtested strategy, simply access
        the strategy attribute.

    Args:
        * strategy (Strategy, Node, StrategyBase): The Strategy to be tested.
        * data (DataFrame): DataFrame containing data used in backtest. This
          will be the Strategy's "universe".
        * name (str): Backtest name - defaults to strategy name
        * initial_capital (float): Initial amount of capital passed to
          Strategy.
        * commissions (fn(quantity, price)): The commission function
          to be used. Ex: commissions=lambda q, p: max(1, abs(q) * 0.01)
        * integer_positions (bool): Whether to use integer positions for securities
          in the backtest. This can have unintended consequences when prices are
          high relative to the amount of capital (i.e. though split-adjusted prices,
          or too-low of a capital amount), causing allocated positions to round to zero.
          While the default is True, try setting to False for more robust behavior.
        * progress_bar (Bool): Display progress bar while running backtest
        * additional_data (dict): Additional kwargs passed to StrategyBase.setup, after preprocessing
          This data can be retrieved by Algos using StrategyBase.get_data.
          The data may also be used by the Strategy itself, i.e.
            - ``bidoffer``: A DataFrame with the same format as 'data', will be used
              by the strategy for transaction cost modeling
            - ``coupons``: A DataFrame with the same format as 'data', will by used
              by :class:`CouponPayingSecurity <bt.core.CouponPayingSecurity>`
              to determine cashflows.
            - ``cost_long``/``cost_short``: A DataFrame with the same format as 'data',
              will by used
              by :class:`CouponPayingSecurity <bt.core.CouponPayingSecurity>`
              to calculate asymmetric holding cost of long (or short) positions.


    Attributes:
        * strategy (Strategy): The Backtest's Strategy. This will be a deepcopy
          of the Strategy that was passed in.
        * data (DataFrame): Data passed in
        * dates (DateTimeIndex): Data's index
        * initial_capital (float): Initial capital
        * name (str): Backtest name
        * stats (ffn.PerformanceStats): Performance statistics
        * has_run (bool): Run flag
        * weights (DataFrame): Weights of each component over time
        * security_weights (DataFrame): Weights of each security as a
          percentage of the whole portfolio over time
        * additional_data (dict): Additional data passed at construction

    """# }}}

    def __init__(# {{{
        self,
        strategy,
        data,
        name=None,
        initial_capital=1000000.0,
        commissions=None,
        integer_positions=True,
        progress_bar=False,
        additional_data=None,
    ):

        if data.columns.duplicated().any():
            cols = data.columns[data.columns.duplicated().tolist()].tolist()
            raise Exception(
                "data provided has some duplicate column names: \n%s \n"
                "Please remove duplicates!" % cols
            )

        # we want to reuse strategy logic - copy it!
        # basically strategy is a template
        self.strategy = deepcopy(strategy)
        self.strategy.use_integer_positions(integer_positions)

        self._process_data(data, additional_data)

        self.initial_capital = initial_capital
        self.name = name if name is not None else strategy.name
        self.progress_bar = progress_bar

        if commissions is not None:
            self.strategy.set_commissions(commissions)

        self.stats = {}
        self._original_prices = None
        self._weights = None
        self._sweights = None
        self.has_run = False# }}}

    def _process_data(self, data, additional_data):# {{{
        # add virtual row at t0-1day with NaNs
        # this is so that any trading action at t0 can be evaluated relative to
        # a clean starting point. This is related to #83. Basically, if you
        # have a big trade / commision on day 0, then the Strategy.prices will
        # be adjusted at 0, and hide the 'total' return. The series should
        # start at 100, but may start at 90, for example. Here, we add a
        # starting point at t0-1day, and this is the reference starting point
        data_new = pd.concat(
            [
                pd.DataFrame(
                    np.nan,
                    columns=data.columns,
                    index=[data.index[0] - pd.DateOffset(days=1)],
                ),
                data,
            ]
        )

        self.data = data_new
        self.dates = data_new.index

        self.additional_data = (additional_data or {}).copy()

        # Look for data frames with the same index as (original) data,
        # and add in the first row as well (i.e. "bidoffer")
        for k in self.additional_data:
            old = self.additional_data[k]
            if isinstance(old, pd.DataFrame) and old.index.equals(data.index):
                empty_row = pd.DataFrame(
                    np.nan,
                    columns=old.columns,
                    index=[old.index[0] - pd.DateOffset(days=1)],
                )
                new = pd.concat([empty_row, old])
                self.additional_data[k] = new
            elif isinstance(old, pd.Series) and old.index.equals(data.index):
                empty_row = pd.Series(
                    np.nan, index=[old.index[0] - pd.DateOffset(days=1)]
                )
                new = pd.concat([empty_row, old])
                self.additional_data[k] = new# }}}

    def run(self):# {{{
        """
        Runs the Backtest.
        """
        if self.has_run:
            return

        # set run flag to avoid running same test more than once
        self.has_run = True

        # setup strategy
        self.strategy.setup(self.data, **self.additional_data)

        # adjust strategy with initial capital
        self.strategy.adjust(self.initial_capital)

        # loop through dates
        # init progress bar
        if self.progress_bar:
            bar = pyprind.ProgBar(len(self.dates), title=self.name, stream=1)

        # since there is a dummy row at time 0, start backtest at date 1.
        # we must still update for t0
        self.strategy.update(self.dates[0])

        # and for the backtest loop, start at date 1
        for dt in self.dates[1:]:
            # update progress bar
            if self.progress_bar:
                bar.update()

            # update strategy
            self.strategy.update(dt)

            if not self.strategy.bankrupt:
                self.strategy.run()
                # need update after to save weights, values and such
                self.strategy.update(dt)
            else:
                if self.progress_bar:
                    bar.stop()

        self.stats = self.strategy.prices.calc_perf_stats()
        self._original_prices = self.strategy.prices# }}}

    @property
    def weights(self):# {{{
        """
        DataFrame of each component's weight over time
        """
        if self._weights is not None:
            return self._weights
        else:
            if self.strategy.fixed_income:
                vals = pd.DataFrame(
                    {x.full_name: x.notional_values for x in self.strategy.members}
                )
                vals = vals.div(self.strategy.notional_values, axis=0)
            else:
                vals = pd.DataFrame(
                    {x.full_name: x.values for x in self.strategy.members}
                )
                vals = vals.div(self.strategy.values, axis=0)
            self._weights = vals
            return vals# }}}

    @property
    def positions(self):# {{{
        """
        DataFrame of each component's position over time
        """
        return self.strategy.positions# }}}

    @property
    def security_weights(self):# {{{
        """
        DataFrame containing weights of each security as a
        percentage of the whole portfolio over time
        """
        if self._sweights is not None:
            return self._sweights
        else:
            # get values for all securities in tree and divide by root values
            # for security weights
            vals = {}
            for m in self.strategy.members:
                if isinstance(m, bt.core.SecurityBase):
                    if self.strategy.fixed_income:
                        m_values = m.notional_values
                    else:
                        m_values = m.values
                    if m.name in vals:
                        vals[m.name] += m_values
                    else:
                        vals[m.name] = m_values
            vals = pd.DataFrame(vals)

            # divide by root strategy values
            if self.strategy.fixed_income:
                vals = vals.div(self.strategy.notional_values, axis=0)
            else:
                vals = vals.div(self.strategy.values, axis=0)

            # save for future use
            self._sweights = vals

            return vals# }}}

    @property
    def herfindahl_index(self):# {{{
        """
        Calculate Herfindahl-Hirschman Index (HHI) for the portfolio.
        For each given day, HHI is defined as a sum of squared weights of
        securities in a portfolio; and varies from 1/N to 1.
        Value of 1/N would correspond to an equally weighted portfolio and
        value of 1 corresponds to an extreme case when all amount is invested
        in a single asset.

        1 / HHI is often considered as "an effective number of assets" in
        a given portfolio
        """
        w = self.security_weights
        return (w ** 2).sum(axis=1)# }}}

    @property
    def turnover(self):# {{{
        """
        Calculate the turnover for the backtest.

        This function will calculate the turnover for the strategy. Turnover is
        defined as the lesser of positive or negative outlays divided by NAV
        """
        s = self.strategy
        outlays = s.outlays

        # seperate positive and negative outlays, sum them up, and keep min
        outlaysp = outlays[outlays >= 0].fillna(value=0).sum(axis=1)
        outlaysn = np.abs(outlays[outlays < 0].fillna(value=0).sum(axis=1))

        # merge and keep minimum
        min_outlay = pd.DataFrame({"pos": outlaysp, "neg": outlaysn}).sum(axis=1)

        # turnover is defined as min outlay / nav
        mrg = pd.DataFrame({"outlay": min_outlay, "nav": s.values})

        return mrg["outlay"] / mrg["nav"]# }}}


class Result(ffn.GroupStats):
# {{{
    """
    Based on ffn's GroupStats with a few extra helper methods.

    Args:
        * backtests (list): List of backtests

    Attributes:
        * backtest_list (list): List of bactests in the same order as provided
        * backtests (dict): Dict of backtests by name

    """# }}}

    def __init__(self, *backtests):# {{{
        tmp = [pd.DataFrame({x.name: x.strategy.prices}) for x in backtests]
        super(Result, self).__init__(*tmp)
        self.backtest_list = backtests
        self.backtests = {x.name: x for x in backtests}# }}}

    def display_monthly_returns(self, backtest=0):# {{{
        """
        Display monthly returns for a specific backtest.

        Args:
            * backtest (str, int): Backtest. Can be either a index (int) or the
                name (str)

        """
        key = self._get_backtest(backtest)
        key.display_monthly_returns()# }}}

    def get_weights(self, backtest=0, filter=None):# {{{
        """

        :param backtest: (str, int) Backtest can be either a index (int) or the
                name (str)
        :param filter: (list, str) filter columns for specific columns. Filter
                is simply passed as is to DataFrame[filter], so use something
                that makes sense with a DataFrame.
        :return: (pd.DataFrame) DataFrame of weights
        """

        backtest = self._get_backtest(backtest)

        if filter is not None:
            data = backtest.weights[filter]
        else:
            data = backtest.weights

        return data# }}}

    def plot_weights(self, backtest=0, filter=None, show=True, **kwds):# {{{
        """
        Plots the weights of a given backtest over time.

        Args:
            * backtest (str, int): Backtest can be either a index (int) or the
              name (str)
            * filter (list, str): filter columns for specific columns. Filter
              is simply passed as is to DataFrame[filter], so use something
              that makes sense with a DataFrame.
            * show (bool): whether to display the figure or return the object
            * kwds (dict): Keywords passed to plot

        """
        data = self.get_weights(backtest, filter)

        fig = data.plot(**kwds)
        if show:
            fig.show()
        else:
            return fig
        # }}}

    def get_security_weights(self, backtest=0, filter=None):# {{{
        """

        :param backtest: (str, int) Backtest can be either a index (int) or the
                name (str)
        :param filter: (list, str) filter columns for specific columns. Filter
                is simply passed as is to DataFrame[filter], so use something
                that makes sense with a DataFrame.
        :return: (pd.DataFrame) DataFrame of security weights
        """

        backtest = self._get_backtest(backtest)

        if filter is not None:
            data = backtest.security_weights[filter]
        else:
            data = backtest.security_weights

        return data# }}}

    def plot_security_weights(self, backtest=0, filter=None, show=True, **kwds):# {{{
        """
        Plots the security weights of a given backtest over time.

        Args:
            * backtest (str, int): Backtest. Can be either a index (int) or the
                name (str)
            * filter (list, str): filter columns for specific columns. Filter
                is simply passed as is to DataFrame[filter], so use something
                that makes sense with a DataFrame.
            * show (bool): whether to display the figure or return the object
            * kwds (dict): Keywords passed to plot

        """
        data = self.get_security_weights(backtest, filter)

        fig = data.plot(**kwds)
        if show:
            fig.show()
        else:
            return fig
        # }}}

    def plot_histogram(self, backtest=0, show=True, **kwds):# {{{
        """
        Plots the return histogram of a given backtest over time.

        Args:
            * backtest (str, int): Backtest. Can be either a index (int) or the
                name (str)
            * show (bool): whether to display the figure or return the object
            * kwds (dict): Keywords passed to plot_histogram

        """
        key = self._get_backtest(backtest)
        key.plot_histogram(**kwds)# }}}

    def _get_backtest(self, backtest):# {{{
        # based on input order
        if type(backtest) == int:
            return self.backtest_list[backtest]

        # default case assume ok
        return self.backtests[backtest]# }}}

    def get_transactions(self, backtest=0):# {{{
        """
        Helper function that returns the transactions in the following format:

            Date, Security | quantity, price

        The result is a MultiIndex DataFrame.

        Args:
            * strategy_name (str): If none, it will take the first backtest's
              strategy (self.backtest_list[0].name)

        """
        backtest = self._get_backtest(backtest)
        return backtest.strategy.get_transactions()# }}}

    def get_turnover(self, backtest=0):# {{{
        """
        Helper function that returns the transactions in the following format:

            Date, Security | quantity, price

        The result is a MultiIndex DataFrame.

        Args:
            * strategy_name (str): If none, it will take the first backtest's
              strategy (self.backtest_list[0].name)

        """
        backtest = self._get_backtest(backtest)
        return backtest.turnover
        # }}}

    def get_positions(self, backtest=0):# {{{
        """
        Helper function that returns the transactions in the following format:

            Date, Security | quantity, price

        The result is a MultiIndex DataFrame.

        Args:
            * backtest: (str, int) Backtest can be either a index (int) or the
                name (str)

        """
        backtest = self._get_backtest(backtest)
        return backtest.positions
        # }}}

    def get_cash(self, backtest=0):# {{{
        """
        Helper function that returns the transactions in the following format:

            Date, Security | quantity, price

        The result is a MultiIndex DataFrame.

        Args:
            * backtest: (str, int) Backtest can be either a index (int) or the
                name (str)

        """
        backtest = self._get_backtest(backtest)
        return backtest.strategy.cash
        # }}}

    def get_fees(self, backtest=0):# {{{
        """
        Helper function that returns the transactions in the following format:

            Date, Security | quantity, price

        The result is a MultiIndex DataFrame.

        Args:
            * backtest: (str, int) Backtest can be either a index (int) or the
                name (str)

        """
        backtest = self._get_backtest(backtest)
        return backtest.strategy.fees
        # }}}

    def get_flows(self, backtest=0):# {{{
        """
        Helper function that returns the transactions in the following format:

            Date, Security | quantity, price

        The result is a MultiIndex DataFrame.

        Args:
            * backtest: (str, int) Backtest can be either a index (int) or the
                name (str)

        """
        backtest = self._get_backtest(backtest)
        return backtest.strategy.flows
        # }}}

    def get_notional(self, backtest=0):# {{{
        """
        Helper function that returns the transactions in the following format:

            Date, Security | quantity, price

        The result is a MultiIndex DataFrame.

        Args:
            * backtest: (str, int) Backtest can be either a index (int) or the
                name (str)

        """
        backtest = self._get_backtest(backtest)
        return backtest.strategy.notional_values
        # }}}

    def get_nav(self, backtest=0):# {{{
        """
        Helper function that returns the transactions in the following format:

            Date, Security | quantity, price

        The result is a MultiIndex DataFrame.

        Args:
            * backtest: (str, int) Backtest can be either a index (int) or the
                name (str)

        """
        backtest = self._get_backtest(backtest)
        return backtest.strategy.prices
        # }}}

    def get_all(self, backtest=0):# {{{
        wghts = self.get_security_weights(backtest)
        turn = self.get_turnover(backtest)
        turn.name = 'turnover'
        trans = self.get_transactions(backtest)
        pos = self.get_positions(backtest)
        cash = self.get_cash(backtest)
        cash.name = 'cash'
        fees = self.get_fees(backtest)
        fees.name = 'fees'
        flows = self.get_flows(backtest)
        flows.name = 'flows'
        nav = self.get_nav(backtest)
        nav.name = 'nav'
        notional = self.get_notional(backtest)
        notional.name = 'notional_value'

        data = bt.merge(nav, turn, flows, fees, cash, notional)

        return {'data': data, 'transactions': trans, 'positions': pos, 'weights': wghts}
        # }}}

    def plot_result(self, backtest=0):# {{{

        result = self.get_all(backtest)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                subplot_titles=['Equity Curve', 'Drawdown', 'Security Weights', 'Transactions'],
                specs=[[{"secondary_y": True}], 
                    [{"secondary_y": False}],
                        [{"secondary_y": False}],
                        [{"secondary_y": False}]])

        d = result['data']

        wght = result['weights']
        cols = list(wght.columns)
        wght = wght.reset_index()
        wght = pd.melt(wght, id_vars='index', value_vars=cols, value_name='weight', var_name='security')
        nothing = wght.groupby('index')['weight'].sum() == 0
        nothing = nothing.loc[nothing.values==True]
        wght = wght.loc[~wght['index'].isin(nothing.index)]

        wght_plot = px.line(wght, x='index', y='weight', color='security')
        wght_plot = [x for x in wght_plot.select_traces()]


        fig.add_trace(go.Scatter(x=d.index, y=d['nav'], name='NAV', legendgroup='1'), row=1, col=1)
        fig.add_trace(go.Bar(x=d.index, y=d['flows'], name='flows', legendgroup='1'), row=1, col=1, secondary_y=True)
        for i in wght_plot:
            # i.update(legendgroup='2')
            i.update(line_shape='hvh')
            fig.add_trace(i, row=3, col=1)

        fig.add_trace(go.Scatter(x=d.index, y=d['nav'].q.to_drawdown_series(), name='drawdown'),row=2,col=1)

        fig.update_yaxes(title_text='NAV', row=1, col=1, type='log', secondary_y=False)
        fig.update_yaxes(title_text='Flows', row=1, col=1, secondary_y=True, showgrid=False)
        fig.update_yaxes(title_text='Weights', row=3, col=1)
        fig.update_yaxes(title_text='Drawdown', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=4, col=1, type='log')
        # olimutils.crosshair(fig)
        fig.show(config=olimutils.fig_config())# }}}






class RandomBenchmarkResult(Result):
    """# {{{
    RandomBenchmarkResult expands on Result to add methods specific
    to random strategy benchmarking.

    Args:
        * backtests (list): List of backtests

    Attributes:
        * base_name (str): Name of backtest being benchmarked
        * r_stats (Result): Stats for random strategies
        * b_stats (Result): Stats for benchmarked strategy

    """# }}}

    def __init__(self, *backtests):# {{{
        super(RandomBenchmarkResult, self).__init__(*backtests)
        self.base_name = backtests[0].name
        # seperate stats to make
        self.r_stats = self.stats.drop(self.base_name, axis=1)
        self.b_stats = self.stats[self.base_name]# }}}

    def plot_histogram(# {{{
        self, statistic="monthly_sharpe", figsize=(15, 5), title=None, bins=20, **kwargs
    ):
        """
        Plots the distribution of a given statistic. The histogram
        represents the distribution of the random strategies' statistic
        and the vertical line is the value of the benchmarked strategy's
        statistic.

        This helps you determine if your strategy is statistically 'better'
        than the random versions.

        Args:
            * statistic (str): Statistic - any numeric statistic in
              Result is valid.
            * figsize ((x, y)): Figure size
            * title (str): Chart title
            * bins (int): Number of bins
            * kwargs (dict): Passed to pandas hist function.

        """
        if statistic not in self.r_stats.index:
            raise ValueError(
                "Invalid statistic. Valid statistics" "are the statistics in self.stats"
            )

        if title is None:
            title = "%s histogram" % statistic

        plt.figure(figsize=figsize)

        ser = self.r_stats.loc[statistic]

        ax = ser.hist(bins=bins, figsize=figsize, density=True, **kwargs)
        ax.set_title(title)
        plt.axvline(self.b_stats[statistic], linewidth=4, color="r")
        ser.plot(kind="kde")# }}}


class RenormalizedFixedIncomeResult(Result):
    """# {{{
    A new result type to help compare results generated from
    :class:`FixedIncomeStrategy <bt.core.FixedIncomeStrategy>`.
    Recall that in a fixed income strategy, the normalized prices are computed
    using additive returns expressed as a percentage of current outstanding
    notional (i.e. fixed-notional equivalent).
    In strategies where the notional is varying, this may lead to counter-
    intuitive results because the different terms in the sum are being scaled by
    different notionals in the denominator (i.e. price could be below par, but
    overall change in value is positive).

    This class provides a way to "renormalize" the results with a different
    denominator value or series, i.e. using max or average notional exposure,
    or the risk exposure of the strategy.

    Args:
        * normalizing_value: pd.Series, float or dict thereof(by strategy name)
        * backtests (list): List of backtests (i.e. from Result.backtest_list)
    """# }}}

    def __init__(self, normalizing_value, *backtests):# {{{
        for backtest in backtests:
            if not backtest.strategy.fixed_income:
                raise ValueError(
                    "Cannot apply RenormalizedFixedIncomeResult "
                    "because backtest %s is not on a fixed income "
                    "strategy" % backtest.name
                )
        if not isinstance(normalizing_value, dict):
            normalizing_value = {x.name: normalizing_value for x in backtests}
        tmp = [
            pd.DataFrame({x.name: self._price(x.strategy, normalizing_value[x.name])})
            for x in backtests
        ]
        super(Result, self).__init__(*tmp)
        self.backtest_list = backtests
        self.backtests = {x.name: x for x in backtests}# }}}

    def _price(self, s, v):# {{{
        """
        Compute the new price series from the strategy (s) and the
        normalizing value (v)
        """
        # Compute additive returns net of flows
        returns = s.values.diff() - s.flows
        prices = bt.core.PAR * (1.0 + (returns / v).cumsum())
        prices.iloc[0] = bt.core.PAR
        return prices# }}}
