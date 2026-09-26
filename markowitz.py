# %%
import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize


# ============================================================
# Default configuration
# ============================================================

DEFAULT_TICKERS = [
    'PETR4.SA',
    'VALE3.SA',
    'BBAS3.SA',
    'CMIG4.SA',
    'ITSA4.SA',
    'TAEE11.SA',
    'CPFE3.SA',
    'PSSA3.SA',
    'ITUB3.SA',
    'WEGE3.SA',
    'SAPR11.SA'
]

TRADING_DAYS = 252


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            'Markowitz portfolio optimization using SciPy. '
            'Finds the maximum Sharpe ratio portfolio '
            'and the minimum variance portfolio.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            'Examples:\n'
            '  py markowitz.py\n'
            '  py markowitz.py --tickers PETR4.SA TAEE11.SA ITUB3.SA\n'
            '  py markowitz.py --tickers example_1.json\n'
            '  py markowitz.py --tickers example_2.json --plot\n'
            '  py markowitz.py --years 10 --rf 11.5 '
            '--max-weight 25 --plot'
        )
    )

    parser.add_argument(
        '-y',
        '--years',
        type=int,
        default=5,
        help='Number of years of historical data'
    )

    parser.add_argument(
        '-r',
        '--rf',
        type=float,
        default=13.0,
        help='Annual risk-free rate in percent'
    )

    parser.add_argument(
        '-t',
        '--tickers',
        nargs='+',
        default=DEFAULT_TICKERS,
        help=(
            'Ticker symbols separated by spaces, or a single '
            'path to a JSON file. A JSON list supplies tickers '
            'only; a JSON object supplies ticker:weight pairs '
            'for a provided portfolio.'
        )
    )

    parser.add_argument(
        '--max-weight',
        type=float,
        default=100.0,
        help='Maximum optimized weight per asset in percent'
    )

    parser.add_argument(
        '--max-iter',
        type=int,
        default=2000,
        help='Maximum number of optimizer iterations'
    )

    parser.add_argument(
        '-p',
        '--plot',
        '--show-plot',
        dest='show_plot',
        action='store_true',
        help='Show the portfolio risk-return plot'
    )

    parser.add_argument(
        '-n',
        '--simulations',
        type=int,
        default=50_000,
        help=(
            'Number of random portfolios used only '
            'for plot visualization'
        )
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for plot reproducibility'
    )

    args = parser.parse_args()

    if args.years <= 0:
        parser.error(
            '--years must be greater than zero.'
        )

    if args.max_iter <= 0:
        parser.error(
            '--max-iter must be greater than zero.'
        )

    if args.simulations <= 0:
        parser.error(
            '--simulations must be greater than zero.'
        )

    if not 0 < args.max_weight <= 100:
        parser.error(
            '--max-weight must be greater than 0 '
            'and less than or equal to 100.'
        )

    return args


# ============================================================
# Ticker / JSON input
# ============================================================

def load_ticker_input(ticker_args):
    """
    Returns
    -------
    tickers : list[str]
        Tickers to download.

    provided_weights_raw : dict[str, float] | None
        Raw portfolio weights when a JSON object is supplied.

    source : str
        Description of the ticker source.
    """

    # --------------------------------------------------------
    # A single .json argument is interpreted as a JSON file
    # --------------------------------------------------------

    if len(ticker_args) == 1:

        candidate = Path(
            ticker_args[0]
        ).expanduser()

        if candidate.suffix.lower() == '.json':

            if not candidate.is_file():
                raise ValueError(
                    f'Ticker JSON file not found: {candidate}'
                )

            try:

                with candidate.open(
                    'r',
                    encoding='utf-8'
                ) as file:

                    data = json.load(file)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f'Invalid JSON in {candidate}: {exc}'
                ) from exc

            # =================================================
            # JSON format 1:
            #
            # [
            #     "PETR4.SA",
            #     "TAEE11.SA",
            #     "ITUB3.SA"
            # ]
            # =================================================

            if isinstance(data, list):

                if len(data) < 2:
                    raise ValueError(
                        'Ticker JSON list must contain '
                        'at least two tickers.'
                    )

                if not all(
                    isinstance(ticker, str)
                    and ticker.strip()
                    for ticker in data
                ):

                    raise ValueError(
                        'Every item in the ticker JSON list '
                        'must be a non-empty string.'
                    )

                tickers = [
                    ticker.strip()
                    for ticker in data
                ]

                if len(set(tickers)) != len(tickers):

                    raise ValueError(
                        'Ticker JSON list contains '
                        'duplicate tickers.'
                    )

                return (
                    tickers,
                    None,
                    str(candidate)
                )

            # =================================================
            # JSON format 2:
            #
            # {
            #     "PETR4.SA": 200.3,
            #     "TAEE11.SA": 21.2,
            #     "ITUB3.SA": 55.5
            # }
            # =================================================

            if isinstance(data, dict):

                if len(data) < 2:

                    raise ValueError(
                        'Ticker JSON object must contain '
                        'at least two tickers.'
                    )

                provided_weights_raw = {}

                for ticker, weight in data.items():

                    if (
                        not isinstance(ticker, str)
                        or not ticker.strip()
                    ):

                        raise ValueError(
                            'Every JSON object key must '
                            'be a non-empty ticker string.'
                        )

                    if (
                        isinstance(weight, bool)
                        or not isinstance(
                            weight,
                            (int, float)
                        )
                    ):

                        raise ValueError(
                            f'Weight for {ticker} '
                            'must be numeric.'
                        )

                    weight = float(weight)

                    if not np.isfinite(weight):

                        raise ValueError(
                            f'Weight for {ticker} '
                            'must be finite.'
                        )

                    if weight < 0:

                        raise ValueError(
                            f'Weight for {ticker} '
                            'cannot be negative.'
                        )

                    provided_weights_raw[
                        ticker.strip()
                    ] = weight

                if (
                    sum(
                        provided_weights_raw.values()
                    )
                    <= 0
                ):

                    raise ValueError(
                        'The sum of provided portfolio '
                        'weights must be greater than zero.'
                    )

                tickers = list(
                    provided_weights_raw.keys()
                )

                return (
                    tickers,
                    provided_weights_raw,
                    str(candidate)
                )

            raise ValueError(
                'Ticker JSON must be either a list '
                'of ticker strings or an object '
                'containing ticker:weight pairs.'
            )

    # --------------------------------------------------------
    # Normal command-line tickers
    # --------------------------------------------------------

    tickers = [
        ticker.strip()
        for ticker in ticker_args
    ]

    if len(tickers) < 2:

        raise ValueError(
            'At least two tickers are required.'
        )

    if len(set(tickers)) != len(tickers):

        raise ValueError(
            'Duplicate tickers are not allowed.'
        )

    return (
        tickers,
        None,
        'Command line'
    )


# ============================================================
# Normalize provided portfolio
# ============================================================

def normalize_provided_weights(
    provided_weights_raw,
    columns
):

    if provided_weights_raw is None:
        return None

    raw = np.array(
        [
            provided_weights_raw[ticker]
            for ticker in columns
        ],
        dtype=float
    )

    total = raw.sum()

    if total <= 0:

        raise ValueError(
            'The sum of provided portfolio '
            'weights must be greater than zero.'
        )

    return raw / total


# ============================================================
# Configuration output
# ============================================================

def print_configuration(
    args,
    tickers,
    ticker_source,
    has_provided_portfolio
):

    print()
    print('Selected Configuration')
    print('----------------------')

    print(
        f'Years:                    '
        f'{args.years}'
    )

    print(
        f'Risk-free rate:           '
        f'{args.rf:.2f} %'
    )

    print(
        f'Maximum optimized weight: '
        f'{args.max_weight:.2f} %'
    )

    print(
        f'Optimizer:                '
        f'SLSQP'
    )

    print(
        f'Max optimizer iterations: '
        f'{args.max_iter:,}'
    )

    print(
        f'Show plot:                '
        f'{"Yes" if args.show_plot else "No"}'
    )

    print(
        f'Plot simulations:         '
        f'{args.simulations:,}'
    )

    print(
        f'Random seed:              '
        f'{args.seed if args.seed is not None else "Random"}'
    )

    print(
        f'Ticker source:            '
        f'{ticker_source}'
    )

    print(
        f'Provided portfolio:       '
        f'{"Yes" if has_provided_portfolio else "No"}'
    )

    print(
        f'Assets:                   '
        f'{len(tickers)}'
    )

    print()
    print('Tickers:')

    for ticker in tickers:
        print(f'  {ticker}')

    print()


# ============================================================
# Portfolio statistics
# ============================================================

def portfolio_return(
    weights,
    mu
):

    return weights @ mu


def portfolio_variance(
    weights,
    covariance
):

    return (
        weights
        @ covariance
        @ weights
    )


def portfolio_risk(
    weights,
    covariance
):

    return np.sqrt(
        portfolio_variance(
            weights,
            covariance
        )
    )


def portfolio_sharpe(
    weights,
    mu,
    covariance,
    rf
):

    risk = portfolio_risk(
        weights,
        covariance
    )

    if risk <= 0:
        return -np.inf

    expected_return = portfolio_return(
        weights,
        mu
    )

    return (
        (expected_return - rf)
        / risk
    )


# ============================================================
# Optimization
# ============================================================

def optimize_portfolios(
    mu,
    covariance,
    rf,
    max_weight,
    max_iter
):

    n_assets = len(mu)

    # --------------------------------------------------------
    # Feasibility check
    # --------------------------------------------------------

    if (
        n_assets * max_weight
        < 1.0 - 1e-12
    ):

        minimum_required = (
            100 / n_assets
        )

        raise ValueError(
            'Maximum weight constraint is infeasible. '
            f'With {n_assets} assets, --max-weight '
            f'must be at least '
            f'{minimum_required:.2f} %.'
        )

    # --------------------------------------------------------
    # Sum of weights = 1
    # --------------------------------------------------------

    constraints = (
        {
            'type': 'eq',
            'fun': lambda weights:
                np.sum(weights) - 1.0
        },
    )

    # --------------------------------------------------------
    # 0 <= weight <= max_weight
    # --------------------------------------------------------

    bounds = [
        (0.0, max_weight)
        for _ in range(n_assets)
    ]

    # --------------------------------------------------------
    # Initial equal-weight portfolio
    # --------------------------------------------------------

    w0 = (
        np.ones(n_assets)
        / n_assets
    )

    options = {
        'maxiter': max_iter,
        'ftol': 1e-12,
        'disp': False
    }

    # ========================================================
    # Minimum Variance Portfolio
    # ========================================================

    min_var_result = minimize(

        fun=lambda weights:
            portfolio_variance(
                weights,
                covariance
            ),

        x0=w0,

        method='SLSQP',

        bounds=bounds,

        constraints=constraints,

        options=options
    )

    if not min_var_result.success:

        raise RuntimeError(
            'Minimum variance optimization failed: '
            f'{min_var_result.message}'
        )

    w_min_var = (
        min_var_result.x
    )

    # ========================================================
    # Maximum Sharpe Ratio Portfolio
    # ========================================================

    def negative_sharpe(weights):

        return -portfolio_sharpe(
            weights,
            mu,
            covariance,
            rf
        )

    # --------------------------------------------------------
    # Multiple starting points
    # --------------------------------------------------------

    starting_points = [
        w0,
        w_min_var
    ]

    # --------------------------------------------------------
    # Return-oriented initial portfolio
    # --------------------------------------------------------

    w_return = np.zeros(
        n_assets
    )

    remaining_weight = 1.0

    for idx in np.argsort(mu)[::-1]:

        allocation = min(
            max_weight,
            remaining_weight
        )

        w_return[idx] = allocation

        remaining_weight -= allocation

        if remaining_weight <= 1e-12:
            break

    starting_points.append(
        w_return
    )

    sharpe_results = []

    for starting_point in starting_points:

        result = minimize(

            fun=negative_sharpe,

            x0=starting_point,

            method='SLSQP',

            bounds=bounds,

            constraints=constraints,

            options=options
        )

        if result.success:

            sharpe_results.append(
                result
            )

    if not sharpe_results:

        raise RuntimeError(
            'Maximum Sharpe optimization failed.'
        )

    max_sharpe_result = min(
        sharpe_results,
        key=lambda result:
            result.fun
    )

    w_max_sharpe = (
        max_sharpe_result.x
    )

    return (
        w_max_sharpe,
        w_min_var
    )


# ============================================================
# Portfolio output
# ============================================================

def print_portfolio(
    title,
    weights,
    tickers,
    mu,
    covariance,
    rf
):

    expected_return = portfolio_return(
        weights,
        mu
    )

    risk = portfolio_risk(
        weights,
        covariance
    )

    sharpe = portfolio_sharpe(
        weights,
        mu,
        covariance,
        rf
    )

    print()
    print(title)
    print('-' * len(title))

    portfolio = sorted(

        zip(
            tickers,
            weights
        ),

        key=lambda x:
            x[1],

        reverse=True
    )

    for ticker, weight in portfolio:

        # Avoid "-0.00 %"
        weight = max(
            0.0,
            weight
        )

        print(
            f'{ticker}: '
            f'{100 * weight:.2f} %'
        )

    print()

    print(
        f'Expected return: '
        f'{100 * expected_return:.2f} %'
    )

    print(
        f'Risk:            '
        f'{100 * risk:.2f} %'
    )

    print(
        f'Sharpe ratio:    '
        f'{sharpe:.3f}'
    )


# ============================================================
# Random portfolios for plot
# ============================================================

def generate_random_weights(
    rng,
    n_portfolios,
    n_assets,
    max_weight
):

    weights = rng.dirichlet(
        np.ones(n_assets),
        size=n_portfolios
    )

    if max_weight >= 1.0:
        return weights

    # --------------------------------------------------------
    # Clip portfolios that violate max_weight
    # and redistribute the excess.
    #
    # Used only for plot visualization.
    # --------------------------------------------------------

    excess = np.maximum(
        weights - max_weight,
        0.0
    )

    excess_sum = np.sum(
        excess,
        axis=1
    )

    weights = np.minimum(
        weights,
        max_weight
    )

    capacity = np.maximum(
        max_weight - weights,
        0.0
    )

    capacity_sum = np.sum(
        capacity,
        axis=1
    )

    mask = (
        excess_sum > 0
    )

    weights[mask] += (

        capacity[mask]

        * (
            excess_sum[mask]
            / capacity_sum[mask]
        )[:, None]
    )

    return weights


# ============================================================
# Plot
# ============================================================

def plot_portfolios(
    rng,
    simulations,
    mu,
    covariance,
    rf,
    w_max_sharpe,
    w_min_var,
    max_weight,
    years,
    provided_weights=None
):

    n_assets = len(mu)

    # --------------------------------------------------------
    # Random portfolios for visualization
    # --------------------------------------------------------

    weights = generate_random_weights(

        rng=rng,

        n_portfolios=simulations,

        n_assets=n_assets,

        max_weight=max_weight
    )

    random_returns = (
        weights @ mu
    )

    random_variances = np.einsum(
        'ij,jk,ik->i',
        weights,
        covariance,
        weights
    )

    random_risks = np.sqrt(
        random_variances
    )

    random_sharpes = (
        (random_returns - rf)
        / random_risks
    )

    # --------------------------------------------------------
    # Maximum Sharpe portfolio
    # --------------------------------------------------------

    max_sharpe_return = portfolio_return(
        w_max_sharpe,
        mu
    )

    max_sharpe_risk = portfolio_risk(
        w_max_sharpe,
        covariance
    )

    # --------------------------------------------------------
    # Minimum Variance portfolio
    # --------------------------------------------------------

    min_var_return = portfolio_return(
        w_min_var,
        mu
    )

    min_var_risk = portfolio_risk(
        w_min_var,
        covariance
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    scatter = plt.scatter(

        100 * random_risks,

        100 * random_returns,

        c=random_sharpes,

        s=5,

        alpha=0.35
    )

    plt.colorbar(
        scatter,
        label='Sharpe Ratio'
    )

    # --------------------------------------------------------
    # Maximum Sharpe Portfolio
    # --------------------------------------------------------

    plt.scatter(

        100 * max_sharpe_risk,

        100 * max_sharpe_return,

        color='red',

        s=130,

        label='Maximum Sharpe',

        zorder=3
    )

    # --------------------------------------------------------
    # Minimum Variance Portfolio
    # --------------------------------------------------------

    plt.scatter(

        100 * min_var_risk,

        100 * min_var_return,

        color='blue',

        s=130,

        label='Minimum Variance',

        zorder=3
    )

    # --------------------------------------------------------
    # Provided Portfolio
    # --------------------------------------------------------

    if provided_weights is not None:

        provided_return = portfolio_return(
            provided_weights,
            mu
        )

        provided_risk = portfolio_risk(
            provided_weights,
            covariance
        )

        plt.scatter(

            100 * provided_risk,

            100 * provided_return,

            color='green',

            marker='D',

            s=140,

            label='Provided Portfolio',

            zorder=4
        )

    # --------------------------------------------------------
    # Capital Allocation Line
    # --------------------------------------------------------

    plt.plot(

        [
            0,
            100 * max_sharpe_risk
        ],

        [
            100 * rf,
            100 * max_sharpe_return
        ],

        color='red',

        linestyle='dotted'
    )

    plt.xlabel(
        'Annualized Risk (%)'
    )

    plt.ylabel(
        'Annualized Expected Return (%)'
    )

    plt.title(
        'Markowitz Portfolio Optimization '
        f'— {years} Years'
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Load tickers / optional provided portfolio
    # --------------------------------------------------------

    try:

        (
            tickers,
            provided_weights_raw,
            ticker_source

        ) = load_ticker_input(
            args.tickers
        )

    except ValueError as exc:

        raise SystemExit(
            f'Error: {exc}'
        ) from exc

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print_configuration(

        args=args,

        tickers=tickers,

        ticker_source=ticker_source,

        has_provided_portfolio=(
            provided_weights_raw
            is not None
        )
    )

    # --------------------------------------------------------
    # Random number generator
    # --------------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    end = date.today()

    start = (
        end
        - relativedelta(
            years=args.years
        )
    )

    # --------------------------------------------------------
    # Download data
    # --------------------------------------------------------

    print(
        f'Downloading {args.years} years of data '
        f'for {len(tickers)} assets...'
    )

    stocks = yf.download(

        tickers=tickers,

        start=start,

        end=end,

        auto_adjust=True

    )['Close']

    # --------------------------------------------------------
    # Remove completely unavailable columns
    # --------------------------------------------------------

    stocks = stocks.dropna(
        axis=1,
        how='all'
    )

    # --------------------------------------------------------
    # Check missing tickers
    # --------------------------------------------------------

    missing_tickers = [

        ticker

        for ticker in tickers

        if ticker not in stocks.columns
    ]

    if missing_tickers:

        raise RuntimeError(
            'No valid price data was returned for: '
            + ', '.join(
                missing_tickers
            )
        )

    # --------------------------------------------------------
    # Preserve requested / JSON ticker order
    # --------------------------------------------------------

    stocks = (
        stocks[tickers]
        .dropna()
    )

    if stocks.shape[1] < 2:

        raise RuntimeError(
            'Could not obtain valid data '
            'for at least two assets.'
        )

    # --------------------------------------------------------
    # Normalize supplied portfolio weights
    # --------------------------------------------------------

    provided_weights = (
        normalize_provided_weights(

            provided_weights_raw,

            stocks.columns
        )
    )

    # --------------------------------------------------------
    # Daily simple returns
    # --------------------------------------------------------

    r = (
        stocks
        .pct_change()
        .dropna()
        .values
    )

    # --------------------------------------------------------
    # Annualized expected returns
    #
    # Internally:
    # 0.20 = 20%
    # --------------------------------------------------------

    mu = (
        TRADING_DAYS
        * np.mean(
            r,
            axis=0
        )
    )

    # --------------------------------------------------------
    # Annualized covariance matrix
    # --------------------------------------------------------

    covariance = (
        TRADING_DAYS
        * np.cov(r.T)
    )

    # --------------------------------------------------------
    # Risk-free rate
    #
    # CLI:
    # --rf 13
    #
    # Internal:
    # 0.13
    # --------------------------------------------------------

    rf = (
        args.rf
        / 100
    )

    # --------------------------------------------------------
    # Maximum optimized weight
    #
    # CLI:
    # --max-weight 25
    #
    # Internal:
    # 0.25
    # --------------------------------------------------------

    max_weight = (
        args.max_weight
        / 100
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    print()
    print('Optimizing portfolios...')

    (
        w_max_sharpe,
        w_min_var

    ) = optimize_portfolios(

        mu=mu,

        covariance=covariance,

        rf=rf,

        max_weight=max_weight,

        max_iter=args.max_iter
    )

    # ========================================================
    # 1. Maximum Sharpe Ratio Portfolio
    # ========================================================

    print_portfolio(

        title='Maximum Sharpe Ratio Portfolio',

        weights=w_max_sharpe,

        tickers=stocks.columns,

        mu=mu,

        covariance=covariance,

        rf=rf
    )

    # ========================================================
    # 2. Minimum Variance Portfolio
    # ========================================================

    print_portfolio(

        title='Minimum Variance Portfolio',

        weights=w_min_var,

        tickers=stocks.columns,

        mu=mu,

        covariance=covariance,

        rf=rf
    )

    # ========================================================
    # 3. Individual Asset Expected Returns
    # ========================================================

    print()
    print('Individual Asset Expected Returns')
    print('---------------------------------')

    individual_returns = sorted(

        zip(
            stocks.columns,
            100 * mu
        ),

        key=lambda x:
            x[1],

        reverse=True
    )

    for ticker, expected_return in individual_returns:

        print(
            f'{ticker}: '
            f'{expected_return:.2f} %'
        )

    # ========================================================
    # 4. Individual Asset Risks
    # ========================================================

    print()
    print('Individual Asset Risks')
    print('----------------------')

    individual_risks = (

        100

        * np.sqrt(
            TRADING_DAYS
        )

        * np.std(
            r,
            axis=0,
            ddof=1
        )
    )

    sorted_risks = sorted(

        zip(
            stocks.columns,
            individual_risks
        ),

        key=lambda x:
            x[1],

        reverse=True
    )

    for ticker, risk in sorted_risks:

        print(
            f'{ticker}: '
            f'{risk:.2f} %'
        )

    # ========================================================
    # 5. Provided Portfolio
    # ========================================================

    if provided_weights is not None:

        print_portfolio(

            title=(
                'Provided Portfolio '
                '— Normalized Weights'
            ),

            weights=provided_weights,

            tickers=stocks.columns,

            mu=mu,

            covariance=covariance,

            rf=rf
        )

    # ========================================================
    # Plot
    # ========================================================

    if args.show_plot:

        plot_portfolios(

            rng=rng,

            simulations=args.simulations,

            mu=mu,

            covariance=covariance,

            rf=rf,

            w_max_sharpe=w_max_sharpe,

            w_min_var=w_min_var,

            max_weight=max_weight,

            years=args.years,

            provided_weights=provided_weights
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    main()