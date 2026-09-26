# %%
import argparse
from datetime import date

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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        help='Ticker symbols separated by spaces'
    )

    parser.add_argument(
        '--max-weight',
        type=float,
        default=100.0,
        help='Maximum weight per asset in percent'
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

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if args.years <= 0:
        parser.error('--years must be greater than zero.')

    if args.max_iter <= 0:
        parser.error('--max-iter must be greater than zero.')

    if args.simulations <= 0:
        parser.error('--simulations must be greater than zero.')

    if not 0 < args.max_weight <= 100:
        parser.error(
            '--max-weight must be greater than 0 '
            'and less than or equal to 100.'
        )

    if len(args.tickers) < 2:
        parser.error('At least two tickers are required.')

    return args


# ============================================================
# Configuration
# ============================================================

def print_configuration(args):

    print()
    print('Selected Configuration')
    print('----------------------')

    print(f'Years:                  {args.years}')
    print(f'Risk-free rate:         {args.rf:.2f} %')
    print(f'Maximum asset weight:   {args.max_weight:.2f} %')
    print(f'Optimizer:              SLSQP')
    print(f'Max optimizer iterations: {args.max_iter:,}')

    print(
        f'Show plot:              '
        f'{"Yes" if args.show_plot else "No"}'
    )

    print(
        f'Plot simulations:       '
        f'{args.simulations:,}'
    )

    print(
        f'Random seed:            '
        f'{args.seed if args.seed is not None else "Random"}'
    )

    print(f'Assets:                 {len(args.tickers)}')

    print()
    print('Tickers:')

    for ticker in args.tickers:
        print(f'  {ticker}')

    print()


# ============================================================
# Portfolio statistics
# ============================================================

def portfolio_return(weights, mu):

    return weights @ mu


def portfolio_variance(weights, covariance):

    return weights @ covariance @ weights


def portfolio_risk(weights, covariance):

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

    if n_assets * max_weight < 1.0 - 1e-12:

        minimum_required = 100 / n_assets

        raise ValueError(
            f'Maximum weight constraint is infeasible. '
            f'With {n_assets} assets, --max-weight must be '
            f'at least {minimum_required:.2f} %.'
        )

    # --------------------------------------------------------
    # Constraints
    #
    # sum(weights) = 1
    # --------------------------------------------------------

    constraints = (
        {
            'type': 'eq',
            'fun': lambda weights:
                np.sum(weights) - 1.0
        },
    )

    # --------------------------------------------------------
    # Bounds
    #
    # 0 <= weight <= max_weight
    # --------------------------------------------------------

    bounds = [
        (0.0, max_weight)
        for _ in range(n_assets)
    ]

    # --------------------------------------------------------
    # Initial guess
    #
    # Equal-weight portfolio
    # --------------------------------------------------------

    w0 = np.ones(n_assets) / n_assets

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

    w_min_var = min_var_result.x

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

    # Try more than one starting point
    starting_points = [
        w0,
        w_min_var
    ]

    # --------------------------------------------------------
    # Add a return-oriented starting point
    # --------------------------------------------------------

    w_return = np.zeros(n_assets)

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
            sharpe_results.append(result)

    if not sharpe_results:

        raise RuntimeError(
            'Maximum Sharpe optimization failed.'
        )

    # Select the best successful optimization
    max_sharpe_result = min(
        sharpe_results,
        key=lambda result: result.fun
    )

    w_max_sharpe = max_sharpe_result.x

    return (
        w_max_sharpe,
        w_min_var,
        max_sharpe_result,
        min_var_result
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
        key=lambda x: x[1],
        reverse=True
    )

    for ticker, weight in portfolio:

        # Avoid displaying "-0.00"
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

    # --------------------------------------------------------
    # If max_weight = 100%, no adjustment is required
    # --------------------------------------------------------

    if max_weight >= 1.0:
        return weights

    # --------------------------------------------------------
    # Enforce maximum weight for visualization portfolios
    #
    # This is used only to generate the plot cloud.
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

    mask = excess_sum > 0

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
    years
):

    n_assets = len(mu)

    # --------------------------------------------------------
    # Generate random portfolios only for visualization
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
    # Optimized portfolios
    # --------------------------------------------------------

    max_sharpe_return = portfolio_return(
        w_max_sharpe,
        mu
    )

    max_sharpe_risk = portfolio_risk(
        w_max_sharpe,
        covariance
    )

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
    # Configuration
    # --------------------------------------------------------

    print_configuration(
        args
    )

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
        f'for {len(args.tickers)} assets...'
    )

    stocks = yf.download(
        tickers=args.tickers,
        start=start,
        end=end,
        auto_adjust=True
    )['Close']

    # Remove assets without data
    stocks = stocks.dropna(
        axis=1,
        how='all'
    )

    # Keep dates common to every asset
    stocks = stocks.dropna()

    if stocks.shape[1] < 2:

        raise RuntimeError(
            'Could not obtain valid data '
            'for at least two assets.'
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
    # Decimal units:
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
    # CLI uses percent:
    # --rf 13 -> 13%
    #
    # Internally:
    # 13% -> 0.13
    # --------------------------------------------------------

    rf = (
        args.rf
        / 100
    )

    # --------------------------------------------------------
    # Maximum asset weight
    #
    # CLI:
    # --max-weight 25 -> 25%
    #
    # Internally:
    # 25% -> 0.25
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
        w_min_var,
        max_sharpe_result,
        min_var_result

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
        key=lambda x: x[1],
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
        * np.sqrt(TRADING_DAYS)
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
        key=lambda x: x[1],
        reverse=True
    )

    for ticker, risk in sorted_risks:

        print(
            f'{ticker}: '
            f'{risk:.2f} %'
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
            years=args.years
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    main()