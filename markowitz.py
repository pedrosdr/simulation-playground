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

DEFAULT_YEARS = 5
DEFAULT_RF = 13.0
DEFAULT_MAX_WEIGHT = 100.0
DEFAULT_MAX_ITER = 2000
DEFAULT_SIMULATIONS = 50_000
DEFAULT_SEED = None
DEFAULT_PLOT = False

TRADING_DAYS = 252


# ============================================================
# Terminal colors
# ============================================================

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


# ============================================================
# JSON configuration
# ============================================================

def load_config(path):

    path = Path(path).expanduser()

    if not path.is_file():
        raise ValueError(
            f'Configuration file not found: {path}'
        )

    try:
        with path.open(
            'r',
            encoding='utf-8'
        ) as file:
            data = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f'Invalid JSON in configuration file '
            f'{path}: {exc}'
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            'Configuration JSON must contain a JSON object.'
        )

    key_map = {
        'years': 'years',
        'rf': 'rf',

        'max-weight': 'max_weight',
        'max_weight': 'max_weight',

        'max-iter': 'max_iter',
        'max_iter': 'max_iter',

        'simulations': 'simulations',

        'seed': 'seed',

        'plot': 'show_plot',
        'show-plot': 'show_plot',
        'show_plot': 'show_plot',

        'tickers': 'tickers'
    }

    config = {}

    for key, value in data.items():

        if key not in key_map:
            raise ValueError(
                f'Unknown configuration parameter: {key}'
            )

        normalized_key = key_map[key]

        if normalized_key in config:
            raise ValueError(
                f'Duplicate configuration parameter: {key}'
            )

        config[normalized_key] = value

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if 'years' in config:

        if (
            isinstance(config['years'], bool)
            or not isinstance(config['years'], int)
            or config['years'] <= 0
        ):
            raise ValueError(
                '"years" must be a positive integer.'
            )

    if 'rf' in config:

        if (
            isinstance(config['rf'], bool)
            or not isinstance(
                config['rf'],
                (int, float)
            )
        ):
            raise ValueError(
                '"rf" must be numeric.'
            )

    if 'max_weight' in config:

        value = config['max_weight']

        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float)
            )
            or not 0 < value <= 100
        ):
            raise ValueError(
                '"max-weight" must be greater than 0 '
                'and at most 100.'
            )

    if 'max_iter' in config:

        if (
            isinstance(config['max_iter'], bool)
            or not isinstance(
                config['max_iter'],
                int
            )
            or config['max_iter'] <= 0
        ):
            raise ValueError(
                '"max-iter" must be a positive integer.'
            )

    if 'simulations' in config:

        if (
            isinstance(config['simulations'], bool)
            or not isinstance(
                config['simulations'],
                int
            )
            or config['simulations'] <= 0
        ):
            raise ValueError(
                '"simulations" must be a positive integer.'
            )

    if 'seed' in config:

        if (
            config['seed'] is not None
            and (
                isinstance(config['seed'], bool)
                or not isinstance(
                    config['seed'],
                    int
                )
            )
        ):
            raise ValueError(
                '"seed" must be an integer or null.'
            )

    if 'show_plot' in config:

        if not isinstance(
            config['show_plot'],
            bool
        ):
            raise ValueError(
                '"plot" must be true or false.'
            )

    return config, path


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():

    # --------------------------------------------------------
    # Parse --config first
    # --------------------------------------------------------

    pre_parser = argparse.ArgumentParser(
        add_help=False
    )

    pre_parser.add_argument(
        '-c',
        '--config',
        type=str,
        default=None
    )

    pre_args, _ = pre_parser.parse_known_args()

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = {}
    config_path = None

    if pre_args.config is not None:

        try:
            config, config_path = load_config(
                pre_args.config
            )

        except ValueError as exc:
            raise SystemExit(
                f'Error: {exc}'
            ) from exc

    # --------------------------------------------------------
    # Main parser
    #
    # Priority:
    # defaults < config JSON < command line
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            'Markowitz portfolio optimization using SciPy.'
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
        epilog=(
            'Examples:\n'
            '  py markowitz.py\n'
            '  py markowitz.py --config portfolio.json\n'
            '  py markowitz.py --config portfolio.json --rf 12\n'
            '  py markowitz.py --tickers PETR4.SA ITUB3.SA VALE3.SA\n'
            '  py markowitz.py --tickers portfolio.json --plot\n'
        )
    )

    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default=pre_args.config,
        help='JSON configuration file'
    )

    parser.add_argument(
        '-y',
        '--years',
        type=int,
        default=config.get(
            'years',
            DEFAULT_YEARS
        ),
        help='Number of years of historical data'
    )

    parser.add_argument(
        '-r',
        '--rf',
        type=float,
        default=config.get(
            'rf',
            DEFAULT_RF
        ),
        help='Annual risk-free rate in percent'
    )

    parser.add_argument(
        '-t',
        '--tickers',
        nargs='+',
        default=None,
        help=(
            'Ticker symbols separated by spaces, '
            'or a path to a ticker JSON file'
        )
    )

    parser.add_argument(
        '--max-weight',
        type=float,
        default=config.get(
            'max_weight',
            DEFAULT_MAX_WEIGHT
        ),
        help='Maximum optimized weight per asset in percent'
    )

    parser.add_argument(
        '--max-iter',
        type=int,
        default=config.get(
            'max_iter',
            DEFAULT_MAX_ITER
        ),
        help='Maximum number of optimizer iterations'
    )

    parser.add_argument(
        '-n',
        '--simulations',
        type=int,
        default=config.get(
            'simulations',
            DEFAULT_SIMULATIONS
        ),
        help=(
            'Number of random portfolios used '
            'for plot visualization'
        )
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=config.get(
            'seed',
            DEFAULT_SEED
        ),
        help='Random seed for plot reproducibility'
    )

    plot_group = (
        parser.add_mutually_exclusive_group()
    )

    plot_group.add_argument(
        '-p',
        '--plot',
        dest='show_plot',
        action='store_true',
        help='Show portfolio risk-return plot'
    )

    plot_group.add_argument(
        '--no-plot',
        dest='show_plot',
        action='store_false',
        help='Disable plot'
    )

    parser.set_defaults(
        show_plot=config.get(
            'show_plot',
            DEFAULT_PLOT
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Ticker source
    # --------------------------------------------------------

    if args.tickers is not None:

        ticker_spec = args.tickers
        ticker_source = 'Command line'

    elif 'tickers' in config:

        ticker_spec = config['tickers']

        ticker_source = (
            f'Configuration file: {config_path}'
        )

    else:

        ticker_spec = DEFAULT_TICKERS
        ticker_source = 'Default tickers'

    return (
        args,
        ticker_spec,
        ticker_source,
        config_path
    )


# ============================================================
# Parse ticker data
# ============================================================

def parse_ticker_data(
    data,
    source
):

    # --------------------------------------------------------
    # List of tickers
    # --------------------------------------------------------

    if isinstance(data, list):

        if len(data) < 2:
            raise ValueError(
                'At least two tickers are required.'
            )

        if not all(
            isinstance(ticker, str)
            and ticker.strip()
            for ticker in data
        ):
            raise ValueError(
                'Every ticker must be a non-empty string.'
            )

        tickers = [
            ticker.strip()
            for ticker in data
        ]

        if len(set(tickers)) != len(tickers):
            raise ValueError(
                'Duplicate tickers are not allowed.'
            )

        return (
            tickers,
            None,
            source
        )

    # --------------------------------------------------------
    # Dictionary:
    # ticker -> current monetary value
    # --------------------------------------------------------

    if isinstance(data, dict):

        if len(data) < 2:
            raise ValueError(
                'At least two tickers are required.'
            )

        values = {}

        for ticker, value in data.items():

            if (
                not isinstance(ticker, str)
                or not ticker.strip()
            ):
                raise ValueError(
                    'Every ticker must be a non-empty string.'
                )

            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float)
                )
            ):
                raise ValueError(
                    f'Value for {ticker} must be numeric.'
                )

            value = float(value)

            if not np.isfinite(value):
                raise ValueError(
                    f'Value for {ticker} must be finite.'
                )

            if value < 0:
                raise ValueError(
                    f'Value for {ticker} cannot be negative.'
                )

            values[ticker.strip()] = value

        if sum(values.values()) <= 0:
            raise ValueError(
                'Provided portfolio values must sum '
                'to a value greater than zero.'
            )

        return (
            list(values.keys()),
            values,
            source
        )

    raise ValueError(
        'Tickers must be either a list '
        'or an object containing ticker:value pairs.'
    )


# ============================================================
# Load standalone ticker JSON
# ============================================================

def load_ticker_json(path):

    path = Path(path).expanduser()

    if not path.is_file():
        raise ValueError(
            f'Ticker JSON file not found: {path}'
        )

    try:

        with path.open(
            'r',
            encoding='utf-8'
        ) as file:

            data = json.load(file)

    except json.JSONDecodeError as exc:

        raise ValueError(
            f'Invalid JSON in {path}: {exc}'
        ) from exc

    return parse_ticker_data(
        data,
        f'Ticker JSON: {path}'
    )


# ============================================================
# Resolve ticker specification
# ============================================================

def resolve_tickers(
    ticker_spec,
    ticker_source
):

    if isinstance(
        ticker_spec,
        (dict, list)
    ):

        # ----------------------------------------------------
        # A single-item list may be a JSON path
        # ----------------------------------------------------

        if (
            isinstance(ticker_spec, list)
            and len(ticker_spec) == 1
            and isinstance(
                ticker_spec[0],
                str
            )
            and Path(
                ticker_spec[0]
            ).suffix.lower() == '.json'
        ):

            return load_ticker_json(
                ticker_spec[0]
            )

        return parse_ticker_data(
            ticker_spec,
            ticker_source
        )

    # --------------------------------------------------------
    # JSON path supplied as a string in configuration
    # --------------------------------------------------------

    if isinstance(
        ticker_spec,
        str
    ):

        path = Path(
            ticker_spec
        ).expanduser()

        if path.suffix.lower() == '.json':

            return load_ticker_json(
                path
            )

    raise ValueError(
        'Invalid ticker specification.'
    )


# ============================================================
# Normalize provided portfolio
# ============================================================

def normalize_provided_weights(
    raw_values,
    columns
):

    if raw_values is None:
        return None

    values = np.array(
        [
            raw_values[ticker]
            for ticker in columns
        ],
        dtype=float
    )

    total = values.sum()

    if total <= 0:
        raise ValueError(
            'Provided portfolio values '
            'must sum to a positive value.'
        )

    return values / total


# ============================================================
# Print configuration
# ============================================================

def print_configuration(
    args,
    tickers,
    ticker_source,
    config_path,
    has_provided_portfolio
):

    print()
    print('Selected Configuration')
    print('----------------------')

    print(
        f'{"Configuration file:":<30}'
        f'{config_path if config_path else "None"}'
    )

    print(
        f'{"Years:":<30}'
        f'{args.years}'
    )

    print(
        f'{"Risk-free rate:":<30}'
        f'{args.rf:.2f} %'
    )

    print(
        f'{"Maximum optimized weight:":<30}'
        f'{args.max_weight:.2f} %'
    )

    print(
        f'{"Optimizer:":<30}'
        f'SLSQP'
    )

    print(
        f'{"Max optimizer iterations:":<30}'
        f'{args.max_iter:,}'
    )

    print(
        f'{"Show plot:":<30}'
        f'{"Yes" if args.show_plot else "No"}'
    )

    print(
        f'{"Plot simulations:":<30}'
        f'{args.simulations:,}'
    )

    print(
        f'{"Random seed:":<30}'
        f'{args.seed if args.seed is not None else "Random"}'
    )

    print(
        f'{"Ticker source:":<30}'
        f'{ticker_source}'
    )

    print(
        f'{"Provided portfolio:":<30}'
        f'{"Yes" if has_provided_portfolio else "No"}'
    )

    print(
        f'{"Assets:":<30}'
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
            f'With {n_assets} assets, '
            f'--max-weight must be at least '
            f'{minimum_required:.2f} %.'
        )

    # --------------------------------------------------------
    # Sum(weights) = 1
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
    # Return-oriented starting point
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
# Standard portfolio output
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

    ticker_width = max(
        len('Ticker'),
        max(
            len(ticker)
            for ticker in tickers
        )
    )

    print(
        f'{"Ticker":<{ticker_width}}  '
        f'{"Weight":<12}'
    )

    print(
        '-' * (
            ticker_width + 14
        )
    )

    for ticker, weight in portfolio:

        weight = max(
            0.0,
            weight
        )

        weight_text = (
            f'{100 * weight:.2f} %'
        )

        print(
            f'{ticker:<{ticker_width}}  '
            f'{weight_text:<12}'
        )

    print()

    print(
        f'{"Expected return:":<20}'
        f'{100 * expected_return:.2f} %'
    )

    print(
        f'{"Risk:":<20}'
        f'{100 * risk:.2f} %'
    )

    print(
        f'{"Sharpe ratio:":<20}'
        f'{sharpe:.3f}'
    )


# ============================================================
# Provided portfolio + rebalancing output
# ============================================================

def print_provided_portfolio(
    title,
    raw_values,
    current_weights,
    target_weights,
    tickers,
    mu,
    covariance,
    rf
):

    # --------------------------------------------------------
    # Current portfolio statistics
    # --------------------------------------------------------

    expected_return = portfolio_return(
        current_weights,
        mu
    )

    risk = portfolio_risk(
        current_weights,
        covariance
    )

    sharpe = portfolio_sharpe(
        current_weights,
        mu,
        covariance,
        rf
    )

    # --------------------------------------------------------
    # Current monetary values
    # --------------------------------------------------------

    current_values = np.array(
        [
            raw_values[ticker]
            for ticker in tickers
        ],
        dtype=float
    )

    total_value = (
        current_values.sum()
    )

    # --------------------------------------------------------
    # Target monetary values
    # --------------------------------------------------------

    target_values = (
        target_weights
        * total_value
    )

    # --------------------------------------------------------
    # Required trades
    # --------------------------------------------------------

    delta_values = (
        target_values
        - current_values
    )

    delta_weights = (
        target_weights
        - current_weights
    )

    # --------------------------------------------------------
    # Rows
    # --------------------------------------------------------

    portfolio = sorted(
        zip(
            tickers,
            current_weights,
            target_weights,
            current_values,
            target_values,
            delta_weights,
            delta_values
        ),
        key=lambda x:
            x[1],
        reverse=True
    )

    # --------------------------------------------------------
    # Column widths
    # --------------------------------------------------------

    ticker_width = max(
        len('Ticker'),
        max(
            len(ticker)
            for ticker in tickers
        )
    )

    current_width = 12
    target_width = 12
    action_width = 8
    amount_width = 14
    delta_width = 12

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print(title)
    print('-' * len(title))

    header = (
        f'{"Ticker":<{ticker_width}}  '
        f'{"Current":<{current_width}}'
        f'{"Target":<{target_width}}'
        f'{"Action":<{action_width}}'
        f'{"Amount":<{amount_width}}'
        f'{"Δ Weight":<{delta_width}}'
    )

    print(header)
    print('-' * len(header))

    # --------------------------------------------------------
    # Trades
    # --------------------------------------------------------

    total_buy = 0.0
    total_sell = 0.0

    for (
        ticker,
        current_weight,
        target_weight,
        current_value,
        target_value,
        delta_weight,
        delta_value

    ) in portfolio:

        tolerance = 1e-8

        if delta_value > tolerance:

            action = 'BUY'
            color = GREEN

            total_buy += delta_value

        elif delta_value < -tolerance:

            action = 'SELL'
            color = RED

            total_sell += abs(
                delta_value
            )

        else:

            action = 'HOLD'
            color = YELLOW

        current_text = (
            f'{100 * current_weight:.2f} %'
        )

        target_text = (
            f'{100 * target_weight:.2f} %'
        )

        amount_text = (
            f'{abs(delta_value):,.2f}'
        )

        delta_text = (
            f'{100 * delta_weight:+.2f} pp'
        )

        print(
            f'{ticker:<{ticker_width}}  '
            f'{current_text:<{current_width}}'
            f'{target_text:<{target_width}}',
            end=''
        )

        print(
            f'{color}'
            f'{action:<{action_width}}'
            f'{RESET}',
            end=''
        )

        print(
            f'{color}'
            f'{amount_text:<{amount_width}}'
            f'{RESET}',
            end=''
        )

        print(
            f'{delta_text:<{delta_width}}'
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()

    print(
        f'{"Portfolio value:":<20}'
        f'{total_value:,.2f}'
    )

    print(
        f'{"Total BUY:":<20}'
        f'{GREEN}'
        f'{total_buy:,.2f}'
        f'{RESET}'
    )

    print(
        f'{"Total SELL:":<20}'
        f'{RED}'
        f'{total_sell:,.2f}'
        f'{RESET}'
    )

    print()

    print(
        f'{"Expected return:":<20}'
        f'{100 * expected_return:.2f} %'
    )

    print(
        f'{"Risk:":<20}'
        f'{100 * risk:.2f} %'
    )

    print(
        f'{"Sharpe ratio:":<20}'
        f'{sharpe:.3f}'
    )


# ============================================================
# Enforce maximum weight
# ============================================================

def enforce_max_weight(
    weights,
    max_weight
):

    if max_weight >= 1.0:
        return weights

    # --------------------------------------------------------
    # Iteratively redistribute weights that exceed
    # the maximum allowed weight.
    # --------------------------------------------------------

    weights = weights.copy()

    for _ in range(100):

        excess = np.maximum(
            weights - max_weight,
            0.0
        )

        excess_sum = excess.sum(
            axis=1
        )

        mask = (
            excess_sum > 1e-12
        )

        if not np.any(mask):
            break

        weights = np.minimum(
            weights,
            max_weight
        )

        capacity = np.maximum(
            max_weight - weights,
            0.0
        )

        capacity_sum = capacity.sum(
            axis=1
        )

        valid = (
            mask
            & (capacity_sum > 1e-12)
        )

        if not np.any(valid):
            break

        weights[valid] += (
            capacity[valid]
            * (
                excess_sum[valid]
                / capacity_sum[valid]
            )[:, None]
        )

    # --------------------------------------------------------
    # Numerical normalization
    # --------------------------------------------------------

    weights /= weights.sum(
        axis=1,
        keepdims=True
    )

    return weights


# ============================================================
# Random portfolios for plot
# ============================================================

def generate_random_weights(
    rng,
    n_portfolios,
    n_assets,
    max_weight
):

    # --------------------------------------------------------
    # Exactly n_portfolios are generated.
    #
    # Each portfolio receives its own Dirichlet concentration
    # parameter.
    #
    # Small alpha -> portfolios closer to edges/corners.
    # Alpha near 1 -> portfolios more toward the interior.
    #
    # Beta(1, 3) favors smaller values, therefore the cloud
    # contains more points near the boundaries while still
    # filling the interior.
    # --------------------------------------------------------

    alpha = (
        0.10
        + 0.90
        * rng.beta(
            1.0,
            3.0,
            size=n_portfolios
        )
    )

    # --------------------------------------------------------
    # Dirichlet with one alpha per portfolio.
    #
    # Dirichlet can be generated through Gamma variables:
    #
    # x_i ~ Gamma(alpha, 1)
    #
    # w_i = x_i / sum(x)
    # --------------------------------------------------------

    gamma_samples = rng.gamma(
        shape=alpha[:, None],
        scale=1.0,
        size=(
            n_portfolios,
            n_assets
        )
    )

    # --------------------------------------------------------
    # Protect against extremely rare numerical underflow
    # --------------------------------------------------------

    row_sums = gamma_samples.sum(
        axis=1,
        keepdims=True
    )

    invalid_rows = (
        row_sums[:, 0] <= 0
    )

    if np.any(invalid_rows):

        gamma_samples[
            invalid_rows
        ] = 1.0

        row_sums = gamma_samples.sum(
            axis=1,
            keepdims=True
        )

    weights = (
        gamma_samples
        / row_sums
    )

    # --------------------------------------------------------
    # Respect max-weight constraint
    # --------------------------------------------------------

    weights = enforce_max_weight(
        weights,
        max_weight
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
    # Exactly "simulations" portfolios
    # --------------------------------------------------------

    weights = generate_random_weights(
        rng=rng,
        n_portfolios=simulations,
        n_assets=n_assets,
        max_weight=max_weight
    )

    # --------------------------------------------------------
    # Portfolio expected returns
    # --------------------------------------------------------

    random_returns = (
        weights @ mu
    )

    # --------------------------------------------------------
    # Portfolio variances
    # --------------------------------------------------------

    random_variances = np.einsum(
        'ij,jk,ik->i',
        weights,
        covariance,
        weights
    )

    random_risks = np.sqrt(
        random_variances
    )

    # --------------------------------------------------------
    # Sharpe ratios
    # --------------------------------------------------------

    random_sharpes = (
        (random_returns - rf)
        / random_risks
    )

    # --------------------------------------------------------
    # Maximum Sharpe Portfolio
    # --------------------------------------------------------

    max_sharpe_return = (
        portfolio_return(
            w_max_sharpe,
            mu
        )
    )

    max_sharpe_risk = (
        portfolio_risk(
            w_max_sharpe,
            covariance
        )
    )

    # --------------------------------------------------------
    # Minimum Variance Portfolio
    # --------------------------------------------------------

    min_var_return = (
        portfolio_return(
            w_min_var,
            mu
        )
    )

    min_var_risk = (
        portfolio_risk(
            w_min_var,
            covariance
        )
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

        provided_return = (
            portfolio_return(
                provided_weights,
                mu
            )
        )

        provided_risk = (
            portfolio_risk(
                provided_weights,
                covariance
            )
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

        # ----------------------------------------------------
        # Direction toward Maximum Sharpe Portfolio
        # ----------------------------------------------------

        plt.annotate(
            '',
            xy=(
                100 * max_sharpe_risk,
                100 * max_sharpe_return
            ),
            xytext=(
                100 * provided_risk,
                100 * provided_return
            ),
            arrowprops={
                'arrowstyle': '->',
                'linestyle': '--'
            }
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

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

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

    (
        args,
        ticker_spec,
        ticker_source,
        config_path

    ) = parse_args()

    # --------------------------------------------------------
    # Resolve ticker input
    # --------------------------------------------------------

    try:

        (
            tickers,
            provided_values_raw,
            ticker_source

        ) = resolve_tickers(
            ticker_spec,
            ticker_source
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
        config_path=config_path,
        has_provided_portfolio=(
            provided_values_raw
            is not None
        )
    )

    # --------------------------------------------------------
    # RNG
    # --------------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    # --------------------------------------------------------
    # Historical period
    # --------------------------------------------------------

    end = date.today()

    start = (
        end
        - relativedelta(
            years=args.years
        )
    )

    # --------------------------------------------------------
    # Download prices
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

    stocks = stocks.dropna(
        axis=1,
        how='all'
    )

    # --------------------------------------------------------
    # Missing tickers
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
    # Preserve ticker order
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
    # Normalize provided portfolio
    # --------------------------------------------------------

    provided_weights = (
        normalize_provided_weights(
            provided_values_raw,
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
    # --------------------------------------------------------

    mu = (
        TRADING_DAYS
        * np.mean(
            r,
            axis=0
        )
    )

    # --------------------------------------------------------
    # Annualized covariance
    # --------------------------------------------------------

    covariance = (
        TRADING_DAYS
        * np.cov(r.T)
    )

    # --------------------------------------------------------
    # Convert percentages to decimal
    # --------------------------------------------------------

    rf = (
        args.rf
        / 100
    )

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

    ticker_width = max(
        len('Ticker'),
        max(
            len(ticker)
            for ticker in stocks.columns
        )
    )

    print(
        f'{"Ticker":<{ticker_width}}  '
        f'{"Expected Return":<18}'
    )

    print(
        '-' * (
            ticker_width + 20
        )
    )

    for ticker, expected_return in individual_returns:

        expected_return_text = (
            f'{expected_return:.2f} %'
        )

        print(
            f'{ticker:<{ticker_width}}  '
            f'{expected_return_text:<18}'
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

    print(
        f'{"Ticker":<{ticker_width}}  '
        f'{"Risk":<12}'
    )

    print(
        '-' * (
            ticker_width + 14
        )
    )

    for ticker, risk in sorted_risks:

        risk_text = (
            f'{risk:.2f} %'
        )

        print(
            f'{ticker:<{ticker_width}}  '
            f'{risk_text:<12}'
        )

    # ========================================================
    # 5. Provided Portfolio + Rebalancing
    # ========================================================

    if provided_weights is not None:

        print_provided_portfolio(
            title=(
                'Provided Portfolio '
                '— Rebalancing to Maximum Sharpe'
            ),
            raw_values=provided_values_raw,
            current_weights=provided_weights,
            target_weights=w_max_sharpe,
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