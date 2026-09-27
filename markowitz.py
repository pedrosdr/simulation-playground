# %%
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from dateutil.relativedelta import relativedelta
from matplotlib.lines import Line2D
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
    'SAPR11.SA',
]

DEFAULT_YEARS = 5
DEFAULT_RF = 13.0
DEFAULT_MIN_WEIGHT = 0.0
DEFAULT_MAX_WEIGHT = 100.0
DEFAULT_MAX_ITER = 2000
DEFAULT_SIMULATIONS = 50_000
DEFAULT_SEED = None
DEFAULT_PLOT = False

TRADING_DAYS = 252
OUTPUT_WIDTH = 72

HISTORY_MIN_OBSERVATION_COVERAGE = 0.95
HISTORY_START_TOLERANCE_DAYS = 14


# ============================================================
# Terminal colors
# ============================================================

BOLD = '\033[1m'
DIM = '\033[2m'
CYAN = '\033[96m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'


# ============================================================
# Contextual help
# ============================================================

TICKERS_HELP = """
--tickers / -t
===============

Specifies the assets used by the portfolio analysis.

Accepted formats:


1. Tickers directly on the command line
----------------------------------------

    py markowitz.py -t AAPL MSFT KO JNJ

or:

    py markowitz.py --tickers AAPL MSFT KO JNJ


2. JSON containing a ticker list
---------------------------------

Example: tickers.json

[
    "AAPL",
    "MSFT",
    "KO",
    "JNJ"
]

Run:

    py markowitz.py -t tickers.json


3. JSON containing a current portfolio
---------------------------------------

Example: portfolio.json

{
    "AAPL": 3200.00,
    "MSFT": 2450.00,
    "KO": 1350.00,
    "JNJ": 900.00
}

Run:

    py markowitz.py -t portfolio.json


Portfolio values
----------------

When ticker:value pairs are supplied, values are interpreted as the
current monetary value of each position.

They do not need to sum to 1 or 100.

The program automatically normalizes them into portfolio weights.


Provided portfolio analysis
---------------------------

When monetary values are supplied, the program also calculates:

    - current portfolio weights;
    - expected return;
    - portfolio risk;
    - Sharpe ratio;
    - Maximum Sharpe target weights;
    - BUY / SELL / HOLD instructions;
    - monetary amount to buy or sell;
    - weight change in percentage points.


Validation
----------

    - At least two tickers are required.
    - Duplicate tickers are not allowed.
    - Position values must be numeric.
    - Position values cannot be negative.
    - Total portfolio value must be greater than zero.
"""


CONFIG_HELP = """
--config / -c
=============

Loads parameters from a JSON configuration file.

Run:

    py markowitz.py -c config.json

or:

    py markowitz.py --config config.json


Precedence
----------

    defaults < configuration JSON < command line

Command-line arguments override configuration values.


Supported keys
--------------

years
    Number of years of historical data.

rf
    Annual risk-free rate in percent.

min-weight
min_weight
    Minimum required weight per asset in optimized and simulated
    portfolios, expressed in percent.

max-weight
max_weight
    Maximum allowed weight per asset in optimized and simulated
    portfolios, expressed in percent.

max-iter
max_iter
    Maximum number of SLSQP optimizer iterations.

simulations
    Number of simulated portfolios displayed in the graph.

seed
    Random seed used for plot reproducibility.

plot
show-plot
show_plot
    Enables or disables the graph.

tickers
    Can contain:

        - a ticker list;
        - ticker:value pairs representing a current portfolio;
        - a path to another ticker JSON file.


Complete example
----------------

{
    "years": 6,
    "rf": 10.60,
    "min-weight": 2.0,
    "max-weight": 40.0,
    "max-iter": 2500,
    "simulations": 50000,
    "seed": 31415,
    "plot": true,
    "tickers": {
        "AAPL": 3200.00,
        "MSFT": 2450.00,
        "KO": 1350.00,
        "JNJ": 900.00
    }
}


Configuration with ticker list
------------------------------

{
    "years": 4,
    "rf": 9.80,
    "min-weight": 1.5,
    "max-weight": 35.0,
    "plot": true,
    "tickers": [
        "AAPL",
        "MSFT",
        "KO",
        "JNJ"
    ]
}


Configuration referencing another ticker JSON
---------------------------------------------

{
    "years": 8,
    "rf": 10.25,
    "min-weight": 3.0,
    "max-weight": 30.0,
    "tickers": "tickers.json"
}
"""


YEARS_HELP = """
--years / -y
============

Number of years of historical market data.

Example:

    py markowitz.py --years 7

The value must be a positive integer.

If one or more assets do not have enough historical observations to
cover the requested period, the program prints a red warning before
aligning the series.

Default:

    5
"""


RF_HELP = """
--rf / -r
=========

Annual risk-free rate expressed in percent.

Example:

    py markowitz.py --rf 10.25

means:

    10.25 %

Internally:

    0.1025

Default:

    13.0 %
"""


MIN_WEIGHT_HELP = """
--min-weight
============

Minimum percentage required for every asset in optimized and simulated
portfolios.

Example:

    py markowitz.py --min-weight 2

means:

    weight_i >= 2 %

for every asset.

The constraint applies to:

    - Maximum Sharpe portfolio;
    - Minimum Variance portfolio;
    - simulated portfolios.

It does not modify the current Provided Portfolio.

The constraint must be feasible.

For N assets:

    N * min_weight <= 100 %

For example, with 10 assets:

    min-weight cannot be greater than 10 %

Default:

    0.0 %
"""


MAX_WEIGHT_HELP = """
--max-weight
============

Maximum percentage allowed for one asset in optimized and simulated
portfolios.

Example:

    py markowitz.py --max-weight 35

means:

    weight_i <= 35 %

The constraint applies to:

    - Maximum Sharpe portfolio;
    - Minimum Variance portfolio;
    - simulated portfolios.

It does not modify the current Provided Portfolio.

The constraint must be feasible.

For N assets:

    N * max_weight >= 100 %

Default:

    100.0 %
"""


MAX_ITER_HELP = """
--max-iter
==========

Maximum number of iterations allowed for SLSQP.

Example:

    py markowitz.py --max-iter 3000

Default:

    2000
"""


SIMULATIONS_HELP = """
--simulations / -n
==================

Number of random portfolios used in the risk-return graph.

Example:

    py markowitz.py --simulations 50000

This generates exactly:

    50,000 simulated portfolios.

The simulated portfolios are displayed as points and colored according
to their Sharpe ratio.

The simulation is used only for visualization.

Maximum Sharpe and Minimum Variance portfolios are calculated
independently using SLSQP.

The random-weight distribution favors lower concentration parameters,
producing more points near the boundaries of the feasible portfolio
region while still retaining interior points.

Both --min-weight and --max-weight are respected.

Default:

    50,000
"""


SEED_HELP = """
--seed
======

Random seed used by the portfolio simulation.

Example:

    py markowitz.py --seed 12345

The same seed produces the same simulated portfolio cloud.

The seed does not affect SLSQP optimization.
"""


PLOT_HELP = """
--plot / -p
===========

Displays the portfolio risk-return graph.

The graph contains:

    - simulated portfolios;
    - Sharpe ratio color scale;
    - Maximum Sharpe portfolio;
    - Minimum Variance portfolio;
    - Provided Portfolio, when supplied;
    - direction toward Maximum Sharpe;
    - Capital Allocation Line.

Example:

    py markowitz.py --plot
"""


NO_PLOT_HELP = """
--no-plot
=========

Explicitly disables the graph.

Useful when the configuration contains:

    "plot": true

Example:

    py markowitz.py -c config.json --no-plot
"""


ARGUMENT_HELP = {
    '-t': TICKERS_HELP,
    '--tickers': TICKERS_HELP,
    '-c': CONFIG_HELP,
    '--config': CONFIG_HELP,
    '-y': YEARS_HELP,
    '--years': YEARS_HELP,
    '-r': RF_HELP,
    '--rf': RF_HELP,
    '--min-weight': MIN_WEIGHT_HELP,
    '--max-weight': MAX_WEIGHT_HELP,
    '--max-iter': MAX_ITER_HELP,
    '-n': SIMULATIONS_HELP,
    '--simulations': SIMULATIONS_HELP,
    '--seed': SEED_HELP,
    '-p': PLOT_HELP,
    '--plot': PLOT_HELP,
    '--no-plot': NO_PLOT_HELP,
}


# ============================================================
# Help and formatting
# ============================================================

def handle_contextual_help():
    args = sys.argv[1:]
    help_flags = {'-h', '--help'}

    for index, argument in enumerate(args):
        if argument not in ARGUMENT_HELP:
            continue

        if index + 1 < len(args) and args[index + 1] in help_flags:
            print(ARGUMENT_HELP[argument].strip())
            raise SystemExit(0)


def format_percent(value, decimals=2):
    return f'{100 * value:.{decimals}f} %'


def format_percent_value(value, decimals=2):
    return f'{value:.{decimals}f} %'


def format_weight_delta(value, decimals=2):
    return f'{100 * value:+.{decimals}f} pp'


def format_money(value):
    return f'{value:,.2f}'


def print_section_title(title):
    title = title.upper()
    print()
    print(f'{BOLD}{CYAN}{"=" * OUTPUT_WIDTH}{RESET}')
    print(f'{BOLD}{CYAN}{title:^{OUTPUT_WIDTH}}{RESET}')
    print(f'{BOLD}{CYAN}{"=" * OUTPUT_WIDTH}{RESET}')


def print_metrics(expected_return, risk, sharpe):
    return_text = format_percent(expected_return)
    risk_text = format_percent(risk)
    sharpe_text = f'{sharpe:.3f}'
    column_width = OUTPUT_WIDTH // 3

    print()
    print('-' * OUTPUT_WIDTH)
    print(f'{"RETURN":<{column_width}}{"RISK":<{column_width}}{"SHARPE":<{column_width}}')
    print(f'{return_text:<{column_width}}{risk_text:<{column_width}}{sharpe_text:<{column_width}}')
    print('-' * OUTPUT_WIDTH)


def print_status(message):
    print(f'{CYAN}{message}{RESET}')


# ============================================================
# Configuration
# ============================================================

def load_config(path):
    path = Path(path).expanduser()

    if not path.is_file():
        raise ValueError(f'Configuration file not found: {path}')

    try:
        with path.open('r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON in configuration file {path}: {exc}') from exc

    if not isinstance(data, dict):
        raise ValueError('Configuration JSON must contain a JSON object.')

    key_map = {
        'years': 'years',
        'rf': 'rf',
        'min-weight': 'min_weight',
        'min_weight': 'min_weight',
        'max-weight': 'max_weight',
        'max_weight': 'max_weight',
        'max-iter': 'max_iter',
        'max_iter': 'max_iter',
        'simulations': 'simulations',
        'seed': 'seed',
        'plot': 'show_plot',
        'show-plot': 'show_plot',
        'show_plot': 'show_plot',
        'tickers': 'tickers',
    }

    config = {}

    for key, value in data.items():
        if key not in key_map:
            raise ValueError(f'Unknown configuration parameter: {key}')

        normalized_key = key_map[key]
        if normalized_key in config:
            raise ValueError(f'Duplicate configuration parameter: {key}')

        config[normalized_key] = value

    if 'years' in config:
        value = config['years']
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError('"years" must be a positive integer.')

    if 'rf' in config:
        value = config['rf']
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('"rf" must be numeric.')

    if 'min_weight' in config:
        value = config['min_weight']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 100:
            raise ValueError('"min-weight" must be between 0 and 100.')

    if 'max_weight' in config:
        value = config['max_weight']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 100:
            raise ValueError('"max-weight" must be greater than 0 and at most 100.')

    min_weight = config.get('min_weight', DEFAULT_MIN_WEIGHT)
    max_weight = config.get('max_weight', DEFAULT_MAX_WEIGHT)
    if min_weight > max_weight:
        raise ValueError('"min-weight" cannot be greater than "max-weight".')

    if 'max_iter' in config:
        value = config['max_iter']
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError('"max-iter" must be a positive integer.')

    if 'simulations' in config:
        value = config['simulations']
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError('"simulations" must be a positive integer.')

    if 'seed' in config:
        value = config['seed']
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError('"seed" must be an integer or null.')

    if 'show_plot' in config and not isinstance(config['show_plot'], bool):
        raise ValueError('"plot" must be true or false.')

    return config, path


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('-c', '--config', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    config = {}
    config_path = None

    if pre_args.config is not None:
        try:
            config, config_path = load_config(pre_args.config)
        except ValueError as exc:
            raise SystemExit(f'Error: {exc}') from exc

    parser = argparse.ArgumentParser(
        description='Markowitz portfolio optimization using SciPy.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '-c', '--config', type=str, default=pre_args.config,
        help='JSON configuration file. Use "-c --help" for details.',
    )
    parser.add_argument(
        '-y', '--years', type=int, default=config.get('years', DEFAULT_YEARS),
        help='Historical period in years. Use "-y --help" for details.',
    )
    parser.add_argument(
        '-r', '--rf', type=float, default=config.get('rf', DEFAULT_RF),
        help='Annual risk-free rate in percent. Use "-r --help" for details.',
    )
    parser.add_argument(
        '-t', '--tickers', nargs='+', default=None,
        help='Tickers or ticker/portfolio JSON file. Use "-t --help" for details.',
    )
    parser.add_argument(
        '--min-weight', type=float, default=config.get('min_weight', DEFAULT_MIN_WEIGHT),
        help='Minimum optimized asset weight in percent. Use "--min-weight --help" for details.',
    )
    parser.add_argument(
        '--max-weight', type=float, default=config.get('max_weight', DEFAULT_MAX_WEIGHT),
        help='Maximum optimized asset weight in percent. Use "--max-weight --help" for details.',
    )
    parser.add_argument(
        '--max-iter', type=int, default=config.get('max_iter', DEFAULT_MAX_ITER),
        help='Maximum optimizer iterations. Use "--max-iter --help" for details.',
    )
    parser.add_argument(
        '-n', '--simulations', type=int,
        default=config.get('simulations', DEFAULT_SIMULATIONS),
        help='Number of simulated portfolios. Use "-n --help" for details.',
    )
    parser.add_argument(
        '--seed', type=int, default=config.get('seed', DEFAULT_SEED),
        help='Random seed. Use "--seed --help" for details.',
    )

    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        '-p', '--plot', dest='show_plot', action='store_true',
        help='Show portfolio graph. Use "-p --help" for details.',
    )
    plot_group.add_argument(
        '--no-plot', dest='show_plot', action='store_false',
        help='Disable portfolio graph. Use "--no-plot --help" for details.',
    )
    parser.set_defaults(show_plot=config.get('show_plot', DEFAULT_PLOT))

    args = parser.parse_args()

    if args.years <= 0:
        parser.error('--years must be greater than zero.')
    if args.max_iter <= 0:
        parser.error('--max-iter must be greater than zero.')
    if args.simulations <= 0:
        parser.error('--simulations must be greater than zero.')
    if not 0 <= args.min_weight <= 100:
        parser.error('--min-weight must be between 0 and 100.')
    if not 0 < args.max_weight <= 100:
        parser.error('--max-weight must be greater than 0 and less than or equal to 100.')
    if args.min_weight > args.max_weight:
        parser.error('--min-weight cannot be greater than --max-weight.')

    if args.tickers is not None:
        ticker_spec = args.tickers
        ticker_source = 'Command line'
    elif 'tickers' in config:
        ticker_spec = config['tickers']
        ticker_source = f'Configuration: {config_path}'
    else:
        ticker_spec = DEFAULT_TICKERS
        ticker_source = 'Built-in defaults'

    return args, ticker_spec, ticker_source, config_path


# ============================================================
# Ticker input
# ============================================================

def parse_ticker_data(data, source):
    if isinstance(data, list):
        if len(data) < 2:
            raise ValueError('At least two tickers are required.')

        if not all(isinstance(ticker, str) and ticker.strip() for ticker in data):
            raise ValueError('Every ticker must be a non-empty string.')

        tickers = [ticker.strip() for ticker in data]
        if len(set(tickers)) != len(tickers):
            raise ValueError('Duplicate tickers are not allowed.')

        return tickers, None, source

    if isinstance(data, dict):
        if len(data) < 2:
            raise ValueError('At least two tickers are required.')

        values = {}

        for ticker, value in data.items():
            if not isinstance(ticker, str) or not ticker.strip():
                raise ValueError('Every ticker must be a non-empty string.')

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'Value for {ticker} must be numeric.')

            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f'Value for {ticker} must be finite.')
            if value < 0:
                raise ValueError(f'Value for {ticker} cannot be negative.')

            values[ticker.strip()] = value

        if sum(values.values()) <= 0:
            raise ValueError('Provided portfolio values must sum to a value greater than zero.')

        return list(values.keys()), values, source

    raise ValueError('Tickers must be either a list or an object containing ticker:value pairs.')


def load_ticker_json(path):
    path = Path(path).expanduser()

    if not path.is_file():
        raise ValueError(f'Ticker JSON file not found: {path}')

    try:
        with path.open('r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON in {path}: {exc}') from exc

    return parse_ticker_data(data, f'Ticker JSON: {path}')


def resolve_tickers(ticker_spec, ticker_source):
    if isinstance(ticker_spec, (dict, list)):
        is_json_path = (
            isinstance(ticker_spec, list)
            and len(ticker_spec) == 1
            and isinstance(ticker_spec[0], str)
            and Path(ticker_spec[0]).suffix.lower() == '.json'
        )
        if is_json_path:
            return load_ticker_json(ticker_spec[0])

        return parse_ticker_data(ticker_spec, ticker_source)

    if isinstance(ticker_spec, str):
        path = Path(ticker_spec).expanduser()
        if path.suffix.lower() == '.json':
            return load_ticker_json(path)

    raise ValueError('Invalid ticker specification.')


def normalize_provided_weights(raw_values, columns):
    if raw_values is None:
        return None

    values = np.array([raw_values[ticker] for ticker in columns], dtype=float)
    total = values.sum()

    if total <= 0:
        raise ValueError('Provided portfolio values must sum to a positive value.')

    return values / total


# ============================================================
# Weight constraints
# ============================================================

def validate_weight_constraints(n_assets, min_weight, max_weight):
    tolerance = 1e-12

    if min_weight > max_weight:
        raise ValueError('Minimum weight cannot be greater than maximum weight.')

    if n_assets * min_weight > 1.0 + tolerance:
        maximum_allowed = 100 / n_assets
        raise ValueError(
            'Minimum weight constraint is infeasible. '
            f'With {n_assets} assets, --min-weight cannot be greater than '
            f'{maximum_allowed:.2f} %.'
        )

    if n_assets * max_weight < 1.0 - tolerance:
        minimum_required = 100 / n_assets
        raise ValueError(
            'Maximum weight constraint is infeasible. '
            f'With {n_assets} assets, --max-weight must be at least '
            f'{minimum_required:.2f} %.'
        )


# ============================================================
# Output
# ============================================================

def print_configuration(args, ticker_source, config_path, n_assets, has_provided_portfolio):
    print_section_title('Configuration')

    rows = [
        ('Period', f'{args.years} years'),
        ('Risk-free rate', format_percent_value(args.rf)),
        ('Minimum weight', format_percent_value(args.min_weight)),
        ('Maximum weight', format_percent_value(args.max_weight)),
        ('Assets', str(n_assets)),
        ('Simulations', f'{args.simulations:,}'),
        ('Optimizer', 'SLSQP'),
        ('Max iterations', f'{args.max_iter:,}'),
        ('Plot', 'Yes' if args.show_plot else 'No'),
        ('Random seed', str(args.seed) if args.seed is not None else 'Random'),
        ('Provided portfolio', 'Yes' if has_provided_portfolio else 'No'),
        ('Source', ticker_source),
    ]

    if config_path is not None:
        rows.append(('Config file', str(config_path)))

    label_width = max(len(label) for label, _ in rows) + 2
    print()

    for label, value in rows:
        print(f'{label:<{label_width}}{value}')


# ============================================================
# Portfolio statistics
# ============================================================

def portfolio_return(weights, mu):
    return weights @ mu


def portfolio_variance(weights, covariance):
    return weights @ covariance @ weights


def portfolio_risk(weights, covariance):
    return np.sqrt(portfolio_variance(weights, covariance))


def portfolio_sharpe(weights, mu, covariance, rf):
    risk = portfolio_risk(weights, covariance)
    if risk <= 0:
        return -np.inf

    expected_return = portfolio_return(weights, mu)
    return (expected_return - rf) / risk


# ============================================================
# Optimization
# ============================================================

def optimize_portfolios(mu, covariance, rf, min_weight, max_weight, max_iter):
    n_assets = len(mu)
    validate_weight_constraints(n_assets, min_weight, max_weight)

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0},)
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    initial_weights = np.ones(n_assets) / n_assets
    options = {'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}

    min_var_result = minimize(
        fun=lambda weights: portfolio_variance(weights, covariance),
        x0=initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options=options,
    )

    if not min_var_result.success:
        raise RuntimeError(f'Minimum variance optimization failed: {min_var_result.message}')

    w_min_var = min_var_result.x

    def negative_sharpe(weights):
        return -portfolio_sharpe(weights, mu, covariance, rf)

    starting_points = [initial_weights, w_min_var]

    return_oriented = np.full(n_assets, min_weight, dtype=float)
    remaining_weight = 1.0 - n_assets * min_weight

    for index in np.argsort(mu)[::-1]:
        if remaining_weight <= 1e-12:
            break

        capacity = max_weight - return_oriented[index]
        allocation = min(capacity, remaining_weight)
        return_oriented[index] += allocation
        remaining_weight -= allocation

    starting_points.append(return_oriented)

    sharpe_results = []
    for starting_point in starting_points:
        result = minimize(
            fun=negative_sharpe,
            x0=starting_point,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options,
        )
        if result.success:
            sharpe_results.append(result)

    if not sharpe_results:
        raise RuntimeError('Maximum Sharpe optimization failed.')

    max_sharpe_result = min(sharpe_results, key=lambda result: result.fun)
    return max_sharpe_result.x, w_min_var


# ============================================================
# Portfolio tables
# ============================================================

def print_portfolio(title, weights, tickers, mu, covariance, rf):
    expected_return = portfolio_return(weights, mu)
    risk = portfolio_risk(weights, covariance)
    sharpe = portfolio_sharpe(weights, mu, covariance, rf)

    print_section_title(title)

    portfolio = sorted(zip(tickers, weights), key=lambda item: item[1], reverse=True)
    ticker_width = max(len('Ticker'), max(len(ticker) for ticker, _ in portfolio))
    weight_width = 14

    print()
    print(f'{"Ticker":<{ticker_width}}  {"Weight":<{weight_width}}')
    print('-' * (ticker_width + weight_width + 2))

    for ticker, weight in portfolio:
        print(f'{ticker:<{ticker_width}}  {format_percent(weight):<{weight_width}}')

    print_metrics(expected_return, risk, sharpe)


def print_asset_statistics(tickers, mu, returns):
    risks = np.sqrt(TRADING_DAYS) * np.std(returns, axis=0, ddof=1)
    rows = sorted(zip(tickers, mu, risks), key=lambda item: item[1], reverse=True)

    print_section_title('Individual Asset Statistics')

    ticker_width = max(len('Ticker'), max(len(ticker) for ticker in tickers))
    return_width = 20
    risk_width = 16

    print()
    print(f'{"Ticker":<{ticker_width}}  {"Expected Return":<{return_width}}{"Risk":<{risk_width}}')
    print('-' * (ticker_width + return_width + risk_width + 2))

    for ticker, expected_return, risk in rows:
        return_text = format_percent(expected_return)
        risk_text = format_percent(risk)
        print(f'{ticker:<{ticker_width}}  {return_text:<{return_width}}{risk_text:<{risk_width}}')


def print_provided_portfolio(
    title,
    raw_values,
    current_weights,
    target_weights,
    tickers,
    mu,
    covariance,
    rf,
):
    expected_return = portfolio_return(current_weights, mu)
    risk = portfolio_risk(current_weights, covariance)
    sharpe = portfolio_sharpe(current_weights, mu, covariance, rf)

    current_values = np.array([raw_values[ticker] for ticker in tickers], dtype=float)
    total_value = current_values.sum()
    target_values = target_weights * total_value
    delta_values = target_values - current_values
    delta_weights = target_weights - current_weights

    portfolio = sorted(
        zip(tickers, current_weights, target_weights, delta_weights, delta_values),
        key=lambda item: item[1],
        reverse=True,
    )

    print_section_title(title)

    ticker_width = max(len('Ticker'), max(len(ticker) for ticker in tickers))
    current_width = 13
    target_width = 13
    action_width = 8
    amount_width = 15
    delta_width = 14

    print()
    header = (
        f'{"Ticker":<{ticker_width}}  '
        f'{"Current":<{current_width}}'
        f'{"Target":<{target_width}}'
        f'{"Action":<{action_width}}'
        f'{"Amount":<{amount_width}}'
        f'{"Delta":<{delta_width}}'
    )
    print(header)
    print('-' * len(header))

    total_buy = 0.0
    total_sell = 0.0
    tolerance = 1e-8

    for ticker, current_weight, target_weight, delta_weight, delta_value in portfolio:
        if delta_value > tolerance:
            action = 'BUY'
            color = GREEN
            total_buy += delta_value
        elif delta_value < -tolerance:
            action = 'SELL'
            color = RED
            total_sell += abs(delta_value)
        else:
            action = 'HOLD'
            color = DIM

        current_text = format_percent(current_weight)
        target_text = format_percent(target_weight)
        amount_text = format_money(abs(delta_value))
        delta_text = format_weight_delta(delta_weight)

        print(
            f'{ticker:<{ticker_width}}  '
            f'{current_text:<{current_width}}'
            f'{target_text:<{target_width}}',
            end='',
        )
        print(f'{color}{action:<{action_width}}{RESET}', end='')
        print(f'{color}{amount_text:<{amount_width}}{RESET}', end='')
        print(f'{delta_text:<{delta_width}}')

    print()
    print('-' * OUTPUT_WIDTH)
    print(f'{"Portfolio value":<24}{format_money(total_value):>16}')
    print(f'{"Total BUY":<24}{GREEN}{format_money(total_buy):>16}{RESET}')
    print(f'{"Total SELL":<24}{RED}{format_money(total_sell):>16}{RESET}')

    print_metrics(expected_return, risk, sharpe)


# ============================================================
# Random portfolios
# ============================================================

def enforce_max_weight(weights, max_weight):
    if max_weight >= 1.0:
        return weights

    weights = weights.copy()

    for _ in range(100):
        excess = np.maximum(weights - max_weight, 0.0)
        excess_sum = excess.sum(axis=1)
        mask = excess_sum > 1e-12

        if not np.any(mask):
            break

        weights = np.minimum(weights, max_weight)
        capacity = np.maximum(max_weight - weights, 0.0)
        capacity_sum = capacity.sum(axis=1)
        valid = mask & (capacity_sum > 1e-12)

        if not np.any(valid):
            break

        weights[valid] += capacity[valid] * (
            excess_sum[valid] / capacity_sum[valid]
        )[:, None]

    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def generate_random_weights(rng, n_portfolios, n_assets, min_weight, max_weight):
    validate_weight_constraints(n_assets, min_weight, max_weight)

    remaining_weight = 1.0 - n_assets * min_weight
    if remaining_weight <= 1e-12:
        return np.full((n_portfolios, n_assets), 1.0 / n_assets, dtype=float)

    alpha = 0.10 + 0.90 * rng.beta(1.0, 3.0, size=n_portfolios)
    gamma_samples = rng.gamma(
        shape=alpha[:, None],
        scale=1.0,
        size=(n_portfolios, n_assets),
    )

    row_sums = gamma_samples.sum(axis=1, keepdims=True)
    invalid_rows = row_sums[:, 0] <= 0

    if np.any(invalid_rows):
        gamma_samples[invalid_rows] = 1.0
        row_sums = gamma_samples.sum(axis=1, keepdims=True)

    residual_weights = gamma_samples / row_sums
    residual_max_weight = (max_weight - min_weight) / remaining_weight
    residual_weights = enforce_max_weight(residual_weights, residual_max_weight)

    return min_weight + remaining_weight * residual_weights


# ============================================================
# Historical coverage warning
# ============================================================

def print_history_coverage_warning(stocks, tickers, requested_start, requested_end, years):
    expected_observations = max(1, round(TRADING_DAYS * years))
    warnings = []

    for ticker in tickers:
        series = stocks[ticker].dropna()
        if series.empty:
            continue

        first_timestamp = series.index[0]
        last_timestamp = series.index[-1]
        first_date = first_timestamp.date() if hasattr(first_timestamp, 'date') else first_timestamp
        last_date = last_timestamp.date() if hasattr(last_timestamp, 'date') else last_timestamp

        observations = len(series)
        coverage = min(observations / expected_observations, 1.0)
        start_delay_days = max(0, (first_date - requested_start).days)

        insufficient_history = (
            start_delay_days > HISTORY_START_TOLERANCE_DAYS
            or coverage < HISTORY_MIN_OBSERVATION_COVERAGE
        )

        if insufficient_history:
            warnings.append((ticker, first_date, last_date, observations, coverage))

    if not warnings:
        return

    common_stocks = stocks[tickers].dropna()
    common_observations = len(common_stocks)
    common_coverage = min(common_observations / expected_observations, 1.0)

    if common_observations:
        first_common = common_stocks.index[0]
        last_common = common_stocks.index[-1]
        common_start = first_common.date() if hasattr(first_common, 'date') else first_common
        common_end = last_common.date() if hasattr(last_common, 'date') else last_common
    else:
        common_start = None
        common_end = None

    ticker_width = max(len('Ticker'), max(len(item[0]) for item in warnings))
    first_width = 14
    last_width = 14
    observations_width = 14
    coverage_width = 12

    print()
    print(f'{RED}{BOLD}{"!" * OUTPUT_WIDTH}{RESET}')
    print(f'{RED}{BOLD}{"WARNING: INCOMPLETE HISTORICAL COVERAGE":^{OUTPUT_WIDTH}}{RESET}')
    print(f'{RED}{BOLD}{"!" * OUTPUT_WIDTH}{RESET}')
    print(f'{RED}Requested with -y {years}: {requested_start} -> {requested_end}{RESET}')
    print(f'{RED}Expected observations (approx.): {expected_observations:,}{RESET}')
    print()

    header = (
        f'{"Ticker":<{ticker_width}}  '
        f'{"First obs":<{first_width}}'
        f'{"Last obs":<{last_width}}'
        f'{"Observations":<{observations_width}}'
        f'{"Coverage":<{coverage_width}}'
    )
    print(f'{RED}{header}{RESET}')
    print(f'{RED}{"-" * len(header)}{RESET}')

    for ticker, first_date, last_date, observations, coverage in warnings:
        print(
            f'{RED}'
            f'{ticker:<{ticker_width}}  '
            f'{str(first_date):<{first_width}}'
            f'{str(last_date):<{last_width}}'
            f'{observations:<{observations_width},}'
            f'{f"{100 * coverage:.1f} %":<{coverage_width}}'
            f'{RESET}'
        )

    print()
    print(f'{RED}One or more series do not fully cover the requested historical period.{RESET}')

    if common_observations:
        print(f'{RED}After aligning all assets, the effective common sample will be:{RESET}')
        print(f'{RED}  Period: {common_start} -> {common_end}{RESET}')
        print(
            f'{RED}  Observations: {common_observations:,} '
            f'({100 * common_coverage:.1f} % of the requested approximate trading observations)'
            f'{RESET}'
        )
    else:
        print(f'{RED}There are no common observations across all assets.{RESET}')

    print(f'{RED}Results will use the common history available after alignment.{RESET}')
    print(f'{RED}{BOLD}{"!" * OUTPUT_WIDTH}{RESET}')


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
    min_weight,
    max_weight,
    years,
    provided_weights=None,
):
    n_assets = len(mu)

    weights = generate_random_weights(
        rng=rng,
        n_portfolios=simulations,
        n_assets=n_assets,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    random_returns = weights @ mu
    random_variances = np.einsum('ij,jk,ik->i', weights, covariance, weights)
    random_risks = np.sqrt(random_variances)
    random_sharpes = (random_returns - rf) / random_risks

    max_sharpe_return = portfolio_return(w_max_sharpe, mu)
    max_sharpe_risk = portfolio_risk(w_max_sharpe, covariance)
    min_var_return = portfolio_return(w_min_var, mu)
    min_var_risk = portfolio_risk(w_min_var, covariance)

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        100 * random_risks,
        100 * random_returns,
        c=random_sharpes,
        s=5,
        alpha=0.35,
    )
    plt.colorbar(scatter, label='Sharpe Ratio')

    plt.scatter(
        100 * max_sharpe_risk,
        100 * max_sharpe_return,
        color='red',
        s=130,
        label='Maximum Sharpe',
        zorder=3,
    )
    plt.scatter(
        100 * min_var_risk,
        100 * min_var_return,
        color='blue',
        s=130,
        label='Minimum Variance',
        zorder=3,
    )

    if provided_weights is not None:
        provided_return = portfolio_return(provided_weights, mu)
        provided_risk = portfolio_risk(provided_weights, covariance)

        plt.scatter(
            100 * provided_risk,
            100 * provided_return,
            color='green',
            marker='D',
            s=140,
            label='Provided Portfolio',
            zorder=4,
        )
        plt.annotate(
            '',
            xy=(100 * max_sharpe_risk, 100 * max_sharpe_return),
            xytext=(100 * provided_risk, 100 * provided_return),
            arrowprops={'arrowstyle': '->', 'linestyle': '--'},
        )

    plt.plot(
        [0, 100 * max_sharpe_risk],
        [100 * rf, 100 * max_sharpe_return],
        color='red',
        linestyle='dotted',
    )

    plt.xlabel('Annualized Risk (%)')
    plt.ylabel('Annualized Expected Return (%)')
    plt.title(f'Markowitz Portfolio Optimization - {years} Years')

    solid_red = (1.0, 0.0, 0.0, 1.0)
    solid_blue = (0.0, 0.0, 1.0, 1.0)
    solid_green = (0.0, 0.5019607843137255, 0.0, 1.0)

    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            color=solid_red,
            markerfacecolor=solid_red,
            markeredgecolor=solid_red,
            markeredgewidth=0.0,
            fillstyle='full',
            markersize=10,
            alpha=1.0,
            label='Maximum Sharpe',
        ),
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            color=solid_blue,
            markerfacecolor=solid_blue,
            markeredgecolor=solid_blue,
            markeredgewidth=0.0,
            fillstyle='full',
            markersize=10,
            alpha=1.0,
            label='Minimum Variance',
        ),
    ]

    if provided_weights is not None:
        legend_handles.append(
            Line2D(
                [0], [0],
                marker='D',
                linestyle='None',
                color=solid_green,
                markerfacecolor=solid_green,
                markeredgecolor=solid_green,
                markeredgewidth=0.0,
                fillstyle='full',
                markersize=9,
                alpha=1.0,
                label='Provided Portfolio',
            )
        )

    legend = plt.legend(
        handles=legend_handles,
        loc='upper left',
        frameon=True,
        fancybox=True,
        framealpha=1.0,
    )

    legend_frame = legend.get_frame()
    legend_frame.set_fill(True)
    legend_frame.set_facecolor((1.0, 1.0, 1.0, 1.0))
    legend_frame.set_edgecolor((0.8, 0.8, 0.8, 1.0))
    legend_frame.set_alpha(1.0)

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    args, ticker_spec, ticker_source, config_path = parse_args()

    try:
        tickers, provided_values_raw, ticker_source = resolve_tickers(ticker_spec, ticker_source)
    except ValueError as exc:
        raise SystemExit(f'Error: {exc}') from exc

    min_weight = args.min_weight / 100
    max_weight = args.max_weight / 100

    try:
        validate_weight_constraints(len(tickers), min_weight, max_weight)
    except ValueError as exc:
        raise SystemExit(f'Error: {exc}') from exc

    print_configuration(
        args=args,
        ticker_source=ticker_source,
        config_path=config_path,
        n_assets=len(tickers),
        has_provided_portfolio=provided_values_raw is not None,
    )

    rng = np.random.default_rng(args.seed)
    end = date.today()
    start = end - relativedelta(years=args.years)

    print()
    print_status(f'Downloading {args.years} years of data for {len(tickers)} assets...')

    stocks = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
    )['Close']
    stocks = stocks.dropna(axis=1, how='all')

    missing_tickers = [ticker for ticker in tickers if ticker not in stocks.columns]
    if missing_tickers:
        raise RuntimeError('No valid price data was returned for: ' + ', '.join(missing_tickers))

    print_history_coverage_warning(
        stocks=stocks,
        tickers=tickers,
        requested_start=start,
        requested_end=end,
        years=args.years,
    )

    stocks = stocks[tickers].dropna()
    if stocks.shape[1] < 2:
        raise RuntimeError('Could not obtain valid data for at least two assets.')

    provided_weights = normalize_provided_weights(provided_values_raw, stocks.columns)

    returns = stocks.pct_change().dropna().values
    mu = TRADING_DAYS * np.mean(returns, axis=0)
    covariance = TRADING_DAYS * np.cov(returns.T)
    rf = args.rf / 100

    print_status('Optimizing portfolios...')

    w_max_sharpe, w_min_var = optimize_portfolios(
        mu=mu,
        covariance=covariance,
        rf=rf,
        min_weight=min_weight,
        max_weight=max_weight,
        max_iter=args.max_iter,
    )

    print_portfolio(
        title='Maximum Sharpe Portfolio',
        weights=w_max_sharpe,
        tickers=stocks.columns,
        mu=mu,
        covariance=covariance,
        rf=rf,
    )
    print_portfolio(
        title='Minimum Variance Portfolio',
        weights=w_min_var,
        tickers=stocks.columns,
        mu=mu,
        covariance=covariance,
        rf=rf,
    )
    print_asset_statistics(stocks.columns, mu, returns)

    if provided_weights is not None:
        print_provided_portfolio(
            title='Provided Portfolio -> Maximum Sharpe',
            raw_values=provided_values_raw,
            current_weights=provided_weights,
            target_weights=w_max_sharpe,
            tickers=stocks.columns,
            mu=mu,
            covariance=covariance,
            rf=rf,
        )

    if args.show_plot:
        plot_portfolios(
            rng=rng,
            simulations=args.simulations,
            mu=mu,
            covariance=covariance,
            rf=rf,
            w_max_sharpe=w_max_sharpe,
            w_min_var=w_min_var,
            min_weight=min_weight,
            max_weight=max_weight,
            years=args.years,
            provided_weights=provided_weights,
        )


if __name__ == '__main__':
    handle_contextual_help()
    main()
