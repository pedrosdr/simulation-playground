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
YELLOW = '\033[93m'
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

    py markowitz.py -t PETR4.SA VALE3.SA ITUB3.SA

or:

    py markowitz.py --tickers PETR4.SA VALE3.SA ITUB3.SA


2. JSON containing a ticker list
---------------------------------

Example: tickers.json

[
    "PETR4.SA",
    "VALE3.SA",
    "BBAS3.SA",
    "ITUB3.SA"
]

Run:

    py markowitz.py -t tickers.json


3. JSON containing a current portfolio
---------------------------------------

Example: portfolio.json

{
    "PETR4.SA": 2840.65,
    "VALE3.SA": 1935.20,
    "BBAS3.SA": 1376.80,
    "ITUB3.SA": 842.45
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
        "PETR4.SA": 2840.65,
        "VALE3.SA": 1935.20,
        "BBAS3.SA": 1376.80,
        "CMIG4.SA": 1188.35,
        "ITSA4.SA": 1043.90,
        "TAEE11.SA": 916.70,
        "CPFE3.SA": 773.55,
        "PSSA3.SA": 621.40,
        "ITUB3.SA": 842.45,
        "WEGE3.SA": 468.25,
        "SAPR11.SA": 296.10,
        "SOJA3.SA": 157.80
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
        "PETR4.SA",
        "VALE3.SA",
        "BBAS3.SA",
        "ITUB3.SA"
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

    '--no-plot': NO_PLOT_HELP
}


# ============================================================
# Contextual help handling
# ============================================================

def handle_contextual_help():

    args = sys.argv[1:]

    help_flags = {
        '-h',
        '--help'
    }

    for index, argument in enumerate(args):

        if argument not in ARGUMENT_HELP:
            continue

        if (
            index + 1 < len(args)
            and args[index + 1] in help_flags
        ):

            print(
                ARGUMENT_HELP[
                    argument
                ].strip()
            )

            raise SystemExit(0)


# ============================================================
# Formatting helpers
# ============================================================

def format_percent(
    value,
    decimals=2
):

    return (
        f'{100 * value:.{decimals}f} %'
    )


def format_percent_value(
    value,
    decimals=2
):

    return (
        f'{value:.{decimals}f} %'
    )


def format_weight_delta(
    value,
    decimals=2
):

    return (
        f'{100 * value:+.{decimals}f} pp'
    )


def format_money(
    value
):

    return (
        f'{value:,.2f}'
    )


# ============================================================
# Output helpers
# ============================================================

def print_section_title(title):

    title = (
        title.upper()
    )

    print()

    print(
        f'{BOLD}{CYAN}'
        f'{"=" * OUTPUT_WIDTH}'
        f'{RESET}'
    )

    print(
        f'{BOLD}{CYAN}'
        f'{title:^{OUTPUT_WIDTH}}'
        f'{RESET}'
    )

    print(
        f'{BOLD}{CYAN}'
        f'{"=" * OUTPUT_WIDTH}'
        f'{RESET}'
    )


def print_metrics(
    expected_return,
    risk,
    sharpe
):

    return_text = (
        format_percent(
            expected_return
        )
    )

    risk_text = (
        format_percent(
            risk
        )
    )

    sharpe_text = (
        f'{sharpe:.3f}'
    )

    column_width = (
        OUTPUT_WIDTH // 3
    )

    print()

    print(
        '-' * OUTPUT_WIDTH
    )

    print(
        f'{"RETURN":<{column_width}}'
        f'{"RISK":<{column_width}}'
        f'{"SHARPE":<{column_width}}'
    )

    print(
        f'{return_text:<{column_width}}'
        f'{risk_text:<{column_width}}'
        f'{sharpe_text:<{column_width}}'
    )

    print(
        '-' * OUTPUT_WIDTH
    )


def print_status(message):

    print(
        f'{CYAN}'
        f'{message}'
        f'{RESET}'
    )


# ============================================================
# JSON configuration
# ============================================================

def load_config(path):

    path = Path(
        path
    ).expanduser()

    if not path.is_file():

        raise ValueError(
            f'Configuration file not found: {path}'
        )

    try:

        with path.open(
            'r',
            encoding='utf-8'
        ) as file:

            data = json.load(
                file
            )

    except json.JSONDecodeError as exc:

        raise ValueError(
            f'Invalid JSON in configuration file '
            f'{path}: {exc}'
        ) from exc

    if not isinstance(
        data,
        dict
    ):

        raise ValueError(
            'Configuration JSON must contain a JSON object.'
        )

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

        'tickers': 'tickers'
    }

    config = {}

    for key, value in data.items():

        if key not in key_map:

            raise ValueError(
                f'Unknown configuration parameter: {key}'
            )

        normalized_key = (
            key_map[key]
        )

        if normalized_key in config:

            raise ValueError(
                f'Duplicate configuration parameter: {key}'
            )

        config[
            normalized_key
        ] = value

    # --------------------------------------------------------
    # years
    # --------------------------------------------------------

    if 'years' in config:

        if (
            isinstance(
                config['years'],
                bool
            )
            or not isinstance(
                config['years'],
                int
            )
            or config['years'] <= 0
        ):

            raise ValueError(
                '"years" must be a positive integer.'
            )

    # --------------------------------------------------------
    # rf
    # --------------------------------------------------------

    if 'rf' in config:

        if (
            isinstance(
                config['rf'],
                bool
            )
            or not isinstance(
                config['rf'],
                (int, float)
            )
        ):

            raise ValueError(
                '"rf" must be numeric.'
            )

    # --------------------------------------------------------
    # min-weight
    # --------------------------------------------------------

    if 'min_weight' in config:

        value = (
            config['min_weight']
        )

        if (
            isinstance(
                value,
                bool
            )
            or not isinstance(
                value,
                (int, float)
            )
            or not 0 <= value <= 100
        ):

            raise ValueError(
                '"min-weight" must be between 0 and 100.'
            )

    # --------------------------------------------------------
    # max-weight
    # --------------------------------------------------------

    if 'max_weight' in config:

        value = (
            config['max_weight']
        )

        if (
            isinstance(
                value,
                bool
            )
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

    # --------------------------------------------------------
    # min <= max
    # --------------------------------------------------------

    config_min_weight = config.get(
        'min_weight',
        DEFAULT_MIN_WEIGHT
    )

    config_max_weight = config.get(
        'max_weight',
        DEFAULT_MAX_WEIGHT
    )

    if (
        config_min_weight
        > config_max_weight
    ):

        raise ValueError(
            '"min-weight" cannot be greater '
            'than "max-weight".'
        )

    # --------------------------------------------------------
    # max-iter
    # --------------------------------------------------------

    if 'max_iter' in config:

        if (
            isinstance(
                config['max_iter'],
                bool
            )
            or not isinstance(
                config['max_iter'],
                int
            )
            or config['max_iter'] <= 0
        ):

            raise ValueError(
                '"max-iter" must be a positive integer.'
            )

    # --------------------------------------------------------
    # simulations
    # --------------------------------------------------------

    if 'simulations' in config:

        if (
            isinstance(
                config['simulations'],
                bool
            )
            or not isinstance(
                config['simulations'],
                int
            )
            or config['simulations'] <= 0
        ):

            raise ValueError(
                '"simulations" must be a positive integer.'
            )

    # --------------------------------------------------------
    # seed
    # --------------------------------------------------------

    if 'seed' in config:

        if (
            config['seed'] is not None
            and (
                isinstance(
                    config['seed'],
                    bool
                )
                or not isinstance(
                    config['seed'],
                    int
                )
            )
        ):

            raise ValueError(
                '"seed" must be an integer or null.'
            )

    # --------------------------------------------------------
    # plot
    # --------------------------------------------------------

    if 'show_plot' in config:

        if not isinstance(
            config['show_plot'],
            bool
        ):

            raise ValueError(
                '"plot" must be true or false.'
            )

    return (
        config,
        path
    )


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():

    # --------------------------------------------------------
    # First pass: configuration file
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

    pre_args, _ = (
        pre_parser.parse_known_args()
    )

    config = {}
    config_path = None

    if pre_args.config is not None:

        try:

            (
                config,
                config_path

            ) = load_config(
                pre_args.config
            )

        except ValueError as exc:

            raise SystemExit(
                f'Error: {exc}'
            ) from exc

    # --------------------------------------------------------
    # Main parser
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            'Markowitz portfolio optimization using SciPy.'
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        )
    )

    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default=pre_args.config,
        help=(
            'JSON configuration file. '
            'Use "-c --help" for details.'
        )
    )

    parser.add_argument(
        '-y',
        '--years',
        type=int,
        default=config.get(
            'years',
            DEFAULT_YEARS
        ),
        help=(
            'Historical period in years. '
            'Use "-y --help" for details.'
        )
    )

    parser.add_argument(
        '-r',
        '--rf',
        type=float,
        default=config.get(
            'rf',
            DEFAULT_RF
        ),
        help=(
            'Annual risk-free rate in percent. '
            'Use "-r --help" for details.'
        )
    )

    parser.add_argument(
        '-t',
        '--tickers',
        nargs='+',
        default=None,
        help=(
            'Tickers or ticker/portfolio JSON file. '
            'Use "-t --help" for details.'
        )
    )

    parser.add_argument(
        '--min-weight',
        type=float,
        default=config.get(
            'min_weight',
            DEFAULT_MIN_WEIGHT
        ),
        help=(
            'Minimum optimized asset weight in percent. '
            'Use "--min-weight --help" for details.'
        )
    )

    parser.add_argument(
        '--max-weight',
        type=float,
        default=config.get(
            'max_weight',
            DEFAULT_MAX_WEIGHT
        ),
        help=(
            'Maximum optimized asset weight in percent. '
            'Use "--max-weight --help" for details.'
        )
    )

    parser.add_argument(
        '--max-iter',
        type=int,
        default=config.get(
            'max_iter',
            DEFAULT_MAX_ITER
        ),
        help=(
            'Maximum optimizer iterations. '
            'Use "--max-iter --help" for details.'
        )
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
            'Number of simulated portfolios. '
            'Use "-n --help" for details.'
        )
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=config.get(
            'seed',
            DEFAULT_SEED
        ),
        help=(
            'Random seed. '
            'Use "--seed --help" for details.'
        )
    )

    plot_group = (
        parser.add_mutually_exclusive_group()
    )

    plot_group.add_argument(
        '-p',
        '--plot',
        dest='show_plot',
        action='store_true',
        help=(
            'Show portfolio graph. '
            'Use "-p --help" for details.'
        )
    )

    plot_group.add_argument(
        '--no-plot',
        dest='show_plot',
        action='store_false',
        help=(
            'Disable portfolio graph. '
            'Use "--no-plot --help" for details.'
        )
    )

    parser.set_defaults(
        show_plot=config.get(
            'show_plot',
            DEFAULT_PLOT
        )
    )

    args = (
        parser.parse_args()
    )

    # --------------------------------------------------------
    # Basic validation
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

    if not 0 <= args.min_weight <= 100:

        parser.error(
            '--min-weight must be between 0 and 100.'
        )

    if not 0 < args.max_weight <= 100:

        parser.error(
            '--max-weight must be greater than 0 '
            'and less than or equal to 100.'
        )

    if (
        args.min_weight
        > args.max_weight
    ):

        parser.error(
            '--min-weight cannot be greater '
            'than --max-weight.'
        )

    # --------------------------------------------------------
    # Ticker source precedence
    #
    # defaults < config < command line
    # --------------------------------------------------------

    if args.tickers is not None:

        ticker_spec = (
            args.tickers
        )

        ticker_source = (
            'Command line'
        )

    elif 'tickers' in config:

        ticker_spec = (
            config['tickers']
        )

        ticker_source = (
            f'Configuration: {config_path}'
        )

    else:

        ticker_spec = (
            DEFAULT_TICKERS
        )

        ticker_source = (
            'Built-in defaults'
        )

    return (
        args,
        ticker_spec,
        ticker_source,
        config_path
    )


# ============================================================
# Ticker data
# ============================================================

def parse_ticker_data(
    data,
    source
):

    # --------------------------------------------------------
    # Ticker list
    # --------------------------------------------------------

    if isinstance(
        data,
        list
    ):

        if len(data) < 2:

            raise ValueError(
                'At least two tickers are required.'
            )

        if not all(
            isinstance(
                ticker,
                str
            )
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

        if (
            len(set(tickers))
            != len(tickers)
        ):

            raise ValueError(
                'Duplicate tickers are not allowed.'
            )

        return (
            tickers,
            None,
            source
        )

    # --------------------------------------------------------
    # Provided portfolio
    # --------------------------------------------------------

    if isinstance(
        data,
        dict
    ):

        if len(data) < 2:

            raise ValueError(
                'At least two tickers are required.'
            )

        values = {}

        for ticker, value in data.items():

            if (
                not isinstance(
                    ticker,
                    str
                )
                or not ticker.strip()
            ):

                raise ValueError(
                    'Every ticker must be a non-empty string.'
                )

            if (
                isinstance(
                    value,
                    bool
                )
                or not isinstance(
                    value,
                    (int, float)
                )
            ):

                raise ValueError(
                    f'Value for {ticker} must be numeric.'
                )

            value = float(
                value
            )

            if not np.isfinite(
                value
            ):

                raise ValueError(
                    f'Value for {ticker} must be finite.'
                )

            if value < 0:

                raise ValueError(
                    f'Value for {ticker} cannot be negative.'
                )

            values[
                ticker.strip()
            ] = value

        if (
            sum(values.values())
            <= 0
        ):

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
# Load ticker JSON
# ============================================================

def load_ticker_json(path):

    path = Path(
        path
    ).expanduser()

    if not path.is_file():

        raise ValueError(
            f'Ticker JSON file not found: {path}'
        )

    try:

        with path.open(
            'r',
            encoding='utf-8'
        ) as file:

            data = json.load(
                file
            )

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

        if (
            isinstance(
                ticker_spec,
                list
            )
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

    if isinstance(
        ticker_spec,
        str
    ):

        path = Path(
            ticker_spec
        ).expanduser()

        if (
            path.suffix.lower()
            == '.json'
        ):

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

    total = (
        values.sum()
    )

    if total <= 0:

        raise ValueError(
            'Provided portfolio values '
            'must sum to a positive value.'
        )

    return (
        values
        / total
    )


# ============================================================
# Weight constraint validation
# ============================================================

def validate_weight_constraints(
    n_assets,
    min_weight,
    max_weight
):

    tolerance = (
        1e-12
    )

    if (
        min_weight
        > max_weight
    ):

        raise ValueError(
            'Minimum weight cannot be greater '
            'than maximum weight.'
        )

    # --------------------------------------------------------
    # Minimum-weight feasibility
    #
    # N * min_weight <= 1
    # --------------------------------------------------------

    if (
        n_assets * min_weight
        > 1.0 + tolerance
    ):

        maximum_allowed = (
            100
            / n_assets
        )

        raise ValueError(
            'Minimum weight constraint is infeasible. '
            f'With {n_assets} assets, '
            f'--min-weight cannot be greater than '
            f'{maximum_allowed:.2f} %.'
        )

    # --------------------------------------------------------
    # Maximum-weight feasibility
    #
    # N * max_weight >= 1
    # --------------------------------------------------------

    if (
        n_assets * max_weight
        < 1.0 - tolerance
    ):

        minimum_required = (
            100
            / n_assets
        )

        raise ValueError(
            'Maximum weight constraint is infeasible. '
            f'With {n_assets} assets, '
            f'--max-weight must be at least '
            f'{minimum_required:.2f} %.'
        )


# ============================================================
# Configuration output
# ============================================================

def print_configuration(
    args,
    ticker_source,
    config_path,
    n_assets,
    has_provided_portfolio
):

    print_section_title(
        'Configuration'
    )

    rows = [
        (
            'Period',
            f'{args.years} years'
        ),
        (
            'Risk-free rate',
            format_percent_value(
                args.rf
            )
        ),
        (
            'Minimum weight',
            format_percent_value(
                args.min_weight
            )
        ),
        (
            'Maximum weight',
            format_percent_value(
                args.max_weight
            )
        ),
        (
            'Assets',
            str(
                n_assets
            )
        ),
        (
            'Simulations',
            f'{args.simulations:,}'
        ),
        (
            'Optimizer',
            'SLSQP'
        ),
        (
            'Max iterations',
            f'{args.max_iter:,}'
        ),
        (
            'Plot',
            (
                'Yes'
                if args.show_plot
                else 'No'
            )
        ),
        (
            'Random seed',
            (
                str(args.seed)
                if args.seed is not None
                else 'Random'
            )
        ),
        (
            'Provided portfolio',
            (
                'Yes'
                if has_provided_portfolio
                else 'No'
            )
        ),
        (
            'Source',
            ticker_source
        )
    ]

    if config_path is not None:

        rows.append(
            (
                'Config file',
                str(
                    config_path
                )
            )
        )

    label_width = (
        max(
            len(label)
            for label, _ in rows
        )
        + 2
    )

    print()

    for label, value in rows:

        print(
            f'{label:<{label_width}}'
            f'{value}'
        )


# ============================================================
# Portfolio statistics
# ============================================================

def portfolio_return(
    weights,
    mu
):

    return (
        weights @ mu
    )


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

    expected_return = (
        portfolio_return(
            weights,
            mu
        )
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
    min_weight,
    max_weight,
    max_iter
):

    n_assets = (
        len(mu)
    )

    validate_weight_constraints(
        n_assets=n_assets,
        min_weight=min_weight,
        max_weight=max_weight
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
    # min_weight <= weight <= max_weight
    # --------------------------------------------------------

    bounds = [
        (
            min_weight,
            max_weight
        )
        for _ in range(
            n_assets
        )
    ]

    # --------------------------------------------------------
    # Equal weight is guaranteed to be feasible when:
    #
    # N * min_weight <= 1
    # N * max_weight >= 1
    # --------------------------------------------------------

    w0 = (
        np.ones(
            n_assets
        )
        / n_assets
    )

    options = {
        'maxiter': max_iter,
        'ftol': 1e-12,
        'disp': False
    }

    # ========================================================
    # Minimum Variance
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
    # Maximum Sharpe
    # ========================================================

    def negative_sharpe(
        weights
    ):

        return -portfolio_sharpe(
            weights,
            mu,
            covariance,
            rf
        )

    starting_points = [
        w0,
        w_min_var
    ]

    # --------------------------------------------------------
    # Return-oriented starting point
    #
    # Start every asset at min_weight, then allocate all
    # remaining capital to assets with highest expected return
    # until each reaches max_weight.
    # --------------------------------------------------------

    w_return = np.full(
        n_assets,
        min_weight,
        dtype=float
    )

    remaining_weight = (
        1.0
        - n_assets * min_weight
    )

    for idx in (
        np.argsort(mu)[::-1]
    ):

        if (
            remaining_weight
            <= 1e-12
        ):

            break

        capacity = (
            max_weight
            - w_return[idx]
        )

        allocation = min(
            capacity,
            remaining_weight
        )

        w_return[
            idx
        ] += allocation

        remaining_weight -= (
            allocation
        )

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
# Optimized portfolio output
# ============================================================

def print_portfolio(
    title,
    weights,
    tickers,
    mu,
    covariance,
    rf
):

    expected_return = (
        portfolio_return(
            weights,
            mu
        )
    )

    risk = (
        portfolio_risk(
            weights,
            covariance
        )
    )

    sharpe = (
        portfolio_sharpe(
            weights,
            mu,
            covariance,
            rf
        )
    )

    print_section_title(
        title
    )

    # --------------------------------------------------------
    # Show all assets.
    # --------------------------------------------------------

    portfolio = sorted(
        zip(
            tickers,
            weights
        ),
        key=lambda item:
            item[1],
        reverse=True
    )

    ticker_width = max(
        len('Ticker'),
        max(
            len(ticker)
            for ticker, _ in portfolio
        )
    )

    weight_width = 14

    print()

    print(
        f'{"Ticker":<{ticker_width}}  '
        f'{"Weight":<{weight_width}}'
    )

    print(
        '-' * (
            ticker_width
            + weight_width
            + 2
        )
    )

    for ticker, weight in portfolio:

        weight_text = (
            format_percent(
                weight
            )
        )

        print(
            f'{ticker:<{ticker_width}}  '
            f'{weight_text:<{weight_width}}'
        )

    print_metrics(
        expected_return=expected_return,
        risk=risk,
        sharpe=sharpe
    )


# ============================================================
# Individual asset statistics
# ============================================================

def print_asset_statistics(
    tickers,
    mu,
    returns
):

    risks = (
        np.sqrt(
            TRADING_DAYS
        )
        * np.std(
            returns,
            axis=0,
            ddof=1
        )
    )

    rows = sorted(
        zip(
            tickers,
            mu,
            risks
        ),
        key=lambda item:
            item[1],
        reverse=True
    )

    print_section_title(
        'Individual Asset Statistics'
    )

    ticker_width = max(
        len('Ticker'),
        max(
            len(ticker)
            for ticker in tickers
        )
    )

    return_width = 20
    risk_width = 16

    print()

    print(
        f'{"Ticker":<{ticker_width}}  '
        f'{"Expected Return":<{return_width}}'
        f'{"Risk":<{risk_width}}'
    )

    print(
        '-' * (
            ticker_width
            + return_width
            + risk_width
            + 2
        )
    )

    for (
        ticker,
        expected_return,
        risk

    ) in rows:

        return_text = (
            format_percent(
                expected_return
            )
        )

        risk_text = (
            format_percent(
                risk
            )
        )

        print(
            f'{ticker:<{ticker_width}}  '
            f'{return_text:<{return_width}}'
            f'{risk_text:<{risk_width}}'
        )


# ============================================================
# Provided portfolio + rebalancing
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

    expected_return = (
        portfolio_return(
            current_weights,
            mu
        )
    )

    risk = (
        portfolio_risk(
            current_weights,
            covariance
        )
    )

    sharpe = (
        portfolio_sharpe(
            current_weights,
            mu,
            covariance,
            rf
        )
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
    # Maximum Sharpe target monetary values
    # --------------------------------------------------------

    target_values = (
        target_weights
        * total_value
    )

    # --------------------------------------------------------
    # Rebalancing
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
    # Sort by current weight, highest to lowest.
    # --------------------------------------------------------

    portfolio = sorted(
        zip(
            tickers,
            current_weights,
            target_weights,
            delta_weights,
            delta_values
        ),
        key=lambda item:
            item[1],
        reverse=True
    )

    print_section_title(
        title
    )

    ticker_width = max(
        len('Ticker'),
        max(
            len(ticker)
            for ticker in tickers
        )
    )

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

    print(
        header
    )

    print(
        '-' * len(
            header
        )
    )

    total_buy = 0.0
    total_sell = 0.0

    for (
        ticker,
        current_weight,
        target_weight,
        delta_weight,
        delta_value

    ) in portfolio:

        tolerance = (
            1e-8
        )

        if (
            delta_value
            > tolerance
        ):

            action = (
                'BUY'
            )

            color = (
                GREEN
            )

            total_buy += (
                delta_value
            )

        elif (
            delta_value
            < -tolerance
        ):

            action = (
                'SELL'
            )

            color = (
                RED
            )

            total_sell += abs(
                delta_value
            )

        else:

            action = (
                'HOLD'
            )

            color = (
                DIM
            )

        current_text = (
            format_percent(
                current_weight
            )
        )

        target_text = (
            format_percent(
                target_weight
            )
        )

        amount_text = (
            format_money(
                abs(
                    delta_value
                )
            )
        )

        delta_text = (
            format_weight_delta(
                delta_weight
            )
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
    # Monetary summary
    # --------------------------------------------------------

    print()

    print(
        '-' * OUTPUT_WIDTH
    )

    portfolio_value_text = (
        format_money(
            total_value
        )
    )

    total_buy_text = (
        format_money(
            total_buy
        )
    )

    total_sell_text = (
        format_money(
            total_sell
        )
    )

    print(
        f'{"Portfolio value":<24}'
        f'{portfolio_value_text:>16}'
    )

    print(
        f'{"Total BUY":<24}'
        f'{GREEN}'
        f'{total_buy_text:>16}'
        f'{RESET}'
    )

    print(
        f'{"Total SELL":<24}'
        f'{RED}'
        f'{total_sell_text:>16}'
        f'{RESET}'
    )

    print_metrics(
        expected_return=expected_return,
        risk=risk,
        sharpe=sharpe
    )


# ============================================================
# Enforce maximum residual weight
# ============================================================

def enforce_max_weight(
    weights,
    max_weight
):

    if max_weight >= 1.0:

        return weights

    weights = (
        weights.copy()
    )

    for _ in range(100):

        excess = np.maximum(
            weights - max_weight,
            0.0
        )

        excess_sum = (
            excess.sum(
                axis=1
            )
        )

        mask = (
            excess_sum
            > 1e-12
        )

        if not np.any(
            mask
        ):

            break

        weights = np.minimum(
            weights,
            max_weight
        )

        capacity = np.maximum(
            max_weight - weights,
            0.0
        )

        capacity_sum = (
            capacity.sum(
                axis=1
            )
        )

        valid = (
            mask
            & (
                capacity_sum
                > 1e-12
            )
        )

        if not np.any(
            valid
        ):

            break

        weights[valid] += (
            capacity[valid]
            * (
                excess_sum[valid]
                / capacity_sum[valid]
            )[:, None]
        )

    weights /= (
        weights.sum(
            axis=1,
            keepdims=True
        )
    )

    return (
        weights
    )


# ============================================================
# Random portfolios
# ============================================================

def generate_random_weights(
    rng,
    n_portfolios,
    n_assets,
    min_weight,
    max_weight
):

    validate_weight_constraints(
        n_assets=n_assets,
        min_weight=min_weight,
        max_weight=max_weight
    )

    # --------------------------------------------------------
    # If every asset must have exactly 1/N, there is only one
    # feasible portfolio.
    # --------------------------------------------------------

    remaining_weight = (
        1.0
        - n_assets * min_weight
    )

    if (
        remaining_weight
        <= 1e-12
    ):

        return np.full(
            (
                n_portfolios,
                n_assets
            ),
            1.0 / n_assets,
            dtype=float
        )

    # --------------------------------------------------------
    # Generate exactly n_portfolios portfolios.
    #
    # Lower alpha values create more points near the
    # boundaries of the feasible region.
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

    gamma_samples = rng.gamma(
        shape=alpha[:, None],
        scale=1.0,
        size=(
            n_portfolios,
            n_assets
        )
    )

    row_sums = (
        gamma_samples.sum(
            axis=1,
            keepdims=True
        )
    )

    invalid_rows = (
        row_sums[:, 0]
        <= 0
    )

    if np.any(
        invalid_rows
    ):

        gamma_samples[
            invalid_rows
        ] = 1.0

        row_sums = (
            gamma_samples.sum(
                axis=1,
                keepdims=True
            )
        )

    residual_weights = (
        gamma_samples
        / row_sums
    )

    # --------------------------------------------------------
    # Transform the original bounds:
    #
    # w_i = min_weight + remaining_weight * z_i
    #
    # with:
    #
    # sum(z_i) = 1
    #
    # The equivalent maximum bound for z_i is:
    #
    # (max_weight - min_weight) / remaining_weight
    # --------------------------------------------------------

    residual_max_weight = (
        (
            max_weight
            - min_weight
        )
        / remaining_weight
    )

    residual_weights = (
        enforce_max_weight(
            residual_weights,
            residual_max_weight
        )
    )

    # --------------------------------------------------------
    # Restore actual portfolio weights.
    #
    # Every asset now satisfies:
    #
    # min_weight <= weight <= max_weight
    # --------------------------------------------------------

    weights = (
        min_weight
        + remaining_weight
        * residual_weights
    )

    return (
        weights
    )


# ============================================================
# Historical coverage warning
# ============================================================

def print_history_coverage_warning(
    stocks,
    tickers,
    requested_start,
    requested_end,
    years
):

    # --------------------------------------------------------
    # Approximate number of trading observations expected for
    # the requested horizon.
    # --------------------------------------------------------

    expected_observations = max(
        1,
        int(
            round(
                TRADING_DAYS
                * years
            )
        )
    )

    warnings = []

    for ticker in tickers:

        series = (
            stocks[ticker]
            .dropna()
        )

        if series.empty:
            continue

        first_timestamp = (
            series.index[0]
        )

        last_timestamp = (
            series.index[-1]
        )

        first_date = (
            first_timestamp.date()
            if hasattr(
                first_timestamp,
                'date'
            )
            else first_timestamp
        )

        last_date = (
            last_timestamp.date()
            if hasattr(
                last_timestamp,
                'date'
            )
            else last_timestamp
        )

        observations = (
            len(series)
        )

        observation_coverage = min(
            observations
            / expected_observations,
            1.0
        )

        start_delay_days = max(
            0,
            (
                first_date
                - requested_start
            ).days
        )

        # ----------------------------------------------------
        # Warn when the series starts materially after the
        # requested date OR has materially fewer observations
        # than expected for the requested number of years.
        #
        # A small calendar tolerance avoids warnings caused by
        # weekends, holidays or a few missing sessions.
        # ----------------------------------------------------

        insufficient_history = (
            start_delay_days
            > HISTORY_START_TOLERANCE_DAYS
            or observation_coverage
            < HISTORY_MIN_OBSERVATION_COVERAGE
        )

        if insufficient_history:

            warnings.append(
                (
                    ticker,
                    first_date,
                    last_date,
                    observations,
                    observation_coverage,
                    start_delay_days
                )
            )

    if not warnings:
        return

    # --------------------------------------------------------
    # Calculate the sample that will actually survive the
    # complete-case alignment used later by stocks.dropna().
    # --------------------------------------------------------

    common_stocks = (
        stocks[
            tickers
        ]
        .dropna()
    )

    common_observations = (
        len(common_stocks)
    )

    common_coverage = min(
        common_observations
        / expected_observations,
        1.0
    )

    if common_observations > 0:

        common_start_timestamp = (
            common_stocks.index[0]
        )

        common_end_timestamp = (
            common_stocks.index[-1]
        )

        common_start = (
            common_start_timestamp.date()
            if hasattr(
                common_start_timestamp,
                'date'
            )
            else common_start_timestamp
        )

        common_end = (
            common_end_timestamp.date()
            if hasattr(
                common_end_timestamp,
                'date'
            )
            else common_end_timestamp
        )

    else:

        common_start = None
        common_end = None

    ticker_width = max(
        len('Ticker'),
        max(
            len(item[0])
            for item in warnings
        )
    )

    first_width = 14
    last_width = 14
    observations_width = 14
    coverage_width = 12

    print()

    print(
        f'{RED}{BOLD}'
        f'{"!" * OUTPUT_WIDTH}'
        f'{RESET}'
    )

    print(
        f'{RED}{BOLD}'
        f'{"WARNING: INCOMPLETE HISTORICAL COVERAGE":^{OUTPUT_WIDTH}}'
        f'{RESET}'
    )

    print(
        f'{RED}{BOLD}'
        f'{"!" * OUTPUT_WIDTH}'
        f'{RESET}'
    )

    print(
        f'{RED}'
        f'Requested with -y {years}: '
        f'{requested_start} -> {requested_end}'
        f'{RESET}'
    )

    print(
        f'{RED}'
        f'Expected observations (approx.): '
        f'{expected_observations:,}'
        f'{RESET}'
    )

    print()

    header = (
        f'{"Ticker":<{ticker_width}}  '
        f'{"First obs":<{first_width}}'
        f'{"Last obs":<{last_width}}'
        f'{"Observations":<{observations_width}}'
        f'{"Coverage":<{coverage_width}}'
    )

    print(
        f'{RED}{header}{RESET}'
    )

    print(
        f'{RED}'
        f'{"-" * len(header)}'
        f'{RESET}'
    )

    for (
        ticker,
        first_date,
        last_date,
        observations,
        observation_coverage,
        start_delay_days

    ) in warnings:

        first_text = (
            str(first_date)
        )

        last_text = (
            str(last_date)
        )

        observations_text = (
            f'{observations:,}'
        )

        coverage_text = (
            f'{100 * observation_coverage:.1f} %'
        )

        print(
            f'{RED}'
            f'{ticker:<{ticker_width}}  '
            f'{first_text:<{first_width}}'
            f'{last_text:<{last_width}}'
            f'{observations_text:<{observations_width}}'
            f'{coverage_text:<{coverage_width}}'
            f'{RESET}'
        )

    print()

    print(
        f'{RED}'
        f'One or more series do not fully cover the requested '
        f'historical period.'
        f'{RESET}'
    )

    if common_observations > 0:

        print(
            f'{RED}'
            f'After aligning all assets, the effective common '
            f'sample will be:'
            f'{RESET}'
        )

        print(
            f'{RED}'
            f'  Period: {common_start} -> {common_end}'
            f'{RESET}'
        )

        print(
            f'{RED}'
            f'  Observations: {common_observations:,} '
            f'({100 * common_coverage:.1f} % of the requested '
            f'approximate trading observations)'
            f'{RESET}'
        )

    else:

        print(
            f'{RED}'
            f'There are no common observations across all assets.'
            f'{RESET}'
        )

    print(
        f'{RED}'
        f'Results will use the common history available after '
        f'alignment.'
        f'{RESET}'
    )

    print(
        f'{RED}{BOLD}'
        f'{"!" * OUTPUT_WIDTH}'
        f'{RESET}'
    )


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
    provided_weights=None
):

    n_assets = (
        len(mu)
    )

    # --------------------------------------------------------
    # Exactly "simulations" portfolios
    # --------------------------------------------------------

    weights = generate_random_weights(
        rng=rng,
        n_portfolios=simulations,
        n_assets=n_assets,
        min_weight=min_weight,
        max_weight=max_weight
    )

    # --------------------------------------------------------
    # Expected returns
    # --------------------------------------------------------

    random_returns = (
        weights @ mu
    )

    # --------------------------------------------------------
    # Variances
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
        (
            random_returns
            - rf
        )
        / random_risks
    )

    # --------------------------------------------------------
    # Maximum Sharpe
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
    # Minimum Variance
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
    # Figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(
            10,
            7
        )
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
    # Maximum Sharpe
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
    # Minimum Variance
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

    if (
        provided_weights
        is not None
    ):

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

    plt.xlabel(
        'Annualized Risk (%)'
    )

    plt.ylabel(
        'Annualized Expected Return (%)'
    )

    plt.title(
        'Markowitz Portfolio Optimization '
        f'- {years} Years'
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
    # Convert weight constraints to decimal
    # --------------------------------------------------------

    min_weight = (
        args.min_weight
        / 100
    )

    max_weight = (
        args.max_weight
        / 100
    )

    # --------------------------------------------------------
    # Validate constraints against the number of assets before
    # downloading market data.
    # --------------------------------------------------------

    try:

        validate_weight_constraints(
            n_assets=len(
                tickers
            ),
            min_weight=min_weight,
            max_weight=max_weight
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
        ticker_source=ticker_source,
        config_path=config_path,
        n_assets=len(
            tickers
        ),
        has_provided_portfolio=(
            provided_values_raw
            is not None
        )
    )

    # --------------------------------------------------------
    # RNG
    # --------------------------------------------------------

    rng = (
        np.random.default_rng(
            args.seed
        )
    )

    # --------------------------------------------------------
    # Historical period
    # --------------------------------------------------------

    end = (
        date.today()
    )

    start = (
        end
        - relativedelta(
            years=args.years
        )
    )

    # --------------------------------------------------------
    # Download prices
    # --------------------------------------------------------

    print()

    print_status(
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
    # Warn before complete-case alignment if one or more series
    # do not sufficiently cover the period requested with -y.
    # --------------------------------------------------------

    print_history_coverage_warning(
        stocks=stocks,
        tickers=tickers,
        requested_start=start,
        requested_end=end,
        years=args.years
    )

    # --------------------------------------------------------
    # Preserve ticker order and common observations
    # --------------------------------------------------------

    stocks = (
        stocks[
            tickers
        ]
        .dropna()
    )

    if (
        stocks.shape[1]
        < 2
    ):

        raise RuntimeError(
            'Could not obtain valid data '
            'for at least two assets.'
        )

    # --------------------------------------------------------
    # Provided portfolio
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

    returns = (
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
            returns,
            axis=0
        )
    )

    # --------------------------------------------------------
    # Annualized covariance
    # --------------------------------------------------------

    covariance = (
        TRADING_DAYS
        * np.cov(
            returns.T
        )
    )

    # --------------------------------------------------------
    # Risk-free rate:
    #
    # percent -> decimal
    # --------------------------------------------------------

    rf = (
        args.rf
        / 100
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    print_status(
        'Optimizing portfolios...'
    )

    (
        w_max_sharpe,
        w_min_var

    ) = optimize_portfolios(
        mu=mu,
        covariance=covariance,
        rf=rf,
        min_weight=min_weight,
        max_weight=max_weight,
        max_iter=args.max_iter
    )

    # ========================================================
    # Maximum Sharpe
    # ========================================================

    print_portfolio(
        title='Maximum Sharpe Portfolio',
        weights=w_max_sharpe,
        tickers=stocks.columns,
        mu=mu,
        covariance=covariance,
        rf=rf
    )

    # ========================================================
    # Minimum Variance
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
    # Individual assets
    # ========================================================

    print_asset_statistics(
        tickers=stocks.columns,
        mu=mu,
        returns=returns
    )

    # ========================================================
    # Provided Portfolio
    # ========================================================

    if (
        provided_weights
        is not None
    ):

        print_provided_portfolio(
            title=(
                'Provided Portfolio -> Maximum Sharpe'
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
            min_weight=min_weight,
            max_weight=max_weight,
            years=args.years,
            provided_weights=provided_weights
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':

    handle_contextual_help()

    main()