import time

import click

from .printers import log_info, log_debug, log_error, log_br, log_success
from .truthtable import TruthTable
from .utils import closed_range
from .version import version as __version__


@click.command(context_settings=dict(
    max_content_width=999,
    help_option_names=['-h', '--help']
))
@click.option('-i', '--truth-table', 'filename_truth_table', metavar='<path>', required=True,
              type=click.Path(exists=True, allow_dash=True),
              help='Input file with truth table')
@click.option('-o', '--output', 'filename_output', metavar='<path>',
              default='-', type=click.Path(writable=True, allow_dash=True),
              help='Output filename for Boolean formula (defaults to stdout)')
@click.option('-Pmin', 'Pmin', metavar='<int>',
              default=1, show_default=True,
              help='Lower bound for P')
@click.option('-Pmax', 'Pmax', metavar='<int>',
              default=15, show_default=True,
              help='Upper bound for P')
@click.option('--sat-solver', metavar='<cmd>',
              default='glucose -model -verb=0', show_default=True,
              # default='cryptominisat5 --verb=0', show_default=True,
              # default='cadical -q', show_default=True,
              help='SAT solver')
@click.option('--solver-type', type=click.Choice(['stream', 'file']),
              default='stream', show_default=True,
              help='[internal] Solver class')
@click.version_option(__version__)
def cli(filename_truth_table, filename_output, Pmin, Pmax, sat_solver, solver_type):
    log_info('Welcome!')
    time_start = time.time()

    truth_table = TruthTable.from_file(filename_truth_table)

    if Pmax >= Pmin:
        for P in closed_range(Pmin, Pmax):
            log_br()
            log_info(f'Trying P = {P}...')
            boolean_formula = truth_table.infer(P, solver_cmd=sat_solver, solver_type=solver_type)
            if boolean_formula:
                log_br()
                log_success(f'Boolean formula: {boolean_formula}')
                break
        else:
            log_br()
            log_error('Can\'t find Boolean formula :c', symbol='-')
    else:
        best = None
        for P in reversed(closed_range(Pmax, Pmin)):
            log_br()
            log_info(f'Trying P = {P}...')
            boolean_formula = truth_table.infer(P, solver_cmd=sat_solver, solver_type=solver_type)
            if boolean_formula:
                best = boolean_formula
            else:
                break
        if best:
            log_br()
            log_success(f'Boolean formula: {boolean_formula}')
        else:
            log_br()
            log_error('Can\'t find Boolean formula :c', symbol='-')

    log_br()
    log_success(f'All done in {time.time() - time_start:.2f} s')
