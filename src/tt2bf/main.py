import time

import click

from .printers import log_info, log_debug, log_error, log_br, log_success
from .truthtable import TruthTable
from .utils import closed_range
from .version import version as __version__

CONTEXT_SETTINGS = dict(
    max_content_width=999,
    help_option_names=['-h', '--help']
)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--truth-table', 'filename_truth_table', metavar='<path>', required=True,
              type=click.Path(exists=True, allow_dash=True),
              help='Input file with truth table')
@click.option('-o', '--output', 'filename_output', metavar='<path>',
              default='-', type=click.Path(writable=True, allow_dash=True),
              help='Output filename for Boolean formula (defaults to stdout)')
@click.option('--sat-solver', metavar='<cmd>',
              default='glucose -model -verb=0', show_default=True,
              # default='cryptominisat5 --verb=0', show_default=True,
              # default='cadical -q', show_default=True,
              help='SAT solver')
def cli(filename_truth_table, filename_output, sat_solver):
    log_info('Welcome!')
    time_start = time.time()

    truth_table = TruthTable.from_file(filename_truth_table)

    for P in closed_range(1, 5):
        log_br()
        log_info(f'Trying P = {P}...')
        boolean_formula = truth_table.infer(P, solver_cmd=sat_solver)
        if boolean_formula:
            log_success(f'Boolean formula: {boolean_formula}')
            break
    else:
        log_error('Can\'t find Boolean formula :c')

    log_br()
    log_success(f'All done in {time.time() - time_start:.2f} s')
