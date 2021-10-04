import os
import sys
from types import SimpleNamespace

LOGO = """
          ____         ______
         / __ \       |  ____|
        | |  | |______| |__ ___  _ __ ___ ___
        | |  | |______|  __/ _ \| '__/ __/ _ \\
        | |__| |      | | | (_) | | | (_|  __/
         \___\_\      |_|  \___/|_|  \___\___|

                     Selim Sami
            University of Groningen - 2020
            ==============================
"""

LOGO_SEMICOL = """
;          ____         ______
;         / __ \       |  ____|
;        | |  | |______| |__ ___  _ __ ___ ___
;        | |  | |______|  __/ _ \| '__/ __/ _ \\
;        | |__| |      | | | (_) | | | (_|  __/
;         \___\_\      |_|  \___/|_|  \___\___|
;
;                     Selim Sami
;            University of Groningen - 2020
;            ==============================
"""

PHASES = ['Initialization', 'Polarization', 'QM', 'Molecule', 'Hessian Fitting',
          'Flexible Dihedral Scan', 'Calculate Frequencies', 'Calculate Force Field']


def check_if_file_exists(file: str) -> str:
    if not os.path.exists(file) and not os.path.exists(f'{file}_qforce'):
        sys.exit(f'ERROR: "{file}" does not exist.\n')
    return file

def print_phase_header(phase: str) -> None:
    print(f'\n#### {phase.upper()} PHASE ####\n')

def check_continue(config: SimpleNamespace, prev: str, next: str) -> None:
    if config.general.debug_mode:
        if prev and next:
            print(f'\n{prev.upper()} phase completed. Next up: {next.upper()} phase.')
        x = input('\nDo you want to continue y/n? ')
        if x not in ['yes', 'y', '']:
            print()
            sys.exit(0)
