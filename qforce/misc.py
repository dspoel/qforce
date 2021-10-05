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

def check_wellposedness(config: SimpleNamespace) -> None:
    if config.opt.fit_type == 'linear' and (config.terms.morse or config.terms.morse_mp):
        raise Exception('Linear optimization is not valid for Morse bond potential')
    elif (config.terms.morse and config.terms.morse_mp) or (config.terms.morse and config.terms.morse_mp2):
        raise Exception('Morse and Morse MP bonds cannot be used at the same time')
    elif config.terms.angle and config.terms.poly_angle:
        raise Exception('Harmonic angle cannot be used at the same time than Poly Angle terms')
    elif config.terms.morse_mp and config.terms.morse_mp2:
        raise Exception('Cannot run two versions of Morse MP at the same time')
    elif config.opt.noise < 0 or config.opt.noise > 1:
        raise Exception('Noise must be in range [0, 1]')
    elif config.ff.compute_ff and not config.ff.scan_dihedrals:
        raise Exception('For FF computation, dihedral scan is necessary.\nPlease set'
                        ' "scan_dihedrals = True" within [ff] in your settings file.')
    else:
        print('Configuration is valid!')

def check_continue(config: SimpleNamespace, prev: str, next: str) -> None:
    if config.general.debug_mode:
        if prev and next:
            print(f'\n{prev.upper()} phase completed. Next up: {next.upper()} phase.')
        x = input('\nDo you want to continue y/n? ')
        if x not in ['yes', 'y', '']:
            print()
            sys.exit(0)
