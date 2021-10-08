import os
import sys
import re
from types import SimpleNamespace
from typing import Union


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


def check_if_file_exists(file: str) -> Union[str, None]:
    """Check if a file exists.
    If the file does not exist, the execution will halt.

    Keyword arguments
    -----------------
        file : str
            the file path

    Returns
    -------
        the file path (str) if the file exists, otherwise None
    """
    if not os.path.exists(file) and not os.path.exists(f'{file}_qforce'):
        sys.exit(f'ERROR: "{file}" does not exist.\n')
    return file


def print_phase_header(phase: str) -> None:
    """Print a header for an execution phase.

    Keyword arguments
    -----------------
        phase : str
            the phase name

    Returns
    -------
        None
    """
    print(f'\n#### {phase.upper()} PHASE ####\n')


def check_wellposedness(config: SimpleNamespace) -> None:
    """Check the global config for inconsistencies.
    If an inconsistency is detected, the excecution is halted.

    Keyword arguments
    -----------------
        config : SimpleNamespace
            the global config settings

    Returns
    -------
        None
    """
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


def check_continue(config: SimpleNamespace, prev: str=None, next: str=None) -> None:
    """Check if the user wants to continue with the run.
    The user will be asked to provide input. If y, yes, or the return key is
    pressed directly, the program will continue; otherwise, it will immediately stop.

    Keyword arguments
    -----------------
        config : SimpleNamespace
            the global config settings
        prev : str (default None)
            the previous phase in the run
        next : str (default None)
            the upcoming phase in the run

    Returns
    -------
        None
    """
    if config.general.debug_mode:
        if prev and next:
            print(f'\n{prev.upper()} phase completed. Next up: {next.upper()} phase.')
        x = input('\nDo you want to continue y/n? ')
        if x not in ['yes', 'y', '']:
            print()
            sys.exit(0)


def term_name_key(name_tuple: tuple[str, list[float]]) -> tuple[str, str]:
    """Return key for sorting terms when exporting to JSON after hessian fitting

    Keyword arguments
    -----------------
        name_tuple : tuple[str, list[float]]
            a tuple containing the name of the term given by the __str__ method first
            and the list of parameters second

    Returns
    -------
        tuple[str, str, int] containing the raw term type first, involved atoms second
        (the info in parenthesis given by __str__) and the term order third
    """
    t_type, atoms = name_tuple[0].split('(', 1)
    atoms = atoms[:-1]  # Delete the last closing parenthesis
    t_type_words = t_type.rstrip('0123456789')  # In case of polyterm, delete the numbers in the end
    match = re.match(r"([a-zA-Z]+)([0-9]+)", t_type, re.I)
    t_type_numbers = int(match.groups()[-1]) if match else ''
    return t_type_words, atoms, t_type_numbers


def add_job_dir_to_json_name(job_dir: str, path: str) -> Union[str, None]:
    """Add the job directory (<molecule_name>_qforce) to a given json file name.

    Keyword arguments
    -----------------
        job_dir : str
            the job directory
        path : str
            the name of the json file (without extension)

    Returns
    -------
        The composite path (str) if "path" is not None, else None
    """
    return job.dir + '/' + path + '.json' if path is not None else None