import sys
from types import SimpleNamespace

from .polarize import polarize
from .initialize import initialize
from .qm.qm import QM
from .qm.qm_base import HessianOutput
from .forcefield import ForceField
from .molecule import Molecule
from .fragment import fragment
from .dihedral_scan import DihedralScan
from .frequencies import calc_qm_vs_md_frequencies
from .hessian import fit_hessian, fit_hessian_nl
from .misc import check_continue, print_phase_header, check_wellposedness,\
    PHASES, add_job_dir_to_json_name


def run_qforce(input_arg: str, ext_q=None, ext_lj=None, config: str=None, pinput: str=None,
               psave: str=None, process_file: str=None, presets=None) -> None:
    """Start a Q-Force run.

    Keyword arguments
    -----------------
        input_arg : str
            the input coordinate file, or directory
        ext_q : (default None)
        ext_lj : (default None)
        config : str (default None)
            the path to the config file
        pinput : str (default None)
            the name of the input params file within the <molecule_name>_qforce directory
        psave : str (default None)
            the name of the output params file within the <molecule_name>_qforce directory
        process_file : str (default None)
            the name of the output process file within the <molecule_name>_qforce directory
        presets : (default None)

    Returns
    -------
        None
    """
    # Phase count
    pc = 0

    #### Initialization phase ####
    print_phase_header(PHASES[pc])
    config, job = initialize(input_arg, config, presets)
    print('Config:')
    print(config, '\n')
    print('Job:')
    print(job, '\n')
    pinput = add_job_dir_to_json_name(job.dir, pinput)
    print(f'pinput path: {pinput}')
    psave = add_job_dir_to_json_name(job.dir, psave)
    print(f'psave path: {psave}')
    process_file = add_job_dir_to_json_name(job.dir, process_file)
    print(f'process_file path: {process_file}')

    check_wellposedness(config)

    check_continue(config, PHASES[pc], PHASES[pc+1])
    pc += 1

    #### Polarization phase ####
    print_phase_header(PHASES[pc])
    if config.ff._polarize:
        polarize(job, config.ff)

    check_continue(config, PHASES[pc], PHASES[pc+1])
    pc += 1

    #### QM phase ####
    print_phase_header(PHASES[pc])
    qm = QM(job, config.qm)
    qm_hessian_out = qm.read_hessian()

    check_continue(config, PHASES[pc], PHASES[pc+1])
    pc += 1

    #### Molecule phase ####
    print_phase_header(PHASES[pc])
    mol = Molecule(config, job, qm_hessian_out, ext_q, ext_lj)

    check_continue(config, PHASES[pc], PHASES[pc+1])
    pc += 1

    #### Hessian fitting phase ####
    print_phase_header(PHASES[pc])
    md_hessian = None
    if config.opt.fit_type == 'linear':
        md_hessian = fit_hessian(config, mol, qm_hessian_out, psave)
    elif config.opt.fit_type == 'non_linear':
        md_hessian = fit_hessian_nl(config, mol, qm_hessian_out, pinput, psave, process_file)

    check_continue(config, PHASES[pc], PHASES[pc+1])
    pc += 1

    #### Flexible dihedral scan phase ####
    if config.ff.scan_dihedrals:
        print_phase_header(PHASES[pc])
        if len(mol.terms['dihedral/flexible']) > 0 and config.scan.do_scan:
            fragments = fragment(mol, qm, job, config)
            DihedralScan(fragments, mol, job, config)

        check_continue(config, PHASES[pc], PHASES[pc+1])
    pc += 1

    #### Calculate frequencies phase ####
    print_phase_header(PHASES[pc])
    calc_qm_vs_md_frequencies(job, qm_hessian_out, md_hessian)

    check_continue(config, PHASES[pc], PHASES[pc+1])
    pc += 1

    #### Calculate Force Field phase ####
    if config.ff.compute_ff:
        print_phase_header(PHASES[pc])
        ff = ForceField(job.name, config, mol, mol.topo.neighbors)
        ff.write_gromacs(job.dir, mol, qm_hessian_out.coords)

        print_outcome(job.dir)
    else:
        print('\nNo Force Field output requested...\n')

    print('\n\nRUN COMPLETED')


def run_hessian_fitting_for_external(job_dir, qm_data, ext_q=None, ext_lj=None,
                                     config=None, presets=None):
    config, job = initialize(job_dir, config, presets)

    qm_hessian_out = HessianOutput(config.qm.vib_scaling, **qm_data)

    mol = Molecule(config, job, qm_hessian_out, ext_q, ext_lj)

    md_hessian = fit_hessian(config.terms, mol, qm_hessian_out)
    calc_qm_vs_md_frequencies(job, qm_hessian_out, md_hessian)

    ff = ForceField(job.name, config, mol, mol.topo.neighbors)
    ff.write_gromacs(job.dir, mol, qm_hessian_out.coords)

    print_outcome(job.dir)

    return mol.terms


def print_outcome(job_dir: str) -> None:
    """Print final coutcome of the run.

    Keyword arguments
    -----------------
        job_dir : str
            Working directory, <molecule_name>_qforce

    Returns
    -------
        None
    """
    print(f'Output files can be found in the directory: {job_dir}.')
    print('- Q-Force force field parameters in GROMACS format (gas.gro, gas.itp, gas.top).')
    print('- QM vs MM vibrational frequencies, pre-dihedral fitting (frequencies.txt,'
          ' frequencies.pdf).')
    print('- Vibrational modes which can be visualized in VMD (frequencies.nmd).')
    print('- QM vs MM dihedral profiles (if any) in "fragments" folder as ".pdf" files.')
