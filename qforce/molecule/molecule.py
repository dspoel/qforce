from types import SimpleNamespace
from ..qm.qm_base import HessianOutput

from .topology import Topology
from .terms import Terms
from .non_bonded import NonBonded


class Molecule(object):
    """Class to represent a molecule."""

    def __init__(self, config: SimpleNamespace, job: SimpleNamespace, qm_out: HessianOutput,
                 ext_q=None, ext_lj=None):
        """Create a new instace of Molecule.

        Keyword arguments
        -----------------
            config : SimpleNamespace
                The global config data structure
            job : SimpleNamespace
                Data structure with job information
            qm_out : HessianOutput
                Output of the hessian file reading
            ext_q :
            ext_lj :
        """
        self.name = job.name
        self.elements = qm_out.elements
        self.charge = qm_out.charge
        self.multiplicity = qm_out.multiplicity
        self.n_atoms = len(self.elements)
        self.topo = Topology(config.ff, qm_out)
        self.non_bonded = NonBonded.from_topology(config.ff, job, qm_out, self.topo, ext_q, ext_lj)
        self.terms = Terms.from_topology(config, self.topo, self.non_bonded)
