from types import SimpleNamespace
from .topology import Topology
from .non_bonded import NonBonded

from abc import ABC, abstractmethod
#
import numpy as np
#
from .storage import TermStorage, MultipleTermStorge


class TermABC(ABC):
    """Represents a ForceField term."""

    __slots__ = ('atomids', 'equ', 'idx', 'fconst', '_typename', '_name', 'n_params')

    name = 'NOT_NAMED'

    def __init__(self, atomids: list[int], equ: float, typename: str, n_params: int,
                 fconst: np.ndarray=None):
        """Initialization of a term.

        Keyword arguments
        -----------------
            atomids : list[int]
                List containing the atom IDs which participate in the term
            equ : float
                Equilibrium for the term
            typename : str
                Representation of the atoms involved in the term and their bonds
            n_params : int
                The number of customizable parameters/constants which the term involves
            fconst : np.ndarray[float](n_params,) (default None)
                Values for the term constants
        """
        self.atomids = np.array(atomids)
        self.equ = equ
        self.idx = 0
        self.fconst = fconst
        self.n_params = n_params
        self.typename = typename
        self._name = f"{self.name}({typename})"

    def __repr__(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name

    def set_idx(self, idx: int) -> None:
        """Set a new ID for the term.

        Keyword arguments
        -----------------
            idx : int
                The new ID

        Returns
        -------
            None
        """
        self.idx = idx

    def do_force(self, crd: np.ndarray, force: np.ndarray) -> float:
        """Force calculation with given geometry."""
        return self._calc_forces(crd, force, self.fconst)

    def do_fitting(self, crd: np.ndarray, forces: np.ndarray, index: int=0,
                   params: np.ndarray=None) -> None:
        """Compute fitting contributions."""
        if params is None:  # Linear least squares
            self._calc_forces(crd, forces[self.idx], np.ones(self.n_params))
        else:  # Non-linear least squares
            self._calc_forces(crd, forces[self.idx], params[index:index+self.n_params])

    @abstractmethod
    def _calc_forces(self, crd: np.ndarray, force: np.ndarray, fconst: np.ndarray) -> float:
        """Perform actual force computation"""

    @classmethod
    def get_terms_container(cls):
        return TermStorage(cls.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self._name
        if isinstance(other, TermABC):
            return str(other) == self._name
        else:
            raise Exception("Cannot compare Term with")

    def __ne__(self, other):
        if isinstance(other, str):
            return other != self._name
        if isinstance(other, TermABC):
            return str(other) != self._name
        else:
            raise Exception("Cannot compare Term with")


class TermFactory(ABC):
    """Factory class to create ForceField Terms of one ore multiple TermABC classes"""

    _term_types = None
    _multiple_terms = True
    name = "NAME_NOT_DEFINED"

    @classmethod
    def get_terms_container(cls):
        if cls._multiple_terms is False:
            return TermStorage(cls.name)
        return MultipleTermStorge(cls.name, {key: value.get_terms_container()
                                             for key, value in cls._term_types.items()})

    @classmethod
    @abstractmethod
    def get_terms(cls, topo: Topology, non_bonded: NonBonded,
                  config: SimpleNamespace) -> list:
        """Get the terms of the current class based on molecule data.

        Keyword arguments
        -----------------
            topo : Topology
                Stores all topology information
            non_bonded : NonBonded object
                Stores all non bonded interaction information
            config : SimpleNamespace
                Stores the global config settings

        Returns
        -------
            list of cls objects
        """
        ...


class TermBase(TermFactory, TermABC):
    """Base class for terms that are TermFactories for themselves as well"""
    _multiple_terms = False
