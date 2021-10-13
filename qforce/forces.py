import numpy as np
import math
from numba import jit
"""
    Calculation of forces on x, y, z directions and also seperately on
    term_ids. So forces are grouped seperately for each unique FF
    parameter.
"""


@jit(nopython=True)
def calc_bonds(coords: np.ndarray, atoms: np.ndarray, r0: float, fconst: np.ndarray,
               force: np.ndarray) -> float:
    """Calculate the potential energy and forces caused by a harmonic bond.

    Keyword arguments
    -----------------
        coords : np.ndarray[float](n_atoms, 3)
            XYZ coordinates for every atom in the molecule
        atoms : np.ndarray[int](2,)
            Atom IDs involved in the bond
        r0 : float
            Bond equilibrium length
        fconst : np.ndarray[float](1,)
            Constant for the potential term
        force : np.ndarray[float](n_atoms, 3)
            Stores the XYZ forces exerted in every atom by the term which is calling
            this function

    Returns
    -------
        The potential energy (float)
    """
    vec12, r12 = get_dist(coords[atoms[0]], coords[atoms[1]])
    # energy = 0.5 * fconst[0] * (r12-r0)**2
    energy = 0.5 * fconst[0] * (r12 - r0)*(r12 - r0)
    f = - fconst[0] * vec12 * (r12-r0) / r12
    force[atoms[0]] += f
    force[atoms[1]] -= f
    return energy


@jit(nopython=True)
def calc_morse(coords: np.ndarray, atoms: np.ndarray, r0: float, fconst: np.ndarray,
               force: np.ndarray, well_depth: float) -> float:
    """Calculate the potential energy and forces caused by a Morse bond.

    Keyword arguments
    -----------------
        coords : np.ndarray[float](n_atoms, 3)
            XYZ coordinates for every atom in the molecule
        atoms : np.ndarray[int](2,)
            Atom IDs involved in the bond
        r0 : float
            Bond equilibrium length
        fconst : np.ndarray[float](1,)
            Constant for the potential term
        force : np.ndarray[float](n_atoms, 3)
            Stores the XYZ forces exerted in every atom by the term which is calling
            this function
        well_depth : float
            the well depth of the current bond type

    Returns
    -------
        The potential energy (float)
    """
    vec12, r12 = get_dist(coords[atoms[0]], coords[atoms[1]])
    beta = math.sqrt(fconst[0]/(2*well_depth))
    exp_term = math.exp(-beta*(r12-r0))
    energy = well_depth * (1-exp_term) * (1-exp_term)
    f = -2 * well_depth * beta * exp_term * (1 - exp_term) * vec12 / r12
    force[atoms[0]] += f
    force[atoms[1]] -= f
    return energy

@jit(nopython=True)
def calc_morse_mp(coords: np.ndarray, atoms: np.ndarray, r0: float, fconst: np.ndarray,
                  force: np.ndarray) -> float:
    """Calculate the potential energy and forces caused by a Morse multi-parameter bond.

    Keyword arguments
    -----------------
        coords : np.ndarray[float](n_atoms, 3)
            XYZ coordinates for every atom in the molecule
        atoms : np.ndarray[int](2,)
            Atom IDs involved in the bond
        r0 : float
            Bond equilibrium length
        fconst : np.ndarray[float](2,)
            Constants for the potential term. First the well depth, next beta_ij
        force : np.ndarray[float](n_atoms, 3)
            Stores the XYZ forces exerted in every atom by the term which is calling
            this function

    Returns
    -------
        The potential energy (float)
    """
    # fconst[0] -> well depth
    # fconst[1] -> beta
    vec12, r12 = get_dist(coords[atoms[0]], coords[atoms[1]])
    exp_term = math.exp(-fconst[1]*(r12-r0))
    energy = fconst[0] * (1-exp_term) * (1-exp_term)
    f = -2 * fconst[0] * fconst[1] * exp_term * (1 - exp_term) * vec12 / r12
    force[atoms[0]] += f
    force[atoms[1]] -= f
    return energy


@jit(nopython=True)
def calc_morse_mp2(coords: np.ndarray, atoms: np.ndarray, r0: float, fconst: np.ndarray,
                   force: np.ndarray) -> float:
    """Calculate the potential energy and forces caused by a Morse multi-parameter-v2 bond.

    Keyword arguments
    -----------------
        coords : np.ndarray[float](n_atoms, 3)
            XYZ coordinates for every atom in the molecule
        atoms : np.ndarray[int](2,)
            Atom IDs involved in the bond
        r0 : float
            Bond equilibrium length
        fconst : np.ndarray[float](2,)
            Constants for the potential term. First the well depth, next k_ij
        force : np.ndarray[float](n_atoms, 3)
            Stores the XYZ forces exerted in every atom by the term which is calling
            this function

    Returns
    -------
        The potential energy (float)
    """
    # fconst[0] -> well depth
    # fconst[1] -> k_ij
    vec12, r12 = get_dist(coords[atoms[0]], coords[atoms[1]])
    beta = math.sqrt(fconst[1]/(2*fconst[0]))
    exp_term = math.exp(-beta*(r12-r0))
    energy = fconst[0] * (1-exp_term) * (1-exp_term)
    f = -2 * fconst[0] * beta * exp_term * (1 - exp_term) * vec12 / r12
    force[atoms[0]] += f
    force[atoms[1]] -= f
    return energy


@jit(nopython=True)
def calc_angles(coords: np.ndarray, atoms: np.ndarray, theta0: float, fconst: np.ndarray,
                force: np.ndarray) -> float:
    """Calculate the potential energy and forces caused by a harmonic angle.

    Keyword arguments
    -----------------
        coords : np.ndarray[float](n_atoms, 3)
            XYZ coordinates for every atom in the molecule
        atoms : np.ndarray[int](3,)
            Atom IDs involved in the bond
        theta0 : float
            Angle of equilibrium in radians
        fconst : np.ndarray[float](1,)
            Constant for the potential term
        force : np.ndarray[float](n_atoms, 3)
            Stores the XYZ forces exerted in every atom by the term which is calling
            this function

    Returns
    -------
        The potential energy (float)
    """
    theta, vec12, vec32, r12, r32 = get_angle(coords[atoms])
    cos_theta = math.cos(theta)
    # cos_theta_sq = cos_theta**2
    cos_theta_sq = cos_theta * cos_theta
    dtheta = theta - theta0
    # energy = 0.5 * fconst[0] * dtheta**2
    energy = 0.5 * fconst[0] * dtheta*dtheta
    if cos_theta_sq < 1:
        st = - fconst[0] * dtheta / np.sqrt(1. - cos_theta_sq)
        sth = st * cos_theta
        c13 = st / r12 / r32
        c11 = sth / r12 / r12
        c33 = sth / r32 / r32

        f1 = c11 * vec12 - c13 * vec32
        f3 = c33 * vec32 - c13 * vec12
        force[atoms[0]] += f1
        force[atoms[2]] += f3
        force[atoms[1]] -= f1 + f3
    return energy

@jit(nopython=True)
def calc_poly_angles(coords: np.ndarray, atoms: np.ndarray, theta0: float, fconst: np.ndarray,
                     force: np.ndarray, order: int) -> float:
    """Calculate the potential energy and forces caused by a polynomial-harmonic angle.

    Keyword arguments
    -----------------
        coords : np.ndarray[float](n_atoms, 3)
            XYZ coordinates for every atom in the molecule
        atoms : np.ndarray[int](3,)
            Atom IDs involved in the bond
        theta0 : float
            Angle of equilibrium in radians
        fconst : np.ndarray[float](1,)
            Constant for the potential term
        force : np.ndarray[float](n_atoms, 3)
            Stores the XYZ forces exerted in every atom by the term which is calling
            this function
        order : int
            the order of the term (2 makes it a regular harmonic angle)

    Returns
    -------
        The potential energy (float)
    """
    theta, vec12, vec32, r12, r32 = get_angle(coords[atoms])
    cos_theta = math.cos(theta)
    # cos_theta_sq = cos_theta**2
    cos_theta_sq = cos_theta * cos_theta
    dtheta = theta - theta0
    energy = fconst[0] * dtheta**order
    if cos_theta_sq < 1 and order > 0:
        st = - order * fconst[0] * (dtheta**(order-1)) / np.sqrt(1. - cos_theta_sq)
        sth = st * cos_theta
        c13 = st / r12 / r32
        c11 = sth / r12 / r12
        c33 = sth / r32 / r32

        f1 = c11 * vec12 - c13 * vec32
        f3 = c33 * vec32 - c13 * vec12
        force[atoms[0]] += f1
        force[atoms[2]] += f3
        force[atoms[1]] -= f1 + f3
    return energy

def calc_cross_bond_bond(coords: np.ndarray, atoms: np.ndarray, r0s: np.ndarray,
                         fconst: np.ndarray, force: np.ndarray) -> float:
    """Calculate the potential energy and forces caused by a Bond-Bond cross-term.

    Keyword arguments
    -----------------
        coords : np.ndarray[float](n_atoms, 3)
            XYZ coordinates for every atom in the molecule
        atoms : np.ndarray[int](3,)
            Atom IDs involved in the bond
        r0s : np.ndarray[float](2,)
            Equilibrium bond lengths
        fconst : np.ndarray[float](1,)
            Constant for the potential term
        force : np.ndarray[float](n_atoms, 3)
            Stores the XYZ forces exerted in every atom by the term which is calling
            this function

    Returns
    -------
        The potential energy (float)
    """
    vec12, r12 = get_dist(coords[atoms[0]], coords[atoms[1]])
    vec32, r32 = get_dist(coords[atoms[2]], coords[atoms[1]])

    s1 = r12 - r0s[0]
    s2 = r32 - r0s[1]

    energy = fconst[0] * s1 * s2

    f1 = - fconst[0] * s2 * vec12 / r12
    f3 = - fconst[0] * s2 * vec32 / r32  # Verify this since the instructions on GROMACS are somewhat unclear

    force[atoms[0]] += f1
    force[atoms[2]] += f3
    force[atoms[1]] -= f1 + f3

    return energy

def calc_cross_bond_angle(coords: np.ndarray, atoms: np.ndarray, r0s: np.ndarray,
                          fconst: np.ndarray, force: np.ndarray) -> float:
    """Calculate the potential energy and forces caused by a Bond-Angle cross-term.

    Keyword arguments
    -----------------
        coords : np.ndarray[float](n_atoms, 3)
            XYZ coordinates for every atom in the molecule
        atoms : np.ndarray[int](3,)
            Atom IDs involved in the bond
        r0s : np.ndarray[float](3,)
            Equilibrium bond lengths
        fconst : np.ndarray[float](1,)
            Constant for the potential term
        force : np.ndarray[float](n_atoms, 3)
            Stores the XYZ forces exerted in every atom by the term which is calling
            this function

    Returns
    -------
        The potential energy (float)
    """
    vec12, r12 = get_dist(coords[atoms[0]], coords[atoms[1]])
    vec32, r32 = get_dist(coords[atoms[2]], coords[atoms[1]])
    vec13, r13 = get_dist(coords[atoms[0]], coords[atoms[1]])

    s1 = r12 - r0s[0]
    s2 = r32 - r0s[1]
    s3 = r13 - r0s[2]

    energy = fconst[0] * s3 * (s1+s2)

    k1 = - fconst[0] * s3/r12
    k2 = - fconst[0] * s3/r32
    k3 = - fconst[0] * (s1+s2)/r13

    f1 = k1*vec12 + k3*vec13
    f3 = k2*vec32 + k3*vec13

    force[atoms[0]] += f1
    force[atoms[2]] += f3
    force[atoms[1]] -= f1 + f3
    return energy


def calc_imp_diheds(coords: np.ndarray, atoms: np.ndarray, phi0: float, fconst: np.ndarray,
                    force: np.ndarray) -> float:
    """Calculate potential energy and force caused by an improper dihedral.

    Keyword arguments
    -----------------
        coords : np.ndarray[float](n_atoms, 3)
            XYZ coordinates for every atom in the molecule
        atoms : np.ndarray[int](4,)
            Atom IDs involved in the dihedral
        phi0 : float
            Equilibrium dihedral angle in radians
        fconst : np.ndarray[float](1,)
            Constant for the potential term
        force : np.ndarray[float](n_atoms, 3)
            Stores the XYZ forces exerted in every atom by the term which is calling
            this function

    Returns
    -------
        The potential energy (float)
    """
    phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(coords[atoms])
    dphi = phi - phi0
    dphi = np.pi - (dphi + np.pi) % (2 * np.pi)  # dphi between -pi to pi
    # energy = 0.5 * fconst[0] * dphi**2
    energy = 0.5 * fconst[0] * dphi*dphi
    ddphi = - fconst[0] * dphi
    force = calc_dih_force(force, atoms, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi)
    return energy


@jit(nopython=True)
def calc_rb_diheds(coords: np.ndarray, atoms: np.ndarray, params: np.ndarray, fconst: np.ndarray,
                   force: np.ndarray) -> float:
    """Calculate potential energy and force caused by a flexible dihedral.

    Keyword arguments
    -----------------
        coords : np.ndarray[float](n_atoms, 3)
            XYZ coordinates for every atom in the molecule
        atoms : np.ndarray[int](4,)
            Atom IDs involved in the dihedral
        params : np.ndarray[float](6,)
            Parameters for the term, in increasing order
        fconst : np.ndarray[float](?,)
            Constant for the potential term? Alredy encapsulated in <params>
        force : np.ndarray[float](n_atoms, 3)
            Stores the XYZ forces exerted in every atom by the term which is calling
            this function

    Returns
    -------
        The potential energy (float)
    """
    phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(coords[atoms])
    phi += np.pi
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    energy = params[0]
    ddphi = 0
    cos_factor = 1

    for i in range(1, 6):
        ddphi += i * cos_factor * params[i]
        cos_factor *= cos_phi
        energy += cos_factor * params[i]

    ddphi *= - sin_phi
    force = calc_dih_force(force, atoms, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi)
    return energy


@jit(nopython=True)
def calc_inversion(coords: np.ndarray, atoms: np.ndarray, phi0: float, fconst: np.ndarray,
                   force: np.ndarray) -> float:
    """Calculate potential energy and force caused by an inversion dihedral.

    Keyword arguments
    -----------------
        coords : np.ndarray[float](n_atoms, 3)
            XYZ coordinates for every atom in the molecule
        atoms : np.ndarray[int](4,)
            Atom IDs involved in the dihedral
        phi0 : float
            Equilibrium dihedral angle in radians
        fconst : np.ndarray[float](1,)
            Constant for the potential term
        force : np.ndarray[float](n_atoms, 3)
            Stores the XYZ forces exerted in every atom by the term which is calling
            this function

    Returns
    -------
        The potential energy (float)
    """
    phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(coords[atoms])
    phi += np.pi

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    c0, c1, c2 = convert_to_inversion_rb(fconst, phi0)

    energy = c0

    ddphi = c1
    energy += cos_phi * c1

    ddphi += 2 * c2 * cos_phi
    # energy += cos_phi**2 * c2
    energy += cos_phi*cos_phi * c2

    ddphi *= - sin_phi
    force = calc_dih_force(force, atoms, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi)
    return energy


@jit(nopython=True)
def calc_periodic_dihed(coords, atoms, phi0, fconst, force):
    """Calculate potential energy and force caused by a periodic
     dihedral.

    Keyword arguments
    -----------------
        coords : np.ndarray[float](n_atoms, 3)
            XYZ coordinates for every atom in the molecule
        atoms : np.ndarray[int](4,)
            Atom IDs involved in the dihedral
        phi0 : float
            Equilibrium dihedral angle in radians
        fconst : np.ndarray[float](1,)
            Constant for the potential term
        force : np.ndarray[float](n_atoms, 3)
            Stores the XYZ forces exerted in every atom by the term which is calling
            this function

    Returns
    -------
        The potential energy (float)
    """
    phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(coords[atoms])
    mult = 3
    phi0 = 0

    mdphi = mult * phi - phi0
    ddphi = fconst[0] * mult * np.sin(mdphi)

    energy = fconst[0] * (1 + np.cos(mdphi))

    force = calc_dih_force(force, atoms, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi)
    return energy


@jit(nopython=True)
def convert_to_inversion_rb(fconst, phi0):
    cos_phi0 = np.cos(phi0)
    # c0 = fconst[0] * cos_phi0**2
    c0 = fconst[0] * cos_phi0*cos_phi0
    c1 = 2 * fconst[0] * cos_phi0
    c2 = fconst[0]
    return c0, c1, c2


@jit("f8(f8[:], f8[:])", nopython=True)
def dot_prod(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the dot project between two XYZ vectors.

    Keyword arguments
    -----------------
        a : np.ndarray[float](3,)
        b : np.ndarray[float](3,)

    Returns
    -------
        The dot product result (float)
    """
    x = a[0]*b[0]
    y = a[1]*b[1]
    z = a[2]*b[2]
    return x+y+z


@jit("void(f8[:,:], i8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8)",
     nopython=True)
def calc_dih_force(force, a, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi) -> None:
    """Calculate the force exerted by a dihedral.

    Keyword arguments
    ----------------
        force : np.ndarray[float](n_atoms, 3)
            Stores the XYZ forces exerted in every atom by the term which is calling
            this function
        a : np.ndarray[int](4,)
            Atom IDs involved in the dihedral
        vec_ij : np.ndarray[float](3,)
            XYZ vector pointing from j to i
        vec_kj : np.ndarray[float](3,)
            XYZ vector pointing from j to k
        vec_kl : np.ndarray[float](3,)
            XYZ vector pointing from l to k
        cross1 : np.ndarray[float](3,)
            XYZ vector which is the result of cross_prod(<vec_ij>, <vec_kj>)
        cross2 : np.ndarray[float](3,)
            XYZ vector which is the result of cross_prod(<vec_kj>, <vec_kl>)
        ddphi : float
            k_{ijkl}^{\gamma} * (\phi_{ijkl} - \phi_{ijkl}^{0})

    Returns
    -------
        None
    """
    inner1 = dot_prod(cross1, cross1)
    inner2 = dot_prod(cross2, cross2)
    nrkj2 = dot_prod(vec_kj, vec_kj)

    nrkj_1 = 1 / np.sqrt(nrkj2)
    nrkj_2 = nrkj_1 * nrkj_1
    nrkj = nrkj2 * nrkj_1
    aa = -ddphi * nrkj / inner1
    f_i = aa * cross1
    bb = ddphi * nrkj / inner2
    f_l = bb * cross2
    p = dot_prod(vec_ij, vec_kj) * nrkj_2
    q = dot_prod(vec_kl, vec_kj) * nrkj_2
    uvec = p * f_i
    vvec = q * f_l
    svec = uvec - vvec

    f_j = f_i - svec
    f_k = f_l + svec

    force[a[0]] += f_i
    force[a[1]] -= f_j
    force[a[2]] -= f_k
    force[a[3]] += f_l


@jit(nopython=True)
def calc_pairs(coords, atoms, params, force):
    c6, c12, qq = params
    vec, r = get_dist(coords[atoms[0]], coords[atoms[1]])
    qq_r = qq/r
    # r_2 = 1/r**2
    r_2 = 1 / (r*r)
    # r_6 = r_2**3
    r_6 = r_2*r_2*r_2
    c6_r6 = c6 * r_6
    # c12_r12 = c12 * r_6**2
    c12_r12 = c12 * r_6*r_6
    energy = qq_r + c12_r12 - c6_r6
    f = (qq_r + 12*c12_r12 - 6*c6_r6) * r_2
    fk = f * vec
    force[atoms[0]] += fk
    force[atoms[1]] -= fk
    return energy


@jit(nopython=True)
def get_dist(coord1: np.ndarray, coord2: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute euclidean (L2) distance between two XYZ coordinates.

    Keyword arguments
    -----------------
        coord1 : np.ndarray[float](3,)
        coord2 : np.ndarray[float](3,)

    Returns
    -------
        A vector 'pointing' from <coord2> to <coord1>
        The L2 distance between <coord1> and <coord2>
    """
    vec = coord1 - coord2
    r = norm(vec)
    return vec, r


@jit(nopython=True)
def get_angle(coords: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Compute the angle between two XYZ vectors.

    Keyword arguments
    -----------------
        coords : np.ndarray[float](3, 3)
            Contains XYZ for 3 atoms i, j, k with bonds i-j, k-j, which form the angle

    Returns
    -------
        The angle (float) in radians between the i-j and k-j vectors (bonds)
        A vector (np.ndarray[float](3,)) pointing from atom j to atom i
        A vector (np.ndarray[float](3,)) pointing from atom j to atom k
        The i-j bond length (float)
        The k-j bond length (float)
    """
    vec12, r12 = get_dist(coords[0], coords[1])
    vec32, r32 = get_dist(coords[2], coords[1])
    dot = np.dot(vec12/r12, vec32/r32)
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0
    return math.acos(dot), vec12, vec32, r12, r32


@jit(nopython=True)
def get_angle_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute the angle between two XYZ vectors.

    Keyword arguments
    -----------------
        vec1 : np.ndarray[float](3,)
        vec2 : np.ndarray[float](3,)

    Returns
    -------
        The angle (float) in radians between <vec1> and <vec2>
    """
    dot = np.dot(vec1/norm(vec1), vec2/norm(vec2))
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0
    return math.acos(dot)


@jit(nopython=True)
def get_dihed(coords: np.ndarray) -> tuple[float, np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray, np.ndarray]:
    """Get information about a dihedral given the coordinates of its four atoms i, j, k, and l.

    Keyword arguments
    -----------------
        coords : np.ndarray[float](4, 3)
            XYZ coordinates of the 4 atoms of the dihedral

    Returns
    -------
        the angle (float) in radians of the dihedral
        a vector pointing from j to i
        a vector pointing from j to k
        a vector pointing from l to k
        a vector which is perpendicular to the plane defined by bonds i-j and k-j
        a vector which is perpendicular to the plane defined by bonds k-j and k-l
    """
    vec12, r12 = get_dist(coords[0], coords[1])
    vec32, r32 = get_dist(coords[2], coords[1])
    vec34, r34 = get_dist(coords[2], coords[3])
    cross1 = cross_prod(vec12, vec32)
    cross2 = cross_prod(vec32, vec34)
    phi = get_angle_from_vectors(cross1, cross2)
    if dot_prod(vec12, cross2) < 0:
        phi = - phi
    return phi, vec12, vec32, vec34, cross1, cross2


@jit("f8[:](f8[:], f8[:])", nopython=True)
def cross_prod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the cross-product (perpendicular vector) of two XYZ vectors.

    Keyword arguments
    -----------------
        a : np.ndarray[float](3,)
        b : np.ndarray[float](3,)

    Returns
    -------
        A vector np.ndarray[double](3,) which is perpendicular to the plane containing <a> and <b>
    """
    c = np.empty(3, dtype=np.double)
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    return c


@jit("f8(f8[:])", nopython=True)
def norm(vec: np.ndarray) -> float:
    """Compute the L2-norm of an XYZ vector.

    Keyword arguments
    -----------------
        vec : np.ndarray[float](3,)

    Returns
    -------
        The L2-norm result (flaot)
    """
    # return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])


# @jit(nopython=True)
# def calc_quartic_angles(oords, atoms, theta0, fconst, force):
#    vec12, r12 = get_dist(coords[atoms[0]], coords[atoms[1]])
#    vec32, r32 = get_dist(coords[atoms[2]], coords[atoms[1]])
#    theta = get_angle(vec12, vec32)
#    cos_theta = np.cos(theta)
#    dtheta = theta - theta0
#
#    coefs = fconst * np.array([1, -0.014, 5.6e-5, -7e-7, 2.2e08])
#
#    dtp = dtheta
#    dvdt = 0
#    energy = coefs[0]
#    for i in range(1, 5):
#        dvdt += i*dtp*coefs[i]
#        dtp *= dtheta
#        energy += dtp * coefs[i]
#
#    st = - dvdt / np.sqrt(1. - cos_theta**2)
#    sth = st * cos_theta
#    c13 = st / r12 / r32
#    c11 = sth / r12 / r12
#    c33 = sth / r32 / r32
#
#    f1 = c11 * vec12 - c13 * vec32
#    f3 = c33 * vec32 - c13 * vec12
#    force[atoms[0]] += f1
#    force[atoms[2]] += f3
#    force[atoms[1]] -= f1 + f3
#    return energy
#
# def calc_g96angles(coords, angles, force):
#    for a, theta0, t in zip(angles.atoms, angles.minima, angles.term_ids):
#        r_ij, r12 = get_dist(coords[a[0]], coords[a[1]])
#        r_kj, r32 = get_dist(coords[a[2]], coords[a[1]])
#        theta = get_angle(r_ij, r_kj)
#        cos_theta = np.cos(theta)
#        dtheta = theta - theta0
#        energy[t] += 0.5 * dtheta**2
#
#        rij_1    = 1 / np.sqrt(np.inner(r_ij, r_ij))
#        rkj_1    = 1 / np.sqrt(np.inner(r_kj, r_kj))
#        rij_2    = rij_1*rij_1
#        rkj_2    = rkj_1*rkj_1
#        rijrkj_1 = rij_1*rkj_1
#
#        f1    = dtheta*(r_kj*rijrkj_1 - r_ij*rij_2*cos_theta);
#        f3    = dtheta*(r_ij*rijrkj_1 - r_kj*rkj_2*cos_theta);
#        f2    = - f1 - f3;
#
#        force[a[0], :, t] += f1;
#        force[a[1], :, t] += f2;
#        force[a[2], :, t] += f3;
#    return force
