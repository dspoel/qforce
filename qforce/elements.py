ATOM_SYM = ("*", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
            "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In")
ELE_COV = (0.00, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57,
           0.58, 1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03,
           1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32,
           1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90,
           1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42,
           1.39, 1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03,
           2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90,
           1.87, 1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36,
           1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15,
           2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69, 1.60, 1.60, 1.60,
           1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60,
           1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60)
ELE_MAXB = (0, 1, 0, 1, 2, 4, 4, 4, 2, 1, 0, 1, 2, 6, 6, 6, 6, 1, 0,
            1, 2, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 3, 4, 3, 2, 1, 0, 1,
            2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 4, 3, 2, 1, 0, 1, 2,
            2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 3, 4, 3, 2, 1, 0, 1, 2, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6)
#
ATOMMASS = (0, 1.00794, 4.002602, 6.941, 9.012182, 10.811, 12.0107,
            14.0067, 15.9994, 18.9984032, 20.1797, 22.98977, 24.3050,
            26.981538, 28.0855, 30.973761, 32.065, 35.453, 39.948,
            39.0983, 40.078, 44.95591, 47.867, 50.9415, 51.9961,
            54.938049, 55.845, 58.9332, 58.6934, 63.546, 65.38,
            69.723, 72.64, 74.92160, 78.96, 79.904, 83.798, 85.4678,
            87.62, 88.90585, 91.224, 92.90638, 95.96, 98, 101.07,
            102.90550, 106.42, 107.8682, 112.411, 114.818, 118.701,
            121.760, 127.60, 126.90447, 131.293, 132.90545, 137.327,
            138.9055, 140.116, 140.90765, 144.24, 145, 150.36,
            151.964, 157.25, 158.92534, 162.500, 164.93032, 167.259,
            168.93421, 173.054, 174.9668, 178.49, 180.9479, 183.84,
            186.207, 190.23, 192.217, 195.078, 196.96655, 200.59,
            204.3833, 207.2, 208.98040, 209, 210, 222, 223, 226, 227,
            232.0381, 231.03588, 238.02891, 237.05, 244.06, 243.06,
            247.07, 247.07, 251.08, 252.08, 257.10, 258.10, 259.10,
            262.11, 265.12, 268.13, 271.13, 270, 277.15, 276.15,
            281.16, 280.16, 285.17, 284.18, 289.19, 288.19, 293, 294,
            294)
# electronegativity
ELE_ENEG = (0.00, 2.20, 0.00, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44,
            3.98, 0.00, 0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16,
            0.00, 0.82, 1.00, 1.36, 1.54, 1.63, 1.66, 1.55, 1.83,
            1.88, 1.91, 1.90, 1.65, 1.81, 2.01, 2.18, 2.55, 2.96,
            3.00, 0.82, 0.95, 1.22, 1.33, 1.60, 2.16, 1.90, 2.20,
            2.28, 2.20, 1.93, 1.69, 1.78)
