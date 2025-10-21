"""
Backend for a 1D spring-element solver.
Unitless: use a consistent unit system for k, F, and u.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


class Node:
    """Single-DOF node for 1D spring problems."""
    def __init__(self, node_id: int):
        self.id = int(node_id)                         # 1-based id
        self.force = 0.0                               # external load (unitless)
        self.fixed = False                             # DOF constrained (includes prescribed-u)
        self.prescribed = False                        # True if a nonzero (or zero) displacement is prescribed
        self.u_prescribed = 0.0                        # prescribed displacement value
        self.u = 0.0                                   # solved displacement
        self.dof = self.id - 1                         # 0-based DOF index

class SpringElement:
    """Two-node axial spring element with stiffness k"""
    def __init__(self, ni: Node, nj: Node, k: float):
        if ni.id == nj.id:
            raise ValueError(f"Element with identical nodes i=j={ni.id} is not allowed.")
        if float(k) <= 0:
            raise ValueError("Stiffness k must be > 0.")
        self.i = ni
        self.j = nj
        self.k = float(k)
        self.axial = 0.0             # internal axial force (+tension)

    def ke(self) -> np.ndarray: # element stiffness matrix
        k = self.k
        return np.array([[k, -k], [-k, k]], dtype=float)

    def connectivity(self) -> Tuple[int, int]: # global DOF indices
        return (self.i.dof, self.j.dof)

    def add_to_global(self, K: np.ndarray) -> None: # assemble into global stiffness matrix
        ii, jj = self.connectivity()
        ke = self.ke()
        K[ii, ii] += ke[0, 0]
        K[ii, jj] += ke[0, 1]
        K[jj, ii] += ke[1, 0]
        K[jj, jj] += ke[1, 1]

    # Post-processing
    def elongation(self, u: np.ndarray): # u: global displacement vector
        ii, jj = self.connectivity()
        return float(u[jj] - u[ii])

    def axial_force(self, u: np.ndarray): # f: element axial force
        return self.k * self.elongation(u)

    def nodal_actions(self, u: np.ndarray): # forces in nodes i and j
        ii, jj = self.connectivity()
        ui, uj = float(u[ii]), float(u[jj])
        return (self.k * (ui - uj), self.k * (uj - ui))


class SpringFEASolver:
    """Assemble and solve a 1D spring system with mixed BCs (forces & prescribed displacements)."""
    def __init__(self, nodes: List[Node], elements: List[SpringElement]):
        self.nodes = nodes # list of Node
        self.elements = elements # list of SpringElement
        self.n = len(nodes) # total DOFs
        self.K_full = np.zeros((self.n, self.n), dtype=float) # global stiffness matrix
        self.F_full = np.zeros(self.n, dtype=float) # global force vector
        self.fixed = np.zeros(self.n, dtype=bool)   # mask of known displacements
        self.u_known = np.zeros(self.n, dtype=float) # known displacements

    def assemble(self) -> np.ndarray:
        n = self.n
        K = np.zeros((n, n), dtype=float)
        for e in self.elements:
            e.add_to_global(K)
        self.K_full = K
        self.F_full = np.array([nd.force for nd in self.nodes], dtype=float)
        self.fixed = np.array([nd.fixed for nd in self.nodes], dtype=bool)
        self.u_known = np.array([(nd.u_prescribed if nd.prescribed else 0.0) for nd in self.nodes], dtype=float)
        return K

    def solve(self):
        if self.K_full.sum() == 0.0: # if still a zero matrix, assemble
            self.assemble()
        free_idx = np.where(~self.fixed)[0] # indices of unknown DOFs
        fixed_idx = np.where(self.fixed)[0] # indices of known DOFs
        u = np.zeros(self.n)
        uc = self.u_known[fixed_idx] if fixed_idx.size else np.array([], dtype=float)

        if free_idx.size == 0:
            # all DOFs known
            u[fixed_idx] = uc
            R = self.K_full @ u - self.F_full
            for i, nd in enumerate(self.nodes):
                nd.u = float(u[i])
            for e in self.elements:
                e.axial = e.axial_force(u)
            return u, R, free_idx, fixed_idx

        Kff = self.K_full[np.ix_(free_idx, free_idx)] # reduce stiffness matrix
        Ff = self.F_full[free_idx]                     # reduce force vector
        rhs = Ff
        if fixed_idx.size > 0: # adjust rhs for known displacements
            Kfc = self.K_full[np.ix_(free_idx, fixed_idx)]
            rhs = Ff - Kfc @ uc
        try: # solve for unknown displacements
            uf = np.linalg.solve(Kff, rhs)
        except np.linalg.LinAlgError as e:
            raise ValueError("Stiffness matrix is singular aka A HOUSE ON WHEELS. Check connectivity and boundary conditions.") from e
        u[free_idx] = uf # assign unknown displacements
        if fixed_idx.size:
            u[fixed_idx] = uc # assign known displacements
        R = self.K_full @ u - self.F_full # reaction forces
        for i, nd in enumerate(self.nodes): 
            nd.u = float(u[i])
        for e in self.elements:
            e.axial = e.axial_force(u)
        return u, R, free_idx, fixed_idx

    def element_forces(self) -> List[float]:
        return [e.axial for e in self.elements]
