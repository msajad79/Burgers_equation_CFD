import numpy as np
from tqdm import tqdm
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

from mesh import Node, Mesh2D
np.linalg.solve

class Solver():
    def __init__(self, mesh:Mesh2D) -> None:
        self.mesh:Mesh2D = mesh

        self.eq1_t1 = lambda j, n: [-1.0/self.mesh.delta_x[j-1], -1.0/2.0, 0.0]
        self.eq1_t2 = lambda j, n: [1.0/self.mesh.delta_x[j-1],  -1.0/2.0, 0.0]
        self.eq1_rhs = lambda j, n: self.R_s(j, n)

        self.eq2_t1 = lambda j, n: [0.0, -1.0/self.mesh.delta_x[j-1], -1.0/2.0]
        self.eq2_t2 = lambda j, n: [0.0, 1.0/self.mesh.delta_x[j-1], -1.0/2.0]
        self.eq2_rhs = lambda j, n: self.R_ss(j, n)

        self.eq3_t1 = lambda j, n: [0.0, 2.0/self.mesh.delta_t[n]+self.mesh.mesh[n+1][j-1].W, 2.0/self.mesh.delta_x[j-1]+self.mesh.mesh[n+1][j-1].U]
        self.eq3_t2 = lambda j, n: [0.0, 2.0/self.mesh.delta_t[n]+self.mesh.mesh[n+1][j].W  , -2.0/self.mesh.delta_x[j-1]+self.mesh.mesh[n+1][j].U]
        self.eq3_rhs = lambda j, n: 4.0*self.R(j, n) + self.R_sss(j, n)

        eq1 = [self.eq1_t1, self.eq1_t2, self.eq1_rhs]
        eq2 = [self.eq2_t1, self.eq2_t2, self.eq2_rhs]
        eq3 = [self.eq3_t1, self.eq3_t2, self.eq3_rhs]
        
        self.manager_eqs = {
            "borrowed" : eq2,
            "owner1"   : eq1,
            "owner2"   : eq3,
        }
    
    def R_s(self, j, n):
        return (
            -(self.mesh.mesh[n+1][j].V-self.mesh.mesh[n+1][j-1].V) / self.mesh.delta_x[j-1] +
            (self.mesh.mesh[n+1][j-1].U + self.mesh.mesh[n+1][j].U) / 2.0
        )

    def R_ss(self, j, n):
        return (
            (self.mesh.mesh[n+1][j-1].W + self.mesh.mesh[n+1][j].W) / 2.0 - 
            (self.mesh.mesh[n+1][j].U - self.mesh.mesh[n+1][j-1].U) / self.mesh.delta_x[j-1]
        )
    
    def R_sss(self, j, n):
        return (
            -4.0*(self.mesh.mesh[n+1][j].U + self.mesh.mesh[n+1][j-1].U) / (2.0*self.mesh.delta_t[n]) + 
            4.0*(self.mesh.mesh[n+1][j].W - self.mesh.mesh[n+1][j-1].W) / (2.0*self.mesh.delta_x[j-1]) -
            (self.mesh.mesh[n+1][j].U*self.mesh.mesh[n+1][j].W) - 
            (self.mesh.mesh[n+1][j-1].U*self.mesh.mesh[n+1][j-1].W)
        )

    def R(self, j, n):
        return (
            (self.mesh.mesh[n][j].U + self.mesh.mesh[n][j-1].U) / (2.0*self.mesh.delta_t[n]) +
            (self.mesh.mesh[n][j].W - self.mesh.mesh[n][j-1].W) / (2.0*self.mesh.delta_x[j-1]) -
            (self.mesh.mesh[n][j-1].U * self.mesh.mesh[n][j-1].W + self.mesh.mesh[n][j].U * self.mesh.mesh[n][j].W) / 4.0
        )
    
    def borrowed_eq_rhs(self, j, n):
        return self.manager_eqs["borrowed"][2](j, n)
    
    def owner1_eq_rhs(self, j, n):
        return self.manager_eqs["owner1"][2](j, n)
    
    def owner2_eq_rhs(self, j, n):
        return self.manager_eqs["owner2"][2](j, n)

    def RHS(self, n):
        rhs = np.array([0.0, 0.0, self.borrowed_eq_rhs(1 ,n)]).reshape(1,-1)
        for j in range(1,len(self.mesh.mesh[0])-1):
            rhs = np.append(rhs, [np.array([self.owner1_eq_rhs(j, n), self.owner2_eq_rhs(j, n),  self.borrowed_eq_rhs(j+1 ,n)])], axis=0)
        I = len(self.mesh.mesh[0]) - 1
        rhs = np.append(rhs, [np.array([self.owner1_eq_rhs(I, n), self.owner2_eq_rhs(I, n), (1.0/(2.0*(1+sum(self.mesh.delta_t[:n+1]))))-self.mesh.mesh[n+1][-1].V])], axis=0)
        return rhs
    
    def borrowed_eq_lhs_t1(self, j, n):
        return self.manager_eqs["borrowed"][0](j, n)
    def borrowed_eq_lhs_t2(self, j, n):
        return self.manager_eqs["borrowed"][1](j, n)
    
    def owner1_eq_lhs_t1(self, j, n):
        return self.manager_eqs["owner1"][0](j, n)
    def owner1_eq_lhs_t2(self, j, n):
        return self.manager_eqs["owner1"][1](j, n)
    
    def owner2_eq_lhs_t1(self, j, n):
        return self.manager_eqs["owner2"][0](j, n)
    def owner2_eq_lhs_t2(self, j, n):
        return self.manager_eqs["owner2"][1](j, n)
    
    def matrix_base(self, n):
        Bs = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            self.borrowed_eq_lhs_t1(1, n)
        ]).reshape((1,3,-1))
        Cs = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            self.borrowed_eq_lhs_t2(1, n)
        ]).reshape((1,3,-1))
        As = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]).reshape((1,3,-1))
        for j in range(1, len(self.mesh.mesh[0])-1):
            As = np.append(As, [np.array([
                self.owner1_eq_lhs_t1(j, n),
                self.owner2_eq_lhs_t1(j, n),
                [0.0, 0.0, 0.0]
            ])],axis=0)
            Bs = np.append(Bs, [np.array([
                self.owner1_eq_lhs_t2(j, n),
                self.owner2_eq_lhs_t2(j, n),
                self.borrowed_eq_lhs_t1(j+1, n)
            ])],axis=0)
            Cs = np.append(Cs, [np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                self.borrowed_eq_lhs_t2(j+1, n)
            ])],axis=0)

        I = len(self.mesh.mesh[0]) - 1
        As = np.append(As, [np.array([
            self.owner1_eq_lhs_t1(I, n),
            self.owner2_eq_lhs_t1(I, n),
            [0.0, 0.0, 0.0]
        ])],axis=0)
        Bs = np.append(Bs, [np.array([
            self.owner1_eq_lhs_t2(I, n),
            self.owner2_eq_lhs_t2(I, n),
            [1.0, 0.0, 0.0]
        ])],axis=0)
        Cs = np.append(Cs, [np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])],axis=0)

        return As, Bs , Cs
    
    def update_params(self, delta_values, n):
        for i,delta_value in enumerate(delta_values):
            self.mesh.mesh[n][i].update_params(delta_value) 
        

    def tridiag_block(self, As, Bs, Cs): 
        # c, u, d are center, upper and lower blocks, repeat N times
        cc = block_diag(*(Bs)) 
        shift = Bs[0].shape[1]
        uu = block_diag(*(Cs)) 
        uu = np.hstack((np.zeros((uu.shape[0], shift)), uu[:,:-shift]))
        dd = block_diag(*(As)) 
        dd = np.hstack((dd[:,shift:],np.zeros((uu.shape[0], shift))))
        return cc+uu+dd

    def KellerBox(self, As, Bs, Cs):
        return self.tridiag_block(As, Bs, Cs)

    def solve(self, solver:callable, MAX_CONV_COND=1.0e-2):
        for n in tqdm(range(len(self.mesh.mesh)-1)):
            self.deltas = []
            while True:
                rhs = self.RHS(n)
                #A = self.KellerBox(As, Bs, Cs)
                As, Bs, Cs = self.matrix_base(n)
                delta_params = solver(As, Bs, Cs, rhs)
                self.update_params(delta_params, n+1)
                if np.max(np.abs(delta_params)) < MAX_CONV_COND:
                    break
        
