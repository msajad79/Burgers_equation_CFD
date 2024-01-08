from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

class Mesh2D():
    def __init__(self, l_x, l_t, f_delta_x:Callable, f_delta_t:Callable, NodeGen:"Node", MAX_NODE_EACH_DIM=1000, MIN_DELTA_T=1.0e-5, MIN_DELTA_X=1.0e-5, defult_val_U=0.0, defult_val_V=0.0, defult_val_W=0.0) -> None:
        self.f_delta_x, self.f_delta_t = f_delta_x, f_delta_t
        self.l_x, self.l_t = l_x, l_t
        self.MAX_NODE_EACH_DIM = MAX_NODE_EACH_DIM
        self.MIN_DELTA_T = MIN_DELTA_T
        self.MIN_DELTA_X = MIN_DELTA_X

        self.defult_val_U = defult_val_U
        self.defult_val_V = defult_val_V
        self.defult_val_W = defult_val_W

        #self.NodeGen:Node = NodeGen
        self.mesh:list[list[Node]] = self.create_mesh_rectangle()

    def generate_guide_node(self, length, f_delta, MIN_DELTA):
        guide_node = [0.0]
        delta_node = []
        c_node = 0
        while c_node < self.MAX_NODE_EACH_DIM:
            new_delta = f_delta(guide_node[-1])
            guide_node_new = guide_node[-1] + new_delta
            if guide_node_new > length:
                new_delta = length-guide_node[-1]
                guide_node_new = length
            if abs(guide_node_new - guide_node[-1]) < MIN_DELTA:
                new_delta += delta_node[-1]
                guide_node_new = length
                delta_node.pop()
                guide_node.pop()
            delta_node.append(new_delta)
            guide_node.append(guide_node_new)
            c_node += 1
            if guide_node[-1] == length:
                break
        else:
            raise Exception("The number of nodes in each dimension is greater than MAX_NODE_EACH_DIM . If you are sure about f_delta function, you can increase MAX_NODE_EACH_DIM.")
        return guide_node, delta_node


    def create_mesh_rectangle(self):
        x, self.delta_x = self.generate_guide_node(self.l_x, self.f_delta_x, self.MIN_DELTA_X)
        t, self.delta_t = self.generate_guide_node(self.l_t, self.f_delta_t, self.MIN_DELTA_T)
        mesh = []
        for n in range(len(t)):
            row = []
            for j in range(len(x)):
                row.append(
                    Node(
                        j,n, x[j], t[n], is_last_node=j+1 == len(x), 
                        defult_val_U=self.defult_val_U, 
                        defult_val_V=self.defult_val_V,
                        defult_val_W=self.defult_val_W
                    )
                )
            mesh.append(row)
        return mesh
    
    def get_U(self):
        return np.vectorize(lambda node: node.U)(self.mesh)

    def get_V(self):
        return np.vectorize(lambda node: node.V)(self.mesh)
    
    def get_W(self):
        return np.vectorize(lambda node: node.W)(self.mesh)
    
    def get_x(self):
        return np.vectorize(lambda node: node.x)(self.mesh)
    
    def get_t(self):
        return np.vectorize(lambda node: node.t)(self.mesh)

    def plot_mesh(self):
        x, t = zip(*[zip(*[(node.x,node.t) for node in nodes]) for nodes in self.mesh])
        x = [item for dt in x for item in dt]
        t = [item for dt in t for item in dt]

        # نمایش تری‌مش
        plt.scatter(x,t, s=2)
        plt.show()

    def plot_params(self, type_val="U", limit=-1):
        x = self.get_x()
        t = self.get_t()
        if type_val == "U":
            value = self.get_U()
        elif type_val == "V":
            value = self.get_V()
        elif type_val == "W":
            value = self.get_W()

        
        plt.pcolormesh(x[:limit], t[:limit], value[:limit])
        plt.ylabel(f'time')
        plt.xlabel('X')
        plt.title(f'{type_val} Values')

        plt.colorbar()
        plt.show() 


    def plot_n_timeframe(self, n, show_diver_U_False=False, show_diver_W_False=False, show_diver_V_False=False):
        plt.plot(self.get_x()[n], self.get_U()[n], label="U")
        plt.plot(self.get_x()[n], self.get_V()[n], label="V")
        plt.plot(self.get_x()[n], self.get_W()[n], label="W")
        if show_diver_U_False:
            plt.plot(self.get_x()[n][:-1], [(y2-y1)/x for y1,y2,x in zip(self.get_U()[n][0:-1],self.get_U()[n][1:], self.delta_x)], label="div U")
        if show_diver_W_False:
            plt.plot(self.get_x()[n][:-1], [(y2-y1)/x for y1,y2,x in zip(self.get_W()[n][0:-1],self.get_W()[n][1:], self.delta_x)], label="div W")
        if show_diver_V_False:
            plt.plot(self.get_x()[n][:-1], [(y2-y1)/x for y1,y2,x in zip(self.get_V()[n][0:-1],self.get_V()[n][1:], self.delta_x)], label="div V")
        plt.ylabel(f'V,U,W')
        plt.xlabel('X')
        plt.title(f'time = {round(sum(self.delta_t[:n]),3)}')
        plt.legend()
        plt.show()
        

class Node():
    def __init__(
            self, i, n, x, t, is_last_node=False, 
            defult_val_U=0.0,
            defult_val_V=0.0,
            defult_val_W=0.0,
        ) -> None:
        self.i = i
        self.n = n
        self.x = x
        self.t = t
        self.is_last_node = is_last_node
        
        # Defult value
        self.U = defult_val_U
        self.V = defult_val_V
        self.W = defult_val_W

        if n == 0:
            # I.C at n=0
            self.U = x
            self.V = x**2.0/2.0
            self.W = 1.0
        if i == 0:
            # B.C at i=0
            self.U = 0.0
            self.V = 0.0
        if self.is_last_node:
            #B.C at i=I
            #self.V = 1.0/(2.0*(1.0+t))-1.0/(2.0*(1.0+t))*.01
            1.0

    def update_params(self, delta_value):
        self.U += delta_value[1]
        self.V += delta_value[0]
        self.W += delta_value[2]
            

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(i={self.i}, n={self.n})"
