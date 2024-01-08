
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from mesh import Mesh2D, Node
from solver import Solver


if __name__ == "__main__":
    f_delta_x = lambda x: .03#(0.0001-0.1)*x + .1
    f_delta_t = lambda x: .03#(0.0001-0.1)*x + .1
    l_x, l_t = 1.0,1.0
    mesh_2D = Mesh2D(
        l_x, l_t, f_delta_x, f_delta_t, NodeGen=Node, MAX_NODE_EACH_DIM=10000000, 
        defult_val_U = .0,
        defult_val_V = .0,
        defult_val_W = .0,
    )
    #mesh_2D.plot_mesh()
    solver = Solver(mesh_2D)
    s= solver.solve()
    [i for i in s]



    print(mesh_2D)