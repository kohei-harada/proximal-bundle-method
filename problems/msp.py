#
# This program is derived from http://peekaboo-vision.blogspot.com/2012/02/simplistic-minimum-spanning-tree-in.html
# written by Andreas Mueller
#

import numpy as np
from scipy.spatial.distance import pdist, squareform
import time
import scipy.io as io
 
def minimum_spanning_tree(X, copy_X=True, dtype="float"):

    if copy_X:
        X = X.copy()
        X = X.astype(float)

    inf = 214748364
    if dtype == "float":
        inf = np.inf
        
    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []
     
    # initialize with node 0:                                                                                         
    visited_vertices = [0]                                                                                            
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = inf
     
    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)
        # 2d encoding of new_edge from flat, get correct indices                                                      
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]                                                       
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = inf
        X[new_edge[1], visited_vertices] = inf
        num_visited += 1
    return np.vstack(spanning_edges)
 
if __name__ == "__main__":

    X = io.loadmat("TSP_hk48")["M"]
    Y = X.astype(np.int32)
    startime = time.time()
    edge_list = minimum_spanning_tree(Y)
    print time.time() - startime
    print edge_list
    startime = time.time()
    edge_list = minimum_spanning_tree(X, "float")
    print time.time() - startime
    print edge_list
    
