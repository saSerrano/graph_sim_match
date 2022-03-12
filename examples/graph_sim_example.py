'''
DESCRIPTION

This code is an example of how to compute node and edge similarity across two
different graphs. The similary is computed based on the which nodes are connec-
ted, thus, function 'graph_match' expects as input a pair of adjacency matrices
of 1's and 0's. Results are saved in directory 'output_dir'.

Author: Sergio A. Serrano
e-mail: sserrano@inaoep.mx
'''
import os
import numpy as np
import pandas as pd
from graph_sim import graph_match

def main():
    # Number of nodes
    n_nodes = 4

    # Generate two random adjacency matrices
    graph_a = np.random.random_integers(low=0,high=1,size=(n_nodes,n_nodes))
    graph_b = np.random.random_integers(low=0,high=1,size=(n_nodes,n_nodes))

    # Lists of node names
    names_a = []
    names_b = []
    for i in range(graph_a.shape[0]):
        names_a.append('a'+str(i))
        names_b.append('b'+str(i))

    # Maximum number of iterations
    max_ite = 1000

    # Convergence error
    epsilon = 0.001

    # Compute similarity
    output_dir = './graph_sim_results'
    os.system('mkdir '+output_dir)
    graph_match(graph_a,names_a,graph_b,names_a,output_dir,max_ite,epsilon)

    # Save adjacency matrices
    df_a = pd.DataFrame(graph_a,index=names_a,columns=names_a)
    df_b = pd.DataFrame(graph_b,index=names_b,columns=names_b)
    df_a.to_csv(os.path.join(output_dir,'graph_a.csv'))
    df_b.to_csv(os.path.join(output_dir,'graph_b.csv'))

if __name__ == '__main__':
    main()
