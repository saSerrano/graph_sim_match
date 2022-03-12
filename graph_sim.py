'''
DESCRIPTION

This file implements the graph-similarity method from the article "Graph similarity scoring and matching", Zager & Verghese, 2008 (https://www.sciencedirect.com/science/article/pii/S0893965907001012).

Implementation author: Sergio A. Serrano
e-mail: sserrano@inaoep.mx
'''

import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def st_edge_mats(adj):
    '''
    DESCRIPTION

    This method builds the source-edge and terminus-edge matrices of an adjacency matrix, as introduced in section 2 in the original article.

    INPUT

    adj (numpy array): input adjacency matrix. All values must be 0 or 1.

    OUTPUT

    src_mat (numpy array): source-edge matrix of 'adj'.
    tgt_mat (numpy array): terminus-edge matrix of 'adj'.
    '''
    n_edges = np.sum(adj)
    n_nodes = adj.shape[0]

    # Edges are sorted from left->right and up->down
    edge_id = 0
    src_mat = np.zeros((n_nodes,n_edges), dtype=int)
    tgt_mat = np.zeros((n_nodes,n_edges), dtype=int)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i,j] > 0:
                src_mat[i,edge_id] = 1
                tgt_mat[j,edge_id] = 1
                edge_id = edge_id + 1

    return src_mat, tgt_mat

def graph_match(adj_a,nodes_a,adj_b,nodes_b,output_dir,max_ite=1000,epsilon=1.0):
    '''
    DESCRIPTION

    This method computes the node and edge similarity between two graphs based on the original article. The node and edge similarity tables are saved in two separate .csv files (nodes_sim.csv and edges_sim.csv), whereas the node and edge matchings are saved in a third file (matchings.txt). The iterative similarity method stops when it reaches 'max_ite' steps, or when the difference between two consecutive similarity matrices is less than 'epsilon', whatever happens first.

    INPUT

    adj_a (numpy array): adjacency matrix of graph A.
    adj_b (numpy array): adjacency matrix of graph B.
    nodes_a (list-string): list of node names in graph A.
    nodes_b (list-string): list of node names in graph B.
    output_dir (string): directory where similarity and matching files will be saved.
    max_ite (int): maximum number of iterations the similarity-method is allowed to perform.
    epsilon (float): tolerance between similarity matrices under which the iterative similarity method is considered to converge.
    '''
    # Verify for valid input
    assert len(adj_a.shape) == 2 and len(adj_b.shape) == 2
    assert adj_a.shape[0] == adj_a.shape[1] and adj_b.shape[0] == adj_b.shape[1]
    all_binary = True
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if adj[i,j] != 0 and adj[i,j] != 1:
                all_binary = False
                break
        if not all_binary:
            break
    assert all_binary
    assert len(nodes_a) == adj_a.shape[0] and len(nodes_b) == adj_b.shape[0]
    assert isinstance(max_ite,int) and max_ite > 0
    assert isinstance(epsilon,float) and epsilon >= 0.0

    # Build the source-edge and terminus-edge matrices
    s_mat_a, t_mat_a = st_edge_mats(adj_a)
    s_mat_b, t_mat_b = st_edge_mats(adj_b)

    # Create edge-names lists
    edges_a = []
    edges_b = []
    for i in range(s_mat_a.shape[1]):
        edges_a.append(nodes_a[s_mat_a[:,i].argmax()]+'->'+nodes_a[t_mat_a[:,i].argmax()])
    for i in range(s_mat_b.shape[1]):
        edges_b.append(nodes_b[s_mat_b[:,i].argmax()]+'->'+nodes_b[t_mat_b[:,i].argmax()])

    # Create fixed matrices
    G = np.kron(s_mat_b.transpose(),s_mat_a.transpose()) + np.kron(t_mat_b.transpose(),t_mat_a.transpose())
    G_t = np.kron(s_mat_b,s_mat_a) + np.kron(t_mat_b,t_mat_a)

    # Initialize similarity vectors
    x = np.ones((s_mat_a.shape[0]*s_mat_b.shape[0],1),dtype=float) # nodes similarity
    y = np.matmul(G,x) # edges similarity

    # Iterate to compute the similarity vectors
    stop_cause = ''
    ite = 0
    x_cp = np.copy(x)
    y_cp = np.copy(y)
    while True:
        # Update the similarity vectors and iteration counter
        x = np.matmul(G_t,y_cp)
        x = x / np.linalg.norm(x,'fro')
        y = np.matmul(G,x_cp)
        y = y / np.linalg.norm(y,'fro')
        ite = ite + 1

        # Evaluate stop condition
        if ite >= max_ite:
            stop_cause = 'Reached max-iterations\nite:'+str(ite)+'\nnodes-similarity-diff:'+str(x_diff)+'\nedges-similarity-diff:'+str(y_diff)
            break
        else:
            x_diff = np.absolute(x_cp - x).sum()
            y_diff = np.absolute(y_cp - y).sum()
            if x_diff <= epsilon and y_diff <= epsilon:
                stop_cause = 'Similarity values converged\nite:'+str(ite)+'\nnodes-similarity-diff:'+str(x_diff)+'\nedges-similarity-diff:'+str(y_diff)
                break

        # Save current similarity vecs for next step
        x_cp = np.copy(x)
        y_cp = np.copy(y)

    # Perform node & edge matching across graphs
    # Reshape similarity vectors to matrices
    x = x.reshape((adj_b.shape[0],adj_a.shape[0])).transpose()
    y = y.reshape((s_mat_b.shape[1],s_mat_a.shape[1])).transpose()
    # Perform nodes and edges matching
    row_x, col_x = linear_sum_assignment(x*(-1.0))
    row_y, col_y = linear_sum_assignment(y*(-1.0))

    # Save results
    # Similarity values
    df_nodes = pd.DataFrame(x,index=nodes_a,columns=nodes_b)
    df_edges = pd.DataFrame(y,index=edges_a,columns=edges_b)
    df_nodes.to_csv(os.path.join(output_dir,'nodes_sim.csv'))
    df_edges.to_csv(os.path.join(output_dir,'edges_sim.csv'))

    # Matching from graph-A to graph-B
    with open(os.path.join(output_dir,'matchings.txt'),'w') as outf:
        outf.write('Stop cause: '+stop_cause+'\n')
        outf.write('Nodes matching:\n')
        for i in range(row_x.shape[0]):
            outf.write(nodes_a[row_x[i]] + ' -> ' + nodes_b[col_x[i]] + '\n')
        outf.write('--------------------------------\n')
        outf.write('\nEdges matching:\n')
        for i in range(row_y.shape[0]):
            outf.write('(' + edges_a[row_y[i]] + ') -> (' + edges_b[col_y[i]] + ')\n')
