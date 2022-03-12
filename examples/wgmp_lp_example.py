'''
DESCRIPTION

This code is an example of how to compute node matching between two different
graphs that have the same number of nodes. The graph matching is computed as a
linear programming optimization problem. Additionally, this script shows how to
reproduce the experiemnts of the original paper.

Author: Sergio A. Serrano
e-mail: sserrano@inaoep.mx
'''
import os
import sys
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from ortools.init import pywrapinit
from wgmp_lp import paper_exp, wgmp_lp

def main(run_case):
    # Output directory
    output_dir = './wgmp_lp_results'
    os.system('mkdir '+output_dir)

    # Reproduce paper experiment
    if run_case == 'rep-exp':
        paper_exp(paper_exp(random_seed=100,output_dir=output_dir))
    # Run a simple example of how to use the matching procedure
    else:
        # Number of nodes
        n_nodes = 4

        # Generate two random graphs
        graph_a = np.multiply(np.random.random_integers(low=0,high=1,size=(n_nodes,n_nodes)),np.random.rand(n_nodes,n_nodes))
        graph_b = np.multiply(np.random.random_integers(low=0,high=1,size=(n_nodes,n_nodes)),np.random.rand(n_nodes,n_nodes))

        # Lists of node names
        names_a = []
        names_b = []
        for i in range(graph_a.shape[0]):
            names_a.append('a'+str(i))
            names_b.append('b'+str(i))

        # Save adjacency matrices
        df_a = pd.DataFrame(graph_a,index=names_a,columns=names_a)
        df_b = pd.DataFrame(graph_b,index=names_b,columns=names_b)
        df_a.to_csv(os.path.join(output_dir,'graph_a.csv'))
        df_b.to_csv(os.path.join(output_dir,'graph_b.csv'))

        '''
        NOTE: Always initialize the OR-Tools-related variables exactly once in a
        program before using the wgmp_lp function.
        '''
        # Initialize OR-Tools-related variables
        pywrapinit.CppBridge.InitLogging('wgmp_lp_example.py')
        cpp_flags = pywrapinit.CppFlags()
        cpp_flags.logtostderr = True
        cpp_flags.log_prefix = False
        pywrapinit.CppBridge.SetFlags(cpp_flags)

        # Compute the matching between graphs A and B
        J, perfect_match, p_mat = wgmp_lp(graph_a,graph_b)

        '''
        NOTE: perfect_match represents the success of correctly matchingthe nodes
        between graph A and B, which only makes sense if graphs A and B are the
        same graph. If graph A and B are two different graphs, then ignore the
        valuel of perfect_match.
        '''

        # Save results
        with open(os.path.join(output_dir,'matching_info.txt'),'w') as f:
            f.write('J-score (the closer to 0 -> more similar): '+str(J)+'\n')
            f.write('Perfect match: '+str(perfect_match))
            f.close()
        df_match = pd.DataFrame(p_mat,index=names_a,columns=names_b)
        df_match.to_csv(os.path.join(output_dir,'matching.csv'))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] != 'rep-exp' and sys.argv[1] != 'example':
            print('Usage: python wgmp_lp_example.py <rep-exp | example>')
        else:
            main(sys.argv[1])
    else:
        print('Usage: python wgmp_lp_example.py <rep-exp | example>')
