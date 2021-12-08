'''
DESCRIPTION

This files implements the graph-matching method from the article "A Linear Programming Approach for the Weighted Graph Matching Problem", Almohamad & Duffuaa, 1993 (https://ieeexplore.ieee.org/abstract/document/211474?casa_token=RWt4qoAVaLkAAAAA:ZSdqi3GAb3gBiXE6uwpmOTs1zA06cqREXEsnrSXkxRUeEWrZt9DXu33yTLN67-y3bD2zylJnFTKN3To).

Implementation author: Sergio A. Serrano
e-mail: sserrano@inaoep.mx
'''

from ortools.linear_solver import pywraplp
from ortools.init import pywrapinit
import numpy as np
import pandas as pd
import os

def paper_exp(random_seed=1,output_dir='.'):
    '''
    DESCRIPTION

    This function implements the evaluation of the algorithm proposed in the article
    "A Linear Programming Approach for the Weighted Graph Matching Problem" by H. A.
    Almohamad and S. 0. Duffuaa (1993).

    INPUT

    random_seed (int): seed used to initialize the pseudo-random generator for the
                shuffle order and selecting values for the noisy adjacency matrix.

    output_dir (string): directory where the results for directed and undirected
                graphs will be saved.
    '''
    # Experimental settings
    noise = [0.0,0.05,0.10,0.15,0.20]
    size = [5,6,7,8,9,10]
    n_sample = 50

    # Build result data-frames for directed and undirected graphs
    col_names = ['e (noise)']
    for s in size:
        col_names.append(str(s)+'-J-avg')
        col_names.append(str(s)+'-J-sd')
        col_names.append(str(s)+'-#-match')
        col_names.append(str(s)+'-E[J]')
    dir_df = pd.DataFrame(np.zeros((len(noise),1 + 4*len(size)),dtype=float),columns=col_names)
    und_df = pd.DataFrame(np.zeros((len(noise),1 + 4*len(size)),dtype=float),columns=col_names)
    for i in range(len(noise)):
        dir_df.loc[i].at['e (noise)'] = noise[i]
        und_df.loc[i].at['e (noise)'] = noise[i]
    for s in size:
        for i in range(len(noise)):
            dir_df.loc[i].at[str(s)+'-E[J]'] = noise[i]*s*(s-1)/2
            und_df.loc[i].at[str(s)+'-E[J]'] = noise[i]*s*(s-1)/2

    # Initialize OR-Tools-related variables
    pywrapinit.CppBridge.InitLogging('wgmp_lp.py')
    cpp_flags = pywrapinit.CppFlags()
    cpp_flags.logtostderr = True
    cpp_flags.log_prefix = False
    pywrapinit.CppBridge.SetFlags(cpp_flags)

    # Initialize the random-genetaion seed
    np.random.seed(random_seed)

    # Evaluate on directed graphs
    for n in size:
        print('Matching directed graphs of size '+str(n)+': ',end='')
        for e in range(len(noise)):
            print(noise[e],end='')
            if e < (len(noise) - 1):
                print(',',end='')
            j_vec = np.zeros((n_sample,),dtype=float)
            n_match = 0
            for s in range(n_sample):
                ad_a = np.zeros((n,n),dtype=float)
                ad_b = np.zeros((n,n),dtype=float)
                ad_b_shuf = np.zeros((n,n),dtype=float)
                so = np.arange(n)
                np.random.shuffle(so)
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            ad_a[i,j] = np.random.uniform(0.0,1.0)
                            ad_b[i,j] = ad_a[i,j] + np.random.uniform(-noise[e],noise[e])
                            # Shuffle graph B
                            ad_b_shuf[so[i],so[j]] = ad_b[i,j]

                # Perform graph matching
                j,pm,p = wgmp_lp(ad_a,ad_b_shuf,so)
                j_vec[s] = j
                if pm:
                    n_match = n_match + 1

            # Save
            dir_df.loc[e].at[str(n)+'-J-avg'] = np.mean(j_vec)
            dir_df.loc[e].at[str(n)+'-J-sd'] = np.std(j_vec)
            dir_df.loc[e].at[str(n)+'-#-match'] = n_match
        print('...done.')
    print('------------------------------')

    # Evaluate on undirected graphs
    for n in size:
        print('Matching undirected graphs of size '+str(n)+': ',end='')
        for e in range(len(noise)):
            print(noise[e],end='')
            if e < (len(noise) - 1):
                print(',',end='')
            j_vec = np.zeros((n_sample,),dtype=float)
            n_match = 0
            for s in range(n_sample):
                ad_a = np.zeros((n,n),dtype=float)
                ad_b = np.zeros((n,n),dtype=float)
                ad_b_shuf = np.zeros((n,n),dtype=float)
                so = np.arange(n)
                np.random.shuffle(so)
                # Create adjacency matrices A and B
                for i in range(n):
                    for j in range(n):
                        if j < i:
                            ad_a[i,j] = ad_a[j,i]
                            ad_b[i,j] = ad_b[j,i]
                        elif i != j:
                            ad_a[i,j] = np.random.uniform(0.0,1.0)
                            ad_b[i,j] = ad_a[i,j] + np.random.uniform(-noise[e],noise[e])
                # Shuffle graph B
                for i in range(n):
                    for j in range(n):
                        ad_b_shuf[so[i],so[j]] = ad_b[i,j]

                # Perform graph matching
                j,pm,p = wgmp_lp(ad_a,ad_b_shuf,so)
                j_vec[s] = j
                if pm:
                    n_match = n_match + 1

            # Save
            und_df.loc[e].at[str(n)+'-J-avg'] = np.mean(j_vec)
            und_df.loc[e].at[str(n)+'-J-sd'] = np.std(j_vec)
            und_df.loc[e].at[str(n)+'-#-match'] = n_match
        print('...done.')

    # Save results
    dir_df.to_csv(os.path.join(output_dir,'directed_seed_'+str(random_seed)+'.csv'))
    und_df.to_csv(os.path.join(output_dir,'undirected_seed_'+str(random_seed)+'.csv'))

def wgmp_lp(ad_a,ad_b,shuffle_order=np.zeros((1,)),debug=False):
    '''
    DESCRIPTION

    This function implements the algorithm proposed in the article "A Linear Pro-
    gramming Approach for the Weighted Graph Matching Problem" by H. A. Almohamad
    and S. 0. Duffuaa (1993).

    INPUT

    ad_a (numpy array): adjacency matrix of graph A.
    ad_b (numpy array): adjacency matrix of graph B.
    shuffle_order (numpy array): vector containing the shuffle order.
    debug (bool): flag that indicates whether info should be displayed in console.

    OUTPUT:

    J (float): value of the criterion that should be minimized, according to the paper.
    perfect_match (bool): flag that indicates whether every node was correctly matched.
    p_mat (numpy array): real-valued permutation matrix P.
    '''
    # number of nodes
    n = ad_a.shape[0]

    # Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create variables: P, S and T
    P = []
    S = []
    T = []
    infinity = solver.infinity()
    for i in range(n):
        p = []
        s = []
        t = []
        for j in range(n):
            p.append(solver.NumVar(0, infinity, 'p_'+str(i)+'_'+str(j)))
            s.append(solver.NumVar(0, infinity, 's_'+str(i)+'_'+str(j)))
            t.append(solver.NumVar(0, infinity, 't_'+str(i)+'_'+str(j)))
        P.append(p)
        S.append(s)
        T.append(t)

    if debug:
        print('Number of variables =', solver.NumVariables())

    # Create restrictions for the real-valued permutation matrix P
    # Row-sum restrictions
    for i in range(n):
        ct = solver.Constraint(1,1, 'ct_p_row_'+str(i)) # elems in row sum 1.0
        for j in range(n):
            ct.SetCoefficient(P[i][j], 1)
    # Col-sum restrictions
    for j in range(n):
        ct = solver.Constraint(1,1, 'ct_p_col_'+str(i)) # elems in row sum 1.0
        for i in range(n):
            ct.SetCoefficient(P[i][j], 1)

    # Restriction: A_AB * p + S - T = 0
    # Build matrix of both graphs adjacencies
    ad_ab = np.zeros((n*n,n*n),dtype=float)
    for i in range(n):
        for j in range(n):
            ad_ab[i*n+j,i*n:i*n+n] = ad_ab[i*n+j,i*n:i*n+n] + ad_a[j,:]
    for i in range(n):
        for j in range(n):
            tmp = np.zeros((n*n,),dtype=float)
            for k in range(n):
                tmp[k*n+j] = ad_b[k,i]
            ad_ab[i*n+j,:] = ad_ab[i*n+j,:] - tmp
    # Define the nxn constraints
    for k in range(n*n):
        ct = solver.Constraint(0,0, 'ct_apst_'+str(k))
        row = k % n
        col = int(k / n)
        ct.SetCoefficient(S[row][col], 1)
        ct.SetCoefficient(T[row][col], -1)
        for j in range(n):
            for i in range(n):
                ct.SetCoefficient(P[i][j], ad_ab[k,j*n+i]) # A_AB

    if debug:
        print('Number of constraints =', solver.NumConstraints())

    # Create objective function
    objective = solver.Objective()
    for i in range(n):
        for j in range(n):
            objective.SetCoefficient(S[i][j], 1)
            objective.SetCoefficient(T[i][j], 1)
    objective.SetMinimization()

    # Solve the opt. problem and show results
    solver.Solve()

    # Compute results
    # Put P-matrix in a numpy array
    p_mat = np.zeros((n,n),dtype=float)
    for i in range(n):
        for j in range(n):
            p_mat[i,j] = P[i][j].solution_value()
    if debug:
        print('Objective value: ', objective.Value())
        print('P matrix: ')
        print(p_mat)

    # Compute the criterion value J
    J = np.abs(ad_a - np.matmul(np.matmul(p_mat,ad_b),p_mat.transpose())).sum()

    # Build the optimal matching vector A -> B
    n_correct_match = 0
    for i in range(n):
        max_col = np.argmax(p_mat[i,:])
        if shuffle_order.shape[0] > 1:
            if max_col == shuffle_order[i]:
                n_correct_match = n_correct_match + 1
        else:
            if max_col == i:
                n_correct_match = n_correct_match + 1
    perfect_match = (n_correct_match == n)

    return J, perfect_match, p_mat
