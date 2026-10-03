import probability as prob
import networkx as nx
import numpy as np
import networks
import sys
import time
import ctypes
'''
This script calculates the percentage of nodes activated with a given alpha value 
'''

def estimate(A_ptr, n, p, seeds, ic_trials, threads=0):
    '''
        This function is modified from probability.py so that we can call the cpp function thousands of times (with a new random seed
        each time) while only converting the graph to an adjacency matrix once.
    '''
    # prepare cpp arguments

    prob_est_cpp = ctypes.CDLL('./cpp/prob_est')

    result = np.zeros(n, dtype=np.float32) # for storing the results of the cpp program

    # prepare array pointers
    ss_ptr = np.array(seeds, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call the C++ function with the 2D array
    prob_est_cpp.estimate(ctypes.c_int(threads), ctypes.c_float(p), ctypes.c_int(n), ctypes.c_int(ic_trials), ctypes.c_int(len(seeds)), A_ptr, ss_ptr, result_ptr)

    return result

def main(network, p, IC_trials):
    if network == "google":
        G = networks.get_google()
    elif network == "email":
        G = networks.get_emailnetwork()
    else:
        raise Exception("Command not recognized")
    #choose a random seed for each trial
    seeds = np.random.choice(G.nodes, size=IC_trials, replace=True)
    num_activated = []
    #prepare the graph for probest
    A = nx.to_numpy_array(G, dtype=np.int32).flatten()
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    n = G.number_of_nodes()
    iters = 0
    #Run probest IC_trials times
    for s in seeds:
        iters = iters + 1
        if (iters % 500 == 0):
            print(num_activated[-1], flush=True)
        test = estimate(A_ptr, n, p, [s], 1)
        num_activated.append(np.sum(test))
    fraction = float(np.mean(num_activated)) / G.number_of_nodes()
    print(f"Percentage of nodes activated for {network} with alpha = {p} is: {fraction*100}", flush=True)

if __name__ == "__main__":
    network = sys.argv[1]
    p = float(sys.argv[2])
    IC_trials = int(sys.argv[3])
    main(network, p, IC_trials)
    print(f"network: {network} alpha: {p} IC_trials: {IC_trials}", flush=True)
