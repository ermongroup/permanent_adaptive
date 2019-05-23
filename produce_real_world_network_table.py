import numpy as np
import networkx as nx
import scipy.io

from constant_num_targets_sample_permenant import conjectured_optimal_bound

def read_network_file(filename):
    '''
    Inputs:
    - filename: (string) filename for a .mtx or .edges matrix

    Outputs:
    - edge_matrix: (np.array i think) array representation for permanent input (double check...)
    '''
    f = open(filename, 'rb')
    if filename.split('.')[-1] == 'mtx': # for .mtx
        edge_matrix = scipy.io.mmread(f).toarray()#[0:2, 0:5]
    else:#for .edges
        assert(filename.split('.')[-1] == 'edges'), (filename.split('.')[-1] )
        graph = nx.read_edgelist(f)
        sparse_matrix = nx.adjacency_matrix(graph)
        edge_matrix = np.asarray(sparse_matrix.todense())
    f.close()
    return edge_matrix

def create_sinkhorn_values_in_table(matrix_filenames):

    for cur_matrix_filename in matrix_filenames:
        edge_matrix = read_network_file(cur_matrix_filename)
        permanent_LB, conjectured_permanent_UB, valid_sinkhorn_UB, runtime = conjectured_optimal_bound(edge_matrix, return_lower_bound=True)
        print('-'*80)
        print(cur_matrix_filename)
        # print(edge_matrix)
        print("sinkhorn approximation from gurvits:")
        print("ln(permanent_LB)", np.log(permanent_LB))
        print("ln(valid_sinkhorn_UB)", np.log(valid_sinkhorn_UB))
        print("ln(conjectured_permanent_UB)", np.log(conjectured_permanent_UB))
        print("runtime", runtime)

if __name__ == "__main__":



#WORKING EXAMPLES
    # matrix_filename = "./networkrepository_data/cage5.mtx"
    # matrix_filename = "./networkrepository_data/bcspwr01.mtx"

    #matrix_filename = "./networkrepository_data/edge_defined/ENZYMES_g192.edges" #change file eading for .edges!!
    # matrix_filename = "./networkrepository_data/edge_defined/ENZYMES_g230.edges" #change file eading for .edges!!
    # matrix_filename = "./networkrepository_data/edge_defined/ENZYMES_g479.edges" #change file eading for .edges!!
    # matrix_filename = "./networkrepository_data/edge_defined/ENZYMES_g490.edges" #change file eading for .edges!!


# WAI faster:
    # matrix_filename = "./networkrepository_data/smaller_networks/can_24.mtx"


    matrix_filenames = ["./networkrepository_data/edge_defined/ENZYMES_g192.edges",\
                    "./networkrepository_data/edge_defined/ENZYMES_g230.edges",\
                    "./networkrepository_data/edge_defined/ENZYMES_g479.edges",\
                    "./networkrepository_data/cage5.mtx",\
                    "./networkrepository_data/bcspwr01.mtx"]

    create_sinkhorn_values_in_table(matrix_filenames)
