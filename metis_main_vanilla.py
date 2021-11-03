import copy
import numpy as np
import pymetis

from lib.parser import Parser
from lib.visualization import Visualization

# TODO: tell the parser what dataset we are using 
p = Parser()

NUM_NODES = 10
PERCENT_RESERVED_EMBEDDINGS = .1 # total = 109,397 embeddings

embedding_files = ['./data/MovieTweetings/latest/users.dat', './data/MovieTweetings/latest/movies.dat']
data_file = './data/MovieTweetings/latest/ratings.dat'

if __name__ == "__main__":
    # conversion betweeen embedding_id (str) and cluster_id (int)
    X_e_to_c, X_c_to_e = p.read_embeddings(embedding_files)
    RESERVED_EMBEDDINGS = int(PERCENT_RESERVED_EMBEDDINGS * len(X_c_to_e)/NUM_NODES) # calculate reserved embeddings from percent reserved

    # reformat data to only use cluster_id instead of embedding_id
    D = p.refactor_data(data_file, X_e_to_c)

    # create an adjaceny list from D
    adjacency_list = p.create_adjaceny_list(D, X_c_to_e)

    # separate the graph
    score, membership = pymetis.part_graph(NUM_NODES, adjacency=adjacency_list)

    # create node_placement graph
    node_placement = p.find_node_placement(NUM_NODES, membership)

    # create network_communication datastructure
    network_communication = p.derive_network_communication(NUM_NODES, D, node_placement)
    external_embedding_accesses_by_node, freq_by_node = p.derive_non_local_communication_frequency(NUM_NODES, network_communication)


    ################################################################################
    # visualize results 
    v = Visualization(NUM_NODES, X_e_to_c, X_c_to_e)

    # print groupings
    #v.print_group_examples(membership, num_examples=10)

    # bar chart showing interal vs external accesses
    v.visualize_communication(network_communication)
    v.visualize_frequency_non_local_embeddings(freq_by_node)
    ################################################################################

    # add back nodes using RESERVED_EMBEDDINGS
    node_placement_with_reserved = p.add_reserved_nodes(RESERVED_EMBEDDINGS, external_embedding_accesses_by_node, copy.deepcopy(node_placement))

    # create network_communication datastructure with reserved
    network_communication_with_reserved = p.derive_network_communication(NUM_NODES, D, node_placement_with_reserved)
    external_embedding_accesses_by_node_with_reserved, freq_by_node_with_reserved = p.derive_non_local_communication_frequency(NUM_NODES, network_communication_with_reserved)
    v.visualize_communication(network_communication_with_reserved)
    v.visualize_frequency_non_local_embeddings(freq_by_node_with_reserved)