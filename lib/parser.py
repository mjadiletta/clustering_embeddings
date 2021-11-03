import os
import numpy as np
from random import sample

class Parser:
    def __init__(self):
        pass
    
    def read_table_MovieTweetings(self, the_dataset_file):
        embeddings = {}
        f = open(the_dataset_file, "rb")
        for line in f.readlines():
            line = str(line)[2:-3]
            data = line.split("::")

            embeddings[data[0]] = []
            for d in data[1:]:
                embeddings[data[0]].append(d)
        return embeddings

    def read_embeddings(self, files):
        X_e_to_c, X_c_to_e = [], {}
        for table_id, f in enumerate(files):
            embeddings = self.read_table_MovieTweetings(f)
            
            base_x = len(X_c_to_e.keys())
            X_e_to_c.append({})
            for i, embedding in enumerate(embeddings.keys()):
                X_e_to_c[table_id][embedding] = (i+base_x, embeddings[embedding])
                X_c_to_e[(i+base_x)] = embedding

        return X_e_to_c, X_c_to_e

    
    # Ratings: [user_id, movie_id, rating, timestamp]
    def read_data_MovieTweetings(self, the_dataset_file):
        data = []
        f = open(the_dataset_file, "rb")
        for line in f.readlines():
            line = str(line)[2:-3]
            d = line.split("::")
            data.append(d)
        return data

    def refactor_data_MovieTweetings(self, data, X_e_to_c):
        D = []
        for d in data:
            u_id = X_e_to_c[0][d[0]][0]
            m_id = X_e_to_c[1][d[1]][0]
            D.append((u_id, m_id))
        return D           


    def refactor_data(self, data_file, X_e_to_c):
        data = self.read_data_MovieTweetings(data_file)
        D = self.refactor_data_MovieTweetings(data, X_e_to_c)
        return D

    def create_adjaceny_list(self, D, X_c_to_e, name="./data/MovieTweetings.npy", remove_old=False):
        if not os.path.exists(name) or remove_old:
            adjacency_list = []
            num_entries = len(X_c_to_e.keys())
            for i in range(num_entries):
                adjacency_list.append(np.array(()))

            import time
            s = time.time()
            for d in D:
                for d_key in d:
                    for d_val in d:
                        if not (d_key == d_val):
                            adjacency_list[d_key] = np.append(adjacency_list[d_key], d_val)
            print("Time to create adjacency matrix: " + str(time.time() - s))
            np.save(name, np.asarray(adjacency_list, dtype="object"))

        adjacency_list = np.load(name, allow_pickle=True)
        return adjacency_list

    def find_node_placement(self, NUM_NODES, membership):
        node_placement = {}
        for n in range(NUM_NODES):
            nodes = np.argwhere(np.array(membership) == n).ravel()
            for embedding in nodes:
                node_placement[embedding] = [n]
        return node_placement

    def derive_network_communication(self, NUM_NODES, D, node_placement):
        # create node score datastructure 
        network_communication = {}
        for n_key in range(NUM_NODES):
            network_communication[n_key] = {}
            for n_val in range(NUM_NODES):
                network_communication[n_key][n_val] = [0, []]

        # takes two lists and find which value overlaps if one exists, or returns two random indices
        def find_intersection(n_keys, n_vals):
            for n_key in n_keys:
                for n_val in n_vals:
                    if n_key == n_val:
                        return n_key, n_val
            return sample(n_keys, 1)[0], sample(n_vals, 1)[0]

        for d in D:
            # assume first is always the key and others are values
            n_keys = node_placement[d[0]]
            for d_val in d[1:]:
                n_vals = node_placement[d_val]
                n_key, n_val = find_intersection(n_keys, n_vals)
                network_communication[n_key][n_val][0] += 1
                network_communication[n_key][n_val][1].append(d_val)
        return network_communication
    
    def derive_non_local_communication_frequency(self, NUM_NODES, network_communication):
        from collections import Counter

        external_embedding_accesses_by_node = {}
        for n_start in range(NUM_NODES):
            external_communication = []
            for n_target in range(NUM_NODES):
                if not (n_start == n_target):
                    external_communication.append(network_communication[n_start][n_target][1])
            external_embedding_accesses_by_node[n_start] = dict(Counter(i for sub in external_communication for i in sub))

        freq_by_node = {}
        for n in range(NUM_NODES):
            freq_by_node[n] = {}
            for key, val in external_embedding_accesses_by_node[n].items():
                if val not in freq_by_node[n].keys():
                    freq_by_node[n][val] = 0
                freq_by_node[n][val] += 1
        
        return external_embedding_accesses_by_node, freq_by_node
    
    def add_reserved_nodes(self, RESERVED_EMBEDDINGS, external_embedding_accesses_by_node, node_placement):
        for n in external_embedding_accesses_by_node.keys():
            embeddings = list(external_embedding_accesses_by_node[n].keys())
            frequencies = list(external_embedding_accesses_by_node[n].values())
            sorted_pairs = list(reversed(sorted(zip(frequencies, embeddings))))
            frequencies_sorted, embeddings_sorted = [ list(t) for t in list(zip(*sorted_pairs)) ]

            num_added = 0
            while num_added < RESERVED_EMBEDDINGS or (num_added + 1) == len(embeddings_sorted):
                # add nodes from embeddings_sorted to node_placement 
                added_embeding = embeddings_sorted[num_added]
                node_placement[added_embeding].append(n)
                num_added += 1
        return node_placement