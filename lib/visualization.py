import numpy as np
import matplotlib.pyplot as plt

class Visualization:
    def __init__(self, NUM_NODES, X_e_to_c, X_c_to_e):
        self.NUM_NODES = NUM_NODES
        self.X_e_to_c = X_e_to_c
        self.X_c_to_e = X_c_to_e
        self.colors = ["cornflowerblue", "goldenrod", "darkslateblue", "mediumseagreen", "firebrick", "orange", "aquamarine"]
    
    def get_embedding_name(self, id):
        return self.X_c_to_e[id]
    
    def print_group_examples(self, membership, num_examples=10):
        for n in range(self.NUM_NODES):
            print("Group %s: " % str(n))
            nodes = np.argwhere(np.array(membership) == n).ravel()
            for embedding in nodes[0:num_examples]:
                print(self.get_embedding_name(embedding), end=" ")
            print("\n")
    
    def visualize_communication(self, network_communication):     
        # visualize network communication statistics 
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes = [axes] if np.shape(axes) == () else axes

        x_pos = np.arange(self.NUM_NODES)
        width = 1/(self.NUM_NODES + 1)

        for n_target in range(self.NUM_NODES):
            y_data = []
            for n_start in range(self.NUM_NODES):
                y_data.append(network_communication[n_start][n_target][0])
            axes[0].bar(x_pos + width*n_target, y_data, width=width, color=self.colors[n_target%len(self.colors)], align="edge", label=("target node " + str(n_target)))
        
        axes[0].set_xticks(x_pos + (width*self.NUM_NODES)/2)
        axes[0].set_xticklabels(x_pos)
        axes[0].set_xlabel("Origin Node")
        axes[0].set_ylabel("Num Accesses")
        axes[0].set_title("Network Communication Distribution with Clustered Sharding")
        axes[0].legend()
        plt.show()

    def visualize_frequency_non_local_embeddings(self, freq_by_node, MAX_NODES=5):       
        MAX_NODES = MAX_NODES if self.NUM_NODES > MAX_NODES else self.NUM_NODES

        fig, axes = plt.subplots(nrows=MAX_NODES, ncols=1)
        axes = [axes] if np.shape(axes) == () else axes

        for n in range(MAX_NODES):
            x = np.arange(max(freq_by_node[n].keys())+1)
            y = []
            for x_val in x:
                if x_val in freq_by_node[n].keys():
                    y.append(freq_by_node[n][x_val])
                else:
                    y.append(0)
            axes[n].plot(x, y)
            axes[n].set_yscale("log")
            axes[n].set_ylabel("Node %s # Emb." % n)
            if n + 1 == MAX_NODES:
                axes[n].set_xlabel("Number of Repeated Non-Local Embeddings")
        plt.show()
                



