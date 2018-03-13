from collections import *

def compute_largest_cc_size(g):
    """
    Returns size of largest connected component from input graph
    """
    max_size = 0
    Q = []
    nodes = g.keys()
    node_dict = dict()

    for node in nodes:
        node_dict[node] = False

    #Iterates through nodes and runs BFS for any that haven't been visited 
    for node in node_dict.keys():
        if not node_dict[node]:
            node_dict[node] = True
            Q.append(node)
            size = 1

            #BFS algorithm 
            while Q:
                k = Q.pop(0)
                for neighbor in g[k]:
                    if not node_dict[neighbor]:
                        node_dict[neighbor] = True
                        Q.append(neighbor)
                        size += 1
            if size > max_size:
                max_size = size

    return max_size

### test cases
# g1 = dict()
# print compute_largest_cc_size(g1)
# expect 0

# g2 = {1: {2, 3}, 2: {3, 1}, 3: {1, 2}}
# print compute_largest_cc_size(g2)
# expect 3

#g3 = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
#print compute_largest_cc_size(g3)
# expect 1

#g4 = {1: {2}, 2: {1, 3}, 3: {2, 4}, 4: {3, 5}, 5: {4}}
#print compute_largest_cc_size(g4)
# expect 5