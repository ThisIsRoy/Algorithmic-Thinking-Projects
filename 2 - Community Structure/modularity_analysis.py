import random
from collections import defaultdict
import collections
import copy
import pylab
import types
import math

##################
# Helper Functions
##################


class LinAl(object):
    """
    Contains code for various linear algebra data structures and operations.
    """

    @staticmethod
    def zeroes(m, n):
        """
        Returns a matrix of zeroes with dimension m x n.
        ex: la.zeroes(3,2) -> [[0,0],[0,0],[0,0]]
        """

        return [[0] * n for i in range(m)]

    @staticmethod
    def trace(matrix):
        """
        Returns the trace of a square matrix. Assumes valid input matrix.
        ex: la.trace([[1,2],[-1,0]]) -> 1.0
        """

        if len(matrix[0]) == 0:
            return 0.0

        return float(sum(matrix[i][i] for i in range(len(matrix))))

    @staticmethod
    def transpose(matrix):
        """
        Returns the transpose of a matrix. Assumes valid input matrix.
        ex: la.transpose([[1,2,3],[4,5,6]]) -> [[1,4],[2,5],[3,6]]
        """

        res = [[0] * len(matrix) for i in range(len(matrix[0]))]

        for i in range(len(matrix[0])):
            for j in range(len(matrix)):
                res[i][j] = matrix[j][i]

        return res

    @staticmethod
    def dot(a, b):
        """
        Returns the dot product of two n x 1 vectors. Assumes valid input vectors.
        ex: la.dot([1,2,3], [3,-1,4]) -> 13.0
        """

        if len(a) != len(b):
            raise Exception("Input vectors must be of same length, not %d and %d" % (len(a), len(b)))

        return float(sum([a[i] * b[i] for i in range(len(a))]))

    @staticmethod
    def multiply(A, B):
        """
        Returns the matrix product of A and B. Assumes valid input matrices.
        ex: la.multiply([[1,2],[3,4]], [[-3,4],[2,-1]]) -> [[1.0,2.0],[-1.0,8.0]]
        """

        if len(A[0]) != len(B):
            raise Exception("Matrix dimensions do not match for matrix multiplication: %d x %d and %d x %d" % (
            len(A), len(A[0]), len(B), len(B[0])))

        result = [[0] * len(B[0]) for i in range(len(A))]

        for i in range(len(A)):
            for j in range(len(B[0])):
                result[i][j] = LinAl.dot(A[i], LinAl.transpose(B)[j])

        return result

    @staticmethod
    def sum(matrix):
        """
        Returns the sum of all the elements in matrix. Assumes valid input matrix.
        ex: la.sum([[1,2],[3,4]]) -> 10.0
        """

        return float(sum([sum(row) for row in matrix]))

    @staticmethod
    def multiply_by_val(matrix, val):
        """
        Returns the result of multiply matrix by a real number val. Assumes valid
        imput matrix and that val is a real number.
        """

        new_mat = LinAl.zeroes(len(matrix), len(matrix[0]))
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                new_mat[i][j] = val * matrix[i][j]
        return new_mat


def remove_edges(g, edgelist):
    """
    Remove the edges in edgelist from the graph g.

    Arguments:
    g -- undirected graph
    edgelist - list of edges in g to remove

    Returns:
    None
    """
    for edge in edgelist:
        (u, v) = tuple(edge)
        g[u].remove(v)
        g[v].remove(u)


def bfs(g, startnode):
    """
    Perform a breadth-first search on g starting at node startnode.

    Arguments:
    g -- undirected graph
    startnode - node in g to start the search from

    Returns:
    The distances from startnode to each node.
    """
    dist = {}

    # Initialize distances and predecessors
    for node in g:
        dist[node] = float('inf')
    dist[startnode] = 0

    # Initialize the search queue
    queue = collections.deque([startnode])

    # Loop until all    connected nodes have been explored
    while queue:
        j = queue.popleft()
        for h in g[j]:
            if dist[h] == float('inf'):
                dist[h] = dist[j] + 1
                queue.append(h)
    return dist


def bfs_mod(g, startnode):
    """
    :param g: graph
    :param start node: starting node in graph to perform bfs
    :return: a tuple of node distances and # of shortest paths
    """
    # initialize bfs algorithm searching for d and n
    nodes = g.keys()
    queue = []
    dist_dict = dict()
    n_dict = dict()
    for node in nodes:
        dist_dict[node] = float('inf')
    dist_dict[startnode] = 0
    n_dict[startnode] = 1

    # modified BFS
    queue.append(startnode)
    while queue:
        j_node = queue.pop(0)

        for neighbor in g[j_node]:
            if dist_dict[neighbor] == float('inf'):
                dist_dict[neighbor] = dist_dict[j_node] + 1
                n_dict[neighbor] = n_dict[j_node]
                queue.append(neighbor)
            elif dist_dict[neighbor] == dist_dict[j_node] + 1:
                n_dict[neighbor] += n_dict[j_node]

    return dist_dict, n_dict


def connected_components(g):
    """
    Find all connected components in g.

    Arguments:
    g -- undirected graph

    Returns:
    A list of sets where each set is all the nodes in
    a connected component.
    """
    # Initially we have no components and all nodes remain to be
    # explored.
    components = []
    remaining = set(g.keys())

    while remaining:
        # Randomly select a remaining node and find all nodes
        # connected to that node
        node = random.choice(list(remaining))
        distances = bfs(g, node)
        visited = set()
        for i in remaining:
            if distances[i] != float('inf'):
                visited.add(i)
        components.append(visited)

        # Remove all nodes in this component from the remaining
        # nodes
        remaining -= visited

    return components


def copy_graph(g):
    """
    Return a copy of the input graph, g

    Arguments:
    g -- a graph

    Returns:
    A copy of the input graph that does not share any objects.
    """
    return copy.deepcopy(g)


def gn_graph_partition(g):
    """
    Partition the graph g using the Girvan-Newman method.

    Requires connected_components, shortest_path_edge_betweenness, and
    compute_q to be defined.  This function assumes/requires these
    functions to return the values specified in the homework handout.

    Arguments:
    g -- undirected graph

    Returns:
    A list of tuples where each tuple contains a Q value and a list of
    connected components.
    """
    ### Start with initial graph
    c = connected_components(g)
    q = compute_q(g, c)
    partitions = [(q, c)]

    ### Copy graph so we can partition it without destroying original
    newg = copy_graph(g)

    ### Iterate until there are no remaining edges in the graph
    while True:
        ### Compute betweenness on the current graph
        btwn = shortest_path_edge_betweenness(newg)
        if not btwn:
            ### No information was computed, we're done
            break

        ### Find all the edges with maximum betweenness and remove them
        maxbtwn = max(btwn.values())
        maxedges = [edge for edge, b in btwn.iteritems() if b == maxbtwn]
        remove_edges(newg, maxedges)

        ### Compute the new list of connected components
        c = connected_components(newg)
        if len(c) > len(partitions[-1][1]):
            ### This is a new partitioning, compute Q and add it to
            ### the list of partitions.
            q = compute_q(g, c)
            partitions.append((q, c))

    return partitions


def _pow_10_round(n, up=True):
    """
    Round n to the nearest power of 10.

    Arguments:
    n  -- number to round
    up -- round up if True, down if False

    Returns:
    rounded number
    """
    if up:
        return 10 ** math.ceil(math.log(n, 10))
    else:
        return 10 ** math.floor(math.log(n, 10))


def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = data.keys()
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals


def _plot_dict_bar(d, xmin=None, label=None):
    """
    Plot data in the dictionary d on the current plot as bars.

    Arguments:
    d     -- dictionary
    xmin  -- optional minimum value for x axis
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if xmin == None:
        xmin = min(xvals) - 1
    else:
        xmin = min(xmin, min(xvals) - 1)
    if label:
        pylab.bar(xvals, yvals, align='center', label=label)
        pylab.xlim([xmin, max(xvals)+1])
    else:
        pylab.bar(xvals, yvals, align='center')
        pylab.xlim([xmin, max(xvals)+1])


def _plot_dict_scatter(d):
    """
    Plot data in the dictionary d on the current plot as points.

    Arguments:
    d     -- dictionary

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    pylab.scatter(xvals, yvals)


def _plot_dist(data, title, xlabel, ylabel, scatter, filename=None):
    """
    Plot the distribution provided in data.

    Arguments:
    data     -- dictionary which will be plotted with the keys
                on the x axis and the values on the y axis
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    scatter  -- True for loglog scatter plot, False for linear bar plot
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a dictionary
    if not isinstance(data, types.DictType):
        msg = "data must be a dictionary, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if scatter:
        _plot_dict_scatter(data)
    else:
        _plot_dict_bar(data, 0)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid
    gca = pylab.gca()
    gca.yaxis.grid(True)
    gca.xaxis.grid(False)

    if scatter:
        ### Use loglog scale
        gca.set_xscale('log')
        gca.set_yscale('log')
        gca.set_xlim([_pow_10_round(min([x for x in data.keys() if x > 0]), False),
                      _pow_10_round(max(data.keys()))])
        gca.set_ylim([_pow_10_round(min([x for x in data.values() if x > 0]), False),
                      _pow_10_round(max(data.values()))])

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)


def read_graph(filename):
    """
    Read a graph from a file.  The file is assumed to hold a graph
    that was written via the write_graph function.

    Arguments:
    filename -- name of file that contains the graph

    Returns:
    The graph that was stored in the input file.
    """
    with open(filename) as f:
        g = eval(f.read())
    return g


def read_attributes(filename):
    """
    Code to read student attributes from the file named filename.

    The attribute file should consist of one line per student, where
    each line is composed of student, college, year, major.  These are
    all anonymized, so each field is a number.  The student number
    corresponds to the node identifier in the Rice Facebook graph.

    Arguments:
    filename -- name of file storing the attributes

    Returns:
    A dictionary with the student numbers as keys, and a dictionary of
    attributes as values.  Each attribute dictionary contains
    'college', 'year', and 'major' as keys with the obvious associated
    values.
    """
    attributes = {}
    with open(filename) as f:
        for line in f:
            # Split line into student, college, year, major
            fields = line.split()
            student = int(fields[0])
            college = int(fields[1])
            year = int(fields[2])
            major = int(fields[3])

            # Store student in the dictionary
            attributes[student] = {'college': college,
                                   'year': year,
                                   'major': major}
    return attributes


def plot_dist_linear(data, title, xlabel, ylabel, filename=None):
    """
    Plot the distribution provided in data as a bar plot on a linear
    scale.

    Arguments:
    data     -- dictionary which will be plotted with the keys
                on the x axis and the values on the y axis
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    _plot_dist(data, title, xlabel, ylabel, False, filename)


###################
# Written Functions
###################

def compute_flow(g, dist, paths):
    """
    :param g: graph
    :param dist: dictionary of distances to each node
    :param paths: dictionary of # of shortest path to each node
    :return: a dictionary of edges to flow value
    """
    node_flow = defaultdict(lambda: float(1))
    edge_flow = dict()

    dist_tuple = []
    for dist_key, dist_val in dist.items():
        dist_tuple.append((dist_key, dist_val))

    dist_tuple = sorted(dist_tuple, key=lambda dict_dist: dict_dist[1])

    for dist_tuple in reversed(dist_tuple):
        src_node = dist_tuple[0]
        for neighbor in g[src_node]:
            if dist[neighbor] == dist[src_node] - 1:
                if neighbor in paths.keys():
                    edge_val = node_flow[src_node] * paths[neighbor] / paths[src_node]
                    edge_flow[frozenset([src_node, neighbor])] = edge_val
                    node_flow[neighbor] += edge_val

    return edge_flow


# Test cases:
# appendix = {0: {1, 2}, 1: {0, 3}, 2: {0, 3, 4}, 3: {1, 2, 5}, 4: {2, 5, 6}, 5: {3, 4}, 6: {4}}
# graph_320 = {'A':{'B','C','D','E'},
#              'B':{'A','C','F'},
#              'C':{'A','B','F'},
#              'D':{'A', 'H', 'G'},
#              'E':{'A', 'H'},
#              'F':{'B','C','I'},
#              'G':{'D','I','J'},
#              'H':{'D','E','J'},
#              'I':{'F','G','K'},
#              'J':{'G','H','K'},
#              'K':{'I','J'}}


# dist1, npath1 = bfs(appendix, 1)
# print dist1, npath1
# print compute_flow(appendix, dist1, npath1)

# dist2, npath2 = bfs(graph_320, 'A')
# print compute_flow(graph_320, dist2, npath2)


def shortest_path_edge_betweenness(g):
    """
    :param g: graph
    :return: dictionary of edge to betweenness value
    """
    bet_dict = defaultdict(float)

    for start_node in g.keys():
        dist, npath = bfs(g, start_node)
        flow_dict = compute_flow(g, dist, npath)
        # print start_node
        # print flow_dict

        for flow_key, flow_val in flow_dict.items():
            bet_dict[flow_key] += flow_val

    return bet_dict


def compute_q(g,c):
    """
    :param g: unconnected graph
    :param c: list of sets containing nodes
    :return: modularity measure
    """
    lin_alg = LinAl()
    matrix_len = len(c)

    # initialize matrix with zeros
    q_mod = [[0 for _ in range(matrix_len)] for _ in range(matrix_len)]
    edge_num = 0

    for node in g.keys():

        for comp_idx in range(len(c)):
            if node in c[comp_idx]:
                node_comp = comp_idx

        for neighbor in g[node]:
            if neighbor in c[node_comp]:
                neighbor_comp = node_comp
            else:
                for neigh_idx in range(len(c)):
                    if neighbor in c[neigh_idx]:
                        neighbor_comp = neigh_idx

            q_mod[node_comp][neighbor_comp] += 1

            if node_comp != neighbor_comp:
                q_mod[neighbor_comp][node_comp] += 1

            edge_num += 1

    for row in range(len(q_mod)):
        for col in range(len(q_mod[row])):
            q_mod[row][col] /= float(edge_num)

    q_trace = lin_alg.trace(q_mod)
    q_squared_sum = lin_alg.sum(lin_alg.multiply(q_mod, q_mod))

    return q_trace - q_squared_sum

# Test cases:
# test_graph314 = {1: {2, 3}, 2: {1, 3}, 3: {1, 2, 7}, 4: {5, 6}, 5: {4, 6},
#                  6: {4, 5, 7}, 7: {3, 6, 8}, 8: {7, 9, 12}, 9: {8, 10, 11},
#                  10: {9, 11}, 11: {9, 10}, 12: {8, 13, 14}, 13: {12, 14}, 14: {12, 13}}
# test_c314_1 = [set([8, 9, 10, 11, 12, 13, 14]), set([1, 2, 3, 4, 5, 6, 7])]
# print compute_q(test_graph314, test_c314_1)
# expect 0.381

# test_graph315 = {1: {2, 3}, 2: {1, 3, 4, 5}, 3: {1, 2, 4, 5}, 4: {2, 3, 5}, 5: {2, 3, 4, 6, 7},
#                  6: {5, 7}, 7: {5, 6, 8, 9, 10}, 8: {7, 9, 10}, 9: {7, 8, 10, 11}, 10: {7, 8, 9, 11},
#                  11: {9, 10}}
# test_c315_1 = [set([1, 2, 3, 4, 5]), set([8, 9, 10, 11, 7]), set([6])]
# print compute_q(test_graph315, test_c315_1)
# expect 0.277

# test_c315_2 = [set([8, 9, 10, 7]), set([2, 3, 4, 5]), set([1]), set([6]), set([11])]
# print compute_q(test_graph315, test_c315_2)
# expect 0.0443


########
# Plots
########

# Karate Plot
# part_g = gn_graph_partition(fig3_13g)
# q_val = [tup[0] for tup in part_g]
# q_key = [len(tup[1]) for tup in part_g]
# q_dict = dict(zip(q_key, q_val))
# _plot_dist(q_dict, 'Karate Club Q Val', 'Community Size', 'Q', False, 'Karate Q Plot')

# Facebook Plot
# facebook_g = read_graph('./rice-facebook.repr')
# facebook_attr = read_attributes('./rice-facebook-undergrads.txt')
# facebook_part = gn_graph_partition(facebook_g)
# q_val = [tup[0] for tup in facebook_part]
# q_key = [len(tup[1]) for tup in facebook_part]
# q_dict = dict(zip(q_key, q_val))
# plot_dist_linear(facebook_dict, 'Facebook Q Val', 'Community Size', 'Q Val', 'Facebook Plot')
# facebook_undergrad_part = gn_graph_partition(facebook_attr)
# print facebook_undergrad_part