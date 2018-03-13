from collections import *
from copy import *

def compute_rdmst_helper(graph, root):
    """
        Computes the RDMST of a weighted digraph rooted at node root.
        It is assumed that:
        (1) root is a node in graph, and
        (2) every other node in graph is reachable from root.

        Arguments:
        graph -- a weighted digraph in standard dictionary representation.
        root -- a node in graph.

        Returns:
        An RDMST of graph rooted at root. The weights of the RDMST
        do not have to be the original weights.
        """

    # reverse the representation of graph
    rgraph = reverse_digraph_representation(graph)

    # Step 1 of the algorithm
    modify_edge_weights(rgraph, root)

    # Step 2 of the algorithm
    rdst_candidate = compute_rdst_candidate(rgraph, root)

    # compute a cycle in rdst_candidate
    cycle = compute_cycle(rdst_candidate)

    # Step 3 of the algorithm
    if not cycle:
        return reverse_digraph_representation(rdst_candidate)
    else:
        # Step 4 of the algorithm

        g_copy = deepcopy(rgraph)
        g_copy = reverse_digraph_representation(g_copy)

        # Step 4(a) of the algorithm
        (contracted_g, cstar) = contract_cycle(g_copy, cycle)
        # cstar = max(contracted_g.keys())

        # Step 4(b) of the algorithm
        new_rdst_candidate = compute_rdmst_helper(contracted_g, root)

        # Step 4(c) of the algorithm
        rdmst = expand_graph(reverse_digraph_representation(rgraph), new_rdst_candidate, cycle, cstar)

        return rdmst


def bfs(graph, startnode):
    """
        Perform a breadth-first search on digraph graph starting at node startnode.

        Arguments:
        graph -- directed graph
        startnode - node in graph to start the search from

        Returns:
        The distances from startnode to each node
    """
    dist = {}

    # Initialize distances
    for node in graph:
        dist[node] = float('inf')
    dist[startnode] = 0

    # Initialize search queue
    queue = deque([startnode])

    # Loop until all connected nodes have been explored
    while queue:
        node = queue.popleft()
        for nbr in graph[node]:
            if dist[nbr] == float('inf'):
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return dist


def reverse_digraph_representation(graph):
    """
    :param graph: weighted digraph
    :return: same digraph but in reverse representation
    """
    reversed_graph = defaultdict(dict)
    for node in graph.keys():
        reversed_graph[node] = {}

    # iterates through edges to reverse head and tail
    for head, edges in graph.items():
        for tail, weight in edges.items():
            reversed_graph[tail][head] = weight

    return dict(reversed_graph)

# Test cases:
# test_g = {1: {2: 3, 3: 1}, 2: {1: 4, 3: 5}, 3: {1: 6, 2: 9}}
# print reverse_digraph_representation(test_g)

# test_g2 = g = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8},
# 4: {1: 4}, 5: {}}
# print reverse_digraph_representation(test_g2)

def modify_edge_weights(rgraph, root):
    """
    :param rgraph: weighted digraph
    :param root: root node of digraph
    :return: input digraph with modified edge weights
    """

    for node in rgraph.keys():

        if node != root:
            # finds min weight excluding root node
            outnode_weights = [weight for weight in rgraph[node].values()]

            # outnode_weights = []
            # for tail, edges in rgraph.items():
            #     for head, weight in edges.items():
            #         outnode_weights.append(weight)

            # print outnode_weights
            min_income_edge = min(outnode_weights) if len(outnode_weights) > 0 else 0

            # subtracts min weight from all incoming edges
            for head in rgraph[node].keys():
                rgraph[node][head] -= min_income_edge

    return rgraph

# Test cases:
# fig3 = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}
# reverse_fig3 = reverse_digraph_representation(fig3)
# print reverse_digraph_representation(modify_edge_weights(reverse_fig3, 0))


def compute_rdst_candidate(rgraph, root):
    """
    :param rgraph: reversed digraph
    :param root: root node
    :return: RDST of the digraph in the reverse representation
    """
    span_tree = defaultdict(dict)

    for tail, edges in rgraph.items():
        if tail != root:
            if edges.values():
                min_weight = min(edges.values())
                for head, weight in edges.items():
                    if weight == min_weight:
                        span_tree[tail][head] = weight
                        break


    return span_tree

# Test cases:
# reverse_fig3 = {0: {}, 1: {0: 20, 4: 4}, 2: {0: 4, 1: 2}, 3: {0: 20, 2: 8}, 4: {2: 20, 3: 4}}
# reverse_mod_fig3 = modify_edge_weights(reverse_fig3, 0)
# print reverse_digraph_representation(compute_rdst_candidate(reverse_mod_fig3, 0))


def compute_cycle(rdst_candidate):
    normal_rdst = reverse_digraph_representation(rdst_candidate)
    visited = dict()
    nodes = normal_rdst.keys()
    for node in nodes:
        visited[node] = False

    for node in nodes:
        queue = []
        visited_copy = dict(visited)
        parent = dict()
        queue.append(node)
        visited_copy[node] = True

        while queue:
            curr_node = queue.pop()
            for neighbor in normal_rdst[curr_node].keys():
                if not visited_copy[neighbor]:
                    visited_copy[neighbor] = True
                    queue.append(neighbor)
                    parent[neighbor] = curr_node

                else:
                    parent[neighbor] = curr_node
                    cycle = [neighbor]
                    cycle_node = curr_node


                    while cycle_node != neighbor:
                        cycle.insert(1, cycle_node)
                        cycle_node = parent[cycle_node]

                    return tuple(cycle)
    return None

# Test cases:
# test_g1 = {0: {1: 1}, 1: {2: 3}, 2: {3: 5}, 3: {1: 3}}
# print compute_cycle(test_g1)
# test_g2 = {1: {2: 1, 3: 1}, 2: dict(), 3: {4: 1, 5: 1}, 4: dict(), 5: {6: 1}, 6: {1: 1}}
# print compute_cycle(test_g2)


def contract_cycle(graph, cycle):
    r_graph = reverse_digraph_representation(graph)
    graph_cont = deepcopy(graph)
    cstar = max(graph.keys()) + 1
    graph_cont[cstar] = dict()

    unique_cdict = defaultdict(lambda: float('inf'))
    # find and add all outgoing edges from c-star
    for cycle_node in cycle:
        for tail, weight in graph[cycle_node].items():
            if tail not in cycle and weight < unique_cdict[tail]:
                unique_cdict[tail] = weight

    graph_cont[cstar] = dict(unique_cdict)

    # compile all incoming edges to cycle to edge_in
    incoming_head = None
    incoming_weight = float('inf')
    for cycle_node in cycle:
        for head, weight in r_graph[cycle_node].items():
            if head not in cycle and weight < incoming_weight:
                incoming_head = head
                incoming_weight = weight

    graph_cont[incoming_head][cstar] = incoming_weight

    # remove all cycle nodes and outgoing edges
    for node in cycle:
        graph_cont.pop(node)

    # remove all incoming edges to cycle nodes
    for head, edges in graph_cont.items():
        for tail in edges.keys():
            if tail in cycle:
                graph_cont[head].pop(tail)

    return graph_cont, cstar

# Test cases:
# fig_3a = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8}, 4: {1 : 4}, 5: {}}
# print contract_cycle(fig_3a, (1, 2, 3, 4))

# test2 = {0: {1: 3, 2: 2}, 1: {2: 0}, 2: {1: 0}}
# print contract_cycle(test2, (1, 2))

def expand_graph(graph, rdst_candidate, cycle, cstar):
    # make expanded graph a nested dict copy of the rdst
    r_graph = reverse_digraph_representation(graph)
    expanded_graph = defaultdict(dict)
    for head, edges in rdst_candidate.items():
        if edges == dict():
            expanded_graph[head] = dict()
        else:
            for tail, weight in edges.items():
                expanded_graph[head][tail] = weight
    if cstar in expanded_graph.keys():
        expanded_graph.pop(cstar)
    reverse_rdst = reverse_digraph_representation(rdst_candidate)

    # find minimum weight incoming edge
    incoming_head = reverse_rdst[cstar].keys()[0]
    incoming_tail = None
    incoming_weight = float('inf')
    for tail, weight in graph[incoming_head].items():
        if tail in cycle and weight < incoming_weight:
            incoming_tail = tail
            incoming_weight = weight

    # replace outgoing edges
    cstar_tails = rdst_candidate[cstar].keys()
    for tail in cstar_tails:
        outgoing_weight = float('inf')
        outgoing_head = None
        for head, weight in r_graph[tail].items():
            if head in cycle and weight < outgoing_weight:
                outgoing_weight = weight
                outgoing_head = head

        expanded_graph[outgoing_head][tail] = outgoing_weight


    expanded_graph[incoming_head].pop(cstar)
    expanded_graph[incoming_head][incoming_tail] = incoming_weight

    # expand outgoing cycle edges and cycle edges except for vstar
    for cycle_node in cycle:


        for tail, weight in graph[cycle_node].items():
            if tail in cycle and tail != incoming_tail:
                expanded_graph[cycle_node][tail] = weight

    return expanded_graph

# Test cases:
# fig_3a = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8}, 4: {1 : 4}, 5: {}}
# reverse_fig_3a = reverse_digraph_representation(fig_3a)
# mod_fig_3a = reverse_digraph_representation(modify_edge_weights(reverse_fig_3a, 0))
# rdst_candidate = {0: {6: 2}, 6: {5: 0}, 5: {}}
# cycle = (1, 2, 3, 4)
# cstar = 6
# print expand_graph(mod_fig_3a, rdst_candidate, cycle, cstar)

def compute_rdmst(graph, root):
    """
        This function checks if:
        (1) root is a node in digraph graph, and
        (2) every node, other than root, is reachable from root
        If both conditions are satisfied, it calls compute_rdmst_helper
        on (graph, root).

        Since compute_rdmst_helper modifies the edge weights as it computes,
        this function reassigns the original weights to the RDMST.

        Arguments:
        graph -- a weighted digraph in standard dictionary representation.
        root -- a node id.

        Returns:
        An RDMST of graph rooted at r and its weight, if one exists;
        otherwise, nothing.
    """

    if root not in graph:
        print "The root node does not exist"
        return

    distances = bfs(graph, root)
    for node in graph:
        if distances[node] == float('inf'):
            print "The root does not reach every other node in the graph"
            return

    rdmst = compute_rdmst_helper(graph, root)

    # reassign the original edge weights to the RDMST and computes the total
    # weight of the RDMST
    rdmst_weight = 0
    for node in rdmst:
        for nbr in rdmst[node]:
            rdmst[node][nbr] = graph[node][nbr]
            rdmst_weight += rdmst[node][nbr]

    return (rdmst, rdmst_weight)

# Test cases:
# g0 = {0: {1: 2, 2: 2, 3: 2}, 1: {2: 2, 5: 2}, 2: {3: 2, 4: 2}, 3: {4: 2, 5: 2}, 4: {1: 2}, 5: {}}
# print compute_rdmst(g0, 0)

# g1 = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}
# print compute_rdmst(g1, 0)

# g2 = {0: {1: 5, 2: 4}, 1: {2: 2}, 2: {1: 2}}
# print compute_rdmst(g2, 0)

# g3 = {1: {2: 2.1, 3: 1.0, 4: 9.1, 5: 1.1}, 2: {1: 2.1, 3: 1.0, 4: 17.0, 5: 1.0}, 3: {1: 1.0, 2: 1.0, 4: 16.0, 5: 0.0}, 4: {1: 9.1, 2: 17.1, 3: 16.0, 5: 16.0}, 5: {1: 1.1, 2: 1.0, 3: 0.0, 4: 16.0}}
# print compute_rdmst(g3, 1)

# g4 = {1: {2: 2.1, 3: 1.0, 4: 9.1, 5: 1.1, 6: 10.1, 7: 10.1, 8: 6.1, 9: 11.0, 10: 10.1}, 2: {1: 2.1, 3: 1.0, 4: 17.0, 5: 1.0, 6: 18.1, 7: 18.1, 8: 14.1, 9: 19.1, 10: 18.0}, 3: {1: 1.0, 2: 1.0, 4: 16.0, 5: 0.0, 6: 17.0, 7: 17.0, 8: 13.1, 9: 18.1, 10: 17.0}, 4: {1: 9.1, 2: 17.1, 3: 16.0, 5: 16.0, 6: 5.1, 7: 5.1, 8: 15.1, 9: 6.1, 10: 5.0}, 5: {1: 1.1, 2: 1.0, 3: 0.0, 4: 16.0, 6: 17.1, 7: 17.1, 8: 13.1, 9: 18.1, 10: 17.0}, 6: {1: 10.1, 2: 18.1, 3: 17.0, 4: 5.1, 5: 17.1, 7: 0.0, 8: 16.1, 9: 7.1, 10: 0.0}, 7: {1: 10.1, 2: 18.1, 3: 17.0, 4: 5.1, 5: 17.1, 6: 0.0, 8: 16.0, 9: 7.1, 10: 0.0}, 8: {1: 6.1, 2: 14.1, 3: 13.1, 4: 15.1, 5: 13.1, 6: 16.1, 7: 16.0, 9: 17.1, 10: 16.1}, 9: {1: 11.1, 2: 19.1, 3: 18.1, 4: 6.1, 5: 18.1, 6: 7.1, 7: 7.1, 8: 17.1, 10: 7.0}, 10: {1: 10.1, 2: 18.1, 3: 17.1, 4: 5.1, 5: 17.0, 6: 0.0, 7: 0.0, 8: 16.1, 9: 7.0}}
# print compute_rdmst(g4, 1)
