import numpy
import random
import copy
import types
from matplotlib import pylab

def make_complete_graph(num_nodes):
    """
    Returns a complete graph containing num_nodes nodes.
 
    The nodes of the returned graph will be 0...(num_nodes-1) if num_nodes-1 is positive.
    An empty graph will be returned in all other cases.
 
    Arguments:
    num_nodes -- The number of nodes in the returned graph.
 
    Returns:
    A complete graph in dictionary form.
    """
    result = {}

    for node_key in range(num_nodes):
        result[node_key] = set()
        for node_value in range(num_nodes):
            if node_key != node_value:
                result[node_key].add(node_value)

    return result

def total_degree(g):
    """
    Compute total degree of the undirected graph g.

    Arguments:
    g -- undirected graph

    Returns:
    Total degree of all nodes in g
    """
    return sum(map(len, g.values()))

def upa(n, m):
    """
    Generate an undirected graph with n node and m edges per node
    using the preferential attachment algorithm.

    Arguments:
    n -- number of nodes
    m -- number of edges per node

    Returns:
    undirected random graph in UPAG(n, m)
    """
    g = {}
    if m <= n:
        g = make_complete_graph(m)
        for new_node in range(m, n):
            # Find <=m nodes to attach to new_node
            totdeg = float(total_degree(g))
            nodes = g.keys()
            probs = []
            for node in nodes:
                probs.append(len(g[node]) / totdeg)
            mult = distinct_multinomial(m, probs)

            # Add new_node and its random neighbors
            g[new_node] = set()
            for idx in mult:
                node = nodes[idx]
                g[new_node].add(node)
                g[node].add(new_node)
    return g    

def distinct_multinomial(ntrials, probs):
    """
    Draw ntrials samples from a multinomial distribution given by
    probs.  Return a list of indices into probs for all distinct
    elements that were selected.  Always returns a list with between 1
    and ntrials elements.

    Arguments:
    ntrials -- number of trials
    probs   -- probability vector for the multinomial, must sum to 1

    Returns: 
    A list of indices into probs for each element that was chosen one
    or more times.  If an element was chosen more than once, it will
    only appear once in the result.  
    """
    ### select ntrials elements randomly
    mult = numpy.random.multinomial(ntrials, probs)

    ### turn the results into a list of indices without duplicates
    result = [i for i, v in enumerate(mult) if v > 0]
    return result

def erdos_renyi(n, p):
    """
    Generate a random Erdos-Renyi graph with n nodes and edge probability p.

    Arguments:
    n -- number of nodes
    p -- probability of an edge between any pair of nodes

    Returns:
    undirected random graph in G(n, p)
    """
    g = {}

    ### Add n nodes to the graph
    for node in range(n):
        g[node] = set()

    ### Iterate through each possible edge and add it with 
    ### probability p.
    for u in range(n):
        for v in range(u+1, n):
            r = random.random()
            if r < p:
                g[u].add(v)
                g[v].add(u)

    return g

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

def write_graph(g, filename):
    """
    Write a graph to a file.  The file will be in a format that can be
    read by the read_graph function.

    Arguments:
    g        -- a graph
    filename -- name of the file to store the graph

    Returns:
    None
    """
    with open(filename, 'w') as f:
        f.write(repr(g))

def copy_graph(g):
    """
    Return a copy of the input graph, g

    Arguments:
    g -- a graph

    Returns:
    A copy of the input graph that does not share any objects.
    """
    return copy.deepcopy(g)

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

def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)

def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments: 
    data     -- a list of dictionaries, each of which will be plotted 
                as a line with the keys on the x axis and the values on
                the y axis.
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a list
    if not isinstance(data, types.ListType):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for i in range(len(data)-len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)

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

def random_attack(g):
    """
    Removes random node and any assiociated edges from graph
    
    Returns graph after attack
    """
    removed_node = random.choice(g.keys())

    if g == dict():
        return dict()

    #removes all edges going to it
    for edge in g[removed_node]:
        g[edge].remove(removed_node)
    
    g.pop(removed_node)
    return g

def targeted_attack(g):
    """
    Remove node with largest degree and any assiociated edges
    Specifies number of attacks
    Returns attacked graph
    """
    if g == dict():
        return dict()

    max_deg = 0

    #find max degree
    for node in g.keys():
        node_len = len(g[node])
        if node_len > max_deg:
            max_deg = node_len

    #removes largest node n times

        #looks for node with largest degree and removes it
    for node in g.keys():
        if len(g[node]) == max_deg:
            removed_node = node
    for edge in g[removed_node]:
        g[edge].remove(removed_node)
    g.pop(removed_node)
    return g

rf7_graph = read_graph('rf7.repr')
rf7_copy1 = copy_graph(rf7_graph)
rf7_copy2 = copy_graph(rf7_graph)
# rf7 graph has 1347 nodes and 3112 edges

# formula is number of edges / possible number of edges
# proability calculated by 3112/((1347*(1347-1))/2) = 0.00343
er_graph = erdos_renyi(1347, 0.00343)
er_copy1 = copy_graph(er_graph)
er_copy2 = copy_graph(er_graph)

# 3112 / 1347 = 2.31
upa_graph = upa(1347, 2)
upa_copy1 = copy_graph(upa_graph)
upa_copy2 = copy_graph(upa_graph)

# 20% of nodes = 270 nodes
attack_plot = []
er_random = dict()
er_targeted = dict()
upa_random = dict()
upa_targeted = dict()
rf7_random = dict()
rf7_targeted = dict()

for nodes_removed in range(270):

    # er graph with random attack
    er_copy1 = random_attack(er_copy1)
    er_random[nodes_removed] = compute_largest_cc_size(er_copy1)

    # er graph with targeted attack
    er_copy2 = targeted_attack(er_copy2)
    er_targeted[nodes_removed] = compute_largest_cc_size(er_copy2)

    # upa graph with random attack
    upa_copy1 = random_attack(upa_copy1)
    upa_random[nodes_removed] = compute_largest_cc_size(upa_copy1)

    # upa graph with targeted attack
    upa_copy2 = targeted_attack(upa_copy2)
    upa_targeted[nodes_removed] = compute_largest_cc_size(upa_copy2)

    # rf7 graph with random attack
    rf7_copy1 = random_attack(rf7_copy1)
    rf7_random[nodes_removed] = compute_largest_cc_size(rf7_copy1)

    # rf7 graph with targeted attack
    rf7_copy2 = targeted_attack(rf7_copy2)
    rf7_targeted[nodes_removed] = compute_largest_cc_size(rf7_copy2)

attack_plot = [er_random, er_targeted, upa_random, upa_targeted, rf7_random, rf7_targeted]
plot_names = ['ER Random Attack', 'ER Targeted Attack', 'UPA Random Attack', 'UPA Targeted Attack', 'rf7 Random Attack', 'rf7 Targeted Attack']
plot_lines(attack_plot, 'Network Attack Analysis With 1437 Starting Nodes', '# of nodes', 'largest cc', plot_names, 'analysis plot')
