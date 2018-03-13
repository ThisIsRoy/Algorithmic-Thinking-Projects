import random
import math
import random
import numpy
from collections import deque
from collections import defaultdict
import types
import pylab

# Modified functions from compute_alignment
def compute_max_global_alignment_scores(X, Y, M):
	"""
	:param X: sequence of DNA
	:param Y: sequene of DNA
	:param M: scoring matrix as a nested dictionary
	:return: opitmal score matrix S
	"""
	x_len = len(X) + 1
	y_len = len(Y) + 1
	S = [[0 for _ in range(y_len)] for _ in range(x_len)]

	# initialize first row and first column
	for row_idx in range(1, x_len):
		S[row_idx][0] = S[row_idx - 1][0] + M[X[row_idx - 1]]['-']

	for col_idx in range(1, y_len):
		S[0][col_idx] = S[0][col_idx - 1] + M['-'][Y[col_idx - 1]]

	# use DP to create rest of matrix
	for row_idx in range(1, x_len):
		for col_idx in range(1, y_len):
			diagonal = S[row_idx - 1][col_idx - 1] + M[X[row_idx - 1]][Y[col_idx - 1]]
			horizontal  = S[row_idx][col_idx - 1] + M['-'][Y[col_idx - 1]]
			vertical = S[row_idx - 1][col_idx] + M[X[row_idx - 1]]['-']
			S[row_idx][col_idx] = max([diagonal, horizontal, vertical])

	return S[len(X)][len(Y)]


def compute_local_alignment_max_score(X, Y, M):
    """
    :param X: sequence of DNA
    :param Y: sequence of DNA
    :param M: scoring matrix as a nested dictionary
    :return: optimal score matrix S
    """
    x_len = len(X) + 1
    y_len = len(Y) + 1
    S = [[0 for _ in range(y_len)] for _ in range(x_len)]

    # initialize first row and first column
    for row_idx in range(1, x_len):
        row_score = S[row_idx - 1][0] + M[X[row_idx - 1]]['-']
        S[row_idx][0] = row_score if row_score > 0 else 0

    for col_idx in range(1, y_len):
        col_score = S[0][col_idx - 1] + M['-'][Y[col_idx - 1]]
        S[0][col_idx] = col_score if col_score > 0 else 0

    # use DP to create rest of matrix
    for row_idx in range(1, x_len):
        for col_idx in range(1, y_len):
            diagonal = S[row_idx - 1][col_idx - 1] + M[X[row_idx - 1]][Y[col_idx - 1]]
            horizontal = S[row_idx][col_idx - 1] + M['-'][Y[col_idx - 1]]
            vertical = S[row_idx - 1][col_idx] + M[X[row_idx - 1]]['-']
            max_score = max([diagonal, horizontal, vertical])
            S[row_idx][col_idx] = max_score if max_score > 0 else 0

    for row in range(len(S)):
        for col in range(len(S[row])):
            if S[row][col] > max_score:
                max_score = S[row][col]

    return max_score

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


def read_protein(filename):
    """
    Read a protein sequence from the file named filename.

    Arguments:
    filename -- name of file containing a protein sequence

    Returns:
    A string representing the protein
    """
    with open(filename) as f:
        p = f.read()
    p = p.rstrip()
    return p


def read_scoring_matrix(filename):
    """
    Read a scoring matrix from the file named filename.

    Argument:
    filename -- name of file containing a scoring matrix

    Returns:
    A dictionary of dictionaries mapping X and Y characters to scores
    """
    M = {}
    with open(filename) as f:
        ykeys = f.readline()
        ykeychars = ykeys.split()
        for line in f.readlines():
            vals = line.split()
            xkey = vals.pop(0)
            M[xkey] = {}
            for ykey, val in zip(ykeychars, vals):
                M[xkey][ykey] = int(val)
    return M


def permute_string(s):
    """
    Return a new string with the characters in s randomly permuted.

    Arguments:
    s -- string

    Returns:
    Random permutation of s
    """
    charlist = list(s)
    random.shuffle(charlist)
    newstr = "".join(charlist)
    return newstr


def generate_null_distribution(x, y, M, t, numiter):
    """
    :param x: DNA sequence
    :param y: DNA sequence
    :param M: scoring matrix
    :param t: time
    :param numiter: time of random samples
    :return: mean, standard deviation, and the z-score of the scores of the randomly generated sequences
    """
    score_list = []

    if t == 0:
        s = compute_max_global_alignment_scores(x, y, M)

    elif t == 1:
        s = compute_local_alignment_max_score(x, y, M)

    else:
        print 'error in t val'
        return

    # generate randomized y sequence and compute max alignment score with x
    for _ in range(numiter):
        random_y = permute_string(y)

        if t == 0:
            score = compute_max_global_alignment_scores(x, random_y, M)

        elif t == 1:
            score = compute_local_alignment_max_score(x, random_y, M)

        else:
            print 'error in t val'
            return

        score_list.append(score)

    # plots the data
    plot_data = dict(enumerate(score_list, 1))
    print plot_data
    _plot_dist(plot_data, 'Scores for Randomized Alignment', 'Randomized Sequences', 'Scores', False, 'Null Distribution Plot')

    # calculate mean, standard deviation, and z-score compared to non-randomized alignment
    print score_list
    mean = numpy.mean(score_list)
    std = numpy.std(score_list)
    z = (float(s) - mean) / std

    return mean, std, z

# Plot fruitfly DNA vs human DNA
# fruitfly_sequence = read_protein('FruitflyEyelessProtein')
# human_sequence = read_protein('HumanEyelessProtein')
# consensus_sequence = read_protein('ConsensusPAXDomain')
# M = read_scoring_matrix('PAM50')
# u, v = ('-HSGVNQLGGVFVNGRPLPDSTRQKIVELAHSGARPCDISRILQVSNGCVSKILGRYYETGSIRPRAIGGSKPRVATPEVVSKIAQYKRECPSIFAWEIRDRLLSEGVCTNDNIPSVSSINRVLRNLASEK-QQ', 'GHGGVNQLGGVFVNGRPLPDVVRQRIVELAHQGVRPCDISRQLRVSHGCVSKILGRYYETGSIKPGVIGGSKPKVATPKVVEKIAEYKRQNPTMFAWEIRDRLLAERVCDNDTVPSVSSINRII-------R--')

# print generate_null_distribution(human_sequence, fruitfly_sequence, M, 1, 100)

# print generate_null_distribution(fruitfly_sequence, human_sequence, M, 0, 100)

# z-score for global: 38.801
# z-score for local: 138.882