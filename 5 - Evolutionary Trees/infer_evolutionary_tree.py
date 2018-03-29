#################   DO NOT EDIT THESE IMPORTS #################
import abc
import random
from copy import *
from collections import *


#################   PASTE PROVIDED CODE HERE AS NEEDED   #################
import abc


class FullBiTree(object):
    """
    Represents a full binary tree.
    """

    def __init__(self, name, left_tree=None, right_tree=None):
        """
        Creates a full binary tree.

        This constructor must be called with exactly one or three parameters.
        That is, a name alone or a name and both a left and right child.

        Arguments:
        name - an identifier for the root node of the tree.
        left_tree - the FullBiTree left substree if the tree's root has children. (optional)
        right_tree - the FullBiTree left substree if the tree's root has children. (optional)
        """

        self.__name = name
        self.__node_props = {}
        if left_tree == None and right_tree == None:
            self.__set_state(TreeNodeStateLeaf())
        elif left_tree != None and right_tree != None:
            self.__set_state(TreeNodeStateInternal(left_tree, right_tree))
        else:
            raise Exception('FullBiTree roots must have 0 or 2 children.')

    def get_name(self):
        """
        Gets the name of the root node of the tree.

        Returns:
        The name of the root node.
        """
        return self.__name

    def get_left_child(self):
        """
        Gets the left subtree of the tree's root if it has children or generates an exception if the root has no children.

        Returns:
        The left subtree of the tree.
        """
        return self.__get_state().get_left_child()

    def get_right_child(self):
        """
        Gets the right subtree of the tree's root if it has children or generates an exception if the root has no children.

        Returns:
        The left subtree of the tree.
        """
        return self.__get_state().get_right_child()

    def set_children(self, left_tree, right_tree):
        """
        Updates the tree's root to contain new children.

        Arguments:
        left_tree - the new left subtree for the tree.
        right_tree - the new right subtree for the tree.
        """
        self.__set_state(TreeNodeStateInternal(left_tree, right_tree))

    def remove_children(self):
        """
        Updates the tree's root to contain no children.

        Arguments:
        left_tree - the new left subtree for the tree.
        right_tree - the new right subtree for the tree.
        """
        self.__set_state(TreeNodeStateLeaf())

    def is_leaf(self):
        """
        Tests whether the tree's root has no children.

        Returns:
        True if the tree is only a single node, else false.
        """
        return self.__get_state().is_leaf()

    def __set_state(self, new_state):
        """
        Sets the internal node/leaf node state for the node.

        Arguments:
        new_state - the new node state.
        """
        self.__node_state = new_state

    def __get_state(self):
        """
        Gets the internal node/leaf node state for the node.

        Returns:
        The current node state.
        """
        return self.__node_state

    def __str__(self):
        " Contract from super. "
        return self.__get_state().to_string(self)

    def get_node_property(self, key):
        """
        Accesses a user specified property of the tree's root.

        Arguments:
        key - the property of the desired key value pair.

        Returns:
        The value of the given key for the tree's root.
        """
        return self.__node_props[key]

    def set_node_property(self, key, value):
        """
        Defines a user specified property of the tree's root.

        Arguments:
        key - the key of the desired property.
        value - the value of the desired property.
        """
        self.__node_props[key] = value

    def get_left_edge_property(self, key):
        """
        Accesses a user specified property of the tree's left subtree edge.
        Throws exception if the tree has no left subtree.

        Arguments:
        key - the property of the desired key value pair.

        Returns:
        The value of the given key for the tree's left subtree edge.
        """
        return self.__get_state().get_left_edge_property(key)

    def set_left_edge_property(self, key, value):
        """
        Defines a user specified property of the tree's left subtree edge.
        Throws exception if the tree has no left subtree.

        Arguments:
        key - the key of the desired property.
        value - the value of the desired property.
        """
        self.__get_state().set_left_edge_property(key, value)

    def get_right_edge_property(self, key):
        """
        Accesses a user specified property of the tree's right subtree edge.
        Throws exception if the tree has no left subtree.

        Arguments:
        key - the property of the desired key value pair.

        Returns:
        The value of the given key for the tree's right subtree edge.
        """
        return self.__get_state().get_right_edge_property(key)

    def set_right_edge_property(self, key, value):
        """
        Defines a user specified property of the tree's right subtree edge.
        Throws exception if the tree has no left subtree.

        Arguments:
        key - the key of the desired property.
        value - the value of the desired property.
        """
        self.__get_state().set_right_edge_property(key, value)


class TreeNodeState(object):
    """
    Abstract class for defining all operations for a node state.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_leaf(self):
        """
        Tests whether the node state represents a leaf.

        Returns:
        True if the node state represents a leaf, else false.
        """
        pass

    @abc.abstractmethod
    def to_string(self, owner):
        """
        Returns a prefix string representation of the whole tree rooted by the node state.

        Returns:
        A prefix string representation of the tree.
        """
        pass

    @abc.abstractmethod
    def get_left_child(self):
        """
        Returns the left child of this node if in the internal state, or generate exeption if in leaf state.

        Returns:
        The left subtree.
        """
        pass

    @abc.abstractmethod
    def get_right_child(self):
        """
        Returns the right child of this node if in the internal state, or generate exeption if in leaf state.

        Returns:
        The right subtree.
        """
        pass

    @abc.abstractmethod
    def get_left_edge_property(self, key):
        """
        Accesses a user specified property of the node state's left subtree edge.
        Throws exception if the tree has no left subtree.

        Arguments:
        key - the property of the desired key value pair.

        Returns:
        The value of the given key for the tree's left subtree edge.
        """
        pass

    @abc.abstractmethod
    def set_left_edge_property(self, key, value):
        """
        Accesses a user specified property of the node state's left subtree edge.
        Throws exception if the node state has no left subtree.

        Arguments:
        key - the property of the desired key value pair.

        Returns:
        The value of the given key for the tree's right subtree edge.
        """
        pass

    @abc.abstractmethod
    def get_right_edge_property(self, key):
        """
        Accesses a user specified property of the node state's right subtree edge.
        Throws exception if the tree has no right subtree.

        Arguments:
        key - the property of the desired key value pair.

        Returns:
        The value of the given key for the tree's right subtree edge.
        """
        pass

    @abc.abstractmethod
    def set_right_edge_property(self, key, value):
        """
        Accesses a user specified property of the node state's right subtree edge.
        Throws exception if the node state has no left subtree.

        Arguments:
        key - the property of the desired key value pair.

        Returns:
        The value of the given key for the tree's right subtree edge.
        """
        pass


class TreeNodeStateLeaf(TreeNodeState):
    """
    TreeNodeState representing a leaf.
    """

    def is_leaf(self):
        "Contract from super."
        return True

    def to_string(self, owner):
        "Contract from super."
        return str(owner.get_name())

    def get_left_child(self):
        "Contract from super."
        raise Exception("A leaf does not have a left child.")

    def get_right_child(self):
        "Contract from super."
        raise Exception("A leaf does not have a right child.")

    def get_left_edge_property(self, key):
        "Contract from super."
        raise Exception("A leaf does not have a left edge.")

    def set_left_edge_property(self, key, value):
        "Contract from super."
        raise Exception("A leaf does not have a left edge.")

    def get_right_edge_property(self, key):
        "Contract from super."
        raise Exception("A leaf does not have a right edge.")

    def set_right_edge_property(self, key, value):
        "Contract from super."
        raise Exception("A leaf does not have a right edge.")


class TreeNodeStateInternal(TreeNodeState):
    """
    TreeNodeState for an internal node.
    """

    def __init__(self, left_tree, right_tree):
        """
        Creates a new TreeNodeState instance.

        Arguments:
        left_tree - The FullBiTree left subtree of this node.
        right_tree - The FullBiTree right subtree of this node.
        """
        self.__left_tree = left_tree
        self.__right_tree = right_tree
        self.__left_edge_props = {}
        self.__right_edge_props = {}

    def is_leaf(self):
        "Contract from super."
        return False

    def get_left_child(self):
        "Contract from super."
        return self.__left_tree;

    def get_right_child(self):
        "Contract from super."
        return self.__right_tree

    def get_left_edge_property(self, key):
        "Contract from super."
        return self.__left_edge_props[key]

    def set_left_edge_property(self, key, value):
        "Contract from super."
        self.__left_edge_props[key] = value

    def get_right_edge_property(self, key):
        "Contract from super."
        return self.__right_edge_props[key]

    def set_right_edge_property(self, key, value):
        "Contract from super."
        self.__right_edge_props[key] = value

    def to_string(self, owner):
        "Contract from super."
        return str(owner.get_name()) + '(' + str(self.get_left_child()) + ', ' + str(self.get_right_child()) + ')'


def test_tree():
    tree = FullBiTree('A', FullBiTree('B'), FullBiTree('C'))

    if tree.is_leaf():
        raise Exception('failed test 1')

    if 'B' != str(tree.get_left_child()):
        raise Exception('failed test 2')
    if 'C' != str(tree.get_right_child()):
        raise Exception('failed test 3')

    if 'A(B, C)' != str(tree):
        raise Exception('failed test 4')
    d = FullBiTree('D')

    if not d.is_leaf():
        raise Exception('failed test 5')

    tree.set_children(tree.get_left_child(), d)
    if 'A(B, D)' != str(tree):
        raise Exception('failed test 6')
    r = FullBiTree('R')
    if 'R' != str(r):
        raise Exception('failed test 7')
    r.set_children(d, FullBiTree('E'))
    if 'R(D, E)' != str(r):
        raise Exception('failed test 8')
    if r.is_leaf():
        raise Exception('failed test 9')

    r.set_node_property('dog', 'cat')
    if not r.get_node_property('dog') == 'cat':
        raise Exception('failed test 10')

    x = FullBiTree('X', d, d)
    x.remove_children()
    if 'X' != str(x):
        raise Exception('failed test 11')

    if not x.is_leaf():
        raise Exception('failed test 12')

    tree2 = FullBiTree('A', FullBiTree('B'), FullBiTree('C'))
    tree2.set_left_edge_property(True, True)
    if not tree2.get_left_edge_property(True) == True:
        raise Exception('failed test 13')
    tree2.set_right_edge_property(False, False)
    if not tree2.get_right_edge_property(False) == False:
        raise Exception('failed test 13')

    tree2.set_children(tree2.get_left_child, FullBiTree('Z'))
    try:
        tree2.get_right_edge_property(False)
        raise Exception('failed test 14')
    except Exception as e:
        pass


"""
Counts the number of leaves in a tree in dict form.
"""


def count_leaves_full_bitree_dict(tree):
    num_leaves = 0
    for adj_set in tree.values():
        adj_set_size = len(adj_set)
        if adj_set_size == 0 or adj_set_size == 1:
            num_leaves += 1

    return num_leaves


"""
Counts the number of leaves in a FullBiTree
"""


def count_leaves_fullbitree(tree):
    if tree.is_leaf():
        return 1
    else:
        return count_leaves_full_bitree_dict(tree.get_right_child()) + \
               count_leaves_full_bitree_dict(tree.get_left_child())


"""
Computes the hight of a FullBiTree.

Arguments:
tree - a full binary tree in FullBiTree form.

Returns:
The height of the tree
"""


def tree_height(tree):
    if tree.is_leaf():
        return 0
    else:
        left_height = tree_height(tree.get_left_child())
        right_height = tree_height(tree.get_right_child())
        if left_height > right_height:
            return left_height + 1
        else:
            return right_height + 1


def infix_string(tree):
    """
    Computes the infix order string of a tree.

    Arguments:
    tree - a full binary tree in FullBiTree form.

    Returns:
    An infix string of the tree.
    """
    if tree.is_leaf():
        return tree.get_name()
    else:
        return infix_string(tree.get_left_child()) + tree.get_name() + infix_string(tree.get_right_child())


def find_paths(tree):
    """
    Computes a string for each path in the givn tree starting at the root and terminating at a leaf.

    Arguments:
    tree - a FullBiTree

    Returns:
    A set of strings encoding the order of nodes in each path from the root to all leaves.
    """
    found_paths = set()
    find_paths_help(tree, "", found_paths)
    return found_paths


def find_paths_help(tree, path_so_far, found_paths):
    """
    Computes a string for each path in the given sub-tree starting at the sub-tree root and terminating at a leaf.
    Stores completed paths in the given set

    Arguments:
    tree - a FullBiTree
    path_so_far - a string encoding the path seen so far from the global tree root to this sub-tree root.
    found_paths - all complete paths seen so far from the tree root to a leaf.
    """
    if tree.is_leaf():
        path = path_so_far + tree.get_name()
        found_paths.add(path)
    else:
        find_paths_help(tree.get_left_child(), path_so_far + tree.get_name(), found_paths)
        find_paths_help(tree.get_right_child(), path_so_far + tree.get_name(), found_paths)


def is_valid_bst(tree):
    """
    Tests to see if the given tree has the binary search property.

    Arguments:
    tree - a FullBiTree where the value of each node is an integer stored as the node's name.

    Returns:
    True if the tree has the binary search property, else false.
    """
    infix_list = list()
    is_valid_bst_help(tree, infix_list)

    prev_element = infix_list[0]

    for element in infix_list:
        if element < prev_element:
            return False
        prev_element = element

    return True


def is_valid_bst_help(tree, infix_list):
    if tree.is_leaf():
        infix_list.append(tree.get_name())
    else:
        is_valid_bst_help(tree.get_left_child(), infix_list)
        infix_list.append(tree.get_name())
        is_valid_bst_help(tree.get_right_child(), infix_list)

from Bio import Phylo
from cStringIO import StringIO
import uuid


def process_clade(clade, taxon_name_key, seen_names):
    """
    Creates a FullBiTree representation of a clade.

    Arguments:
    clade - a clade object as contructed by Phylo.read(...)
    seen_names - a set of string names of each node previously
             processed during construction.
    taxon_name_key - the property key that will be used to apply a taxon's name to a leaf's node
                     property when creating FullBiTree leafs.

    Returns:
    A FullBiTree representation of the given clade.
    """

    if (len(clade.clades) == 0) and (clade.name is None or len(clade.name) < 1):
        # Leaf nodes have to have a name (the taxon)
        raise Exception("Leaf node must have a name.")

    if clade.name is None or len(clade.name) < 1:
        # Give the internal node a unique name
        clade.name = str(uuid.uuid4())

    if len(clade.clades) != 0 and len(clade.clades) != 2:
        raise Exception("Each tree node must have zero or two children.")

    if clade.name in seen_names:
        raise Exception("Every node name in the tree must be unique. " +
                        clade.name + " is duplicated.")

    seen_names.add(clade.name)

    if len(clade.clades) == 0:
        # This is a leaf node
        tree = FullBiTree(clade.name)
        tree.set_node_property(taxon_name_key, clade.name)
        return tree
    else:
        # This is an internal node
        left  = process_clade(clade.clades[0], taxon_name_key, seen_names)
        right = process_clade(clade.clades[1], taxon_name_key, seen_names)
        tree = FullBiTree(clade.name, left, right)
        return tree


def parse_newick(newickstr, taxon_name_key):
    """
    Creates a FullBiTree representation of a newick string.

    Arguments:
    newickstr - a newick string
    taxon_name_key - the property key that will be used to apply a taxon's name to a leaf's node
                     property when creating FullBiTree leafs.

    Returns:
    A FullBiTree representation of the given newick string.
    """
    seen_names = set()
    tree = Phylo.read(StringIO(newickstr), "newick")
    return process_clade(tree.root, taxon_name_key, seen_names)

# print parse_newick('(A,B)', 'taxon')

def read_phylip(filename):
    """
    Read a file in Phylip format and return the length of the
    sequences and the taxa and sequences.

    Arguments:
    filename -- name of file in Phylip format

    Returns:
    A tuple where the first element is the length of the sequences and
    the second argument is a dictionary mapping taxa to sequences.
    """
    # Initialize return values in case file is bogus
    m = 0
    tsmap = {}

    with open(filename) as f:
        # First line contains n and m separated by a space
        nm = f.readline()
        nm = nm.split()
        n = int(nm[0])
        m = int(nm[1])

        # Subsequent lines contain taxon and sequence separated by a space
        for i in range(n):
            l = f.readline()
            l = l.split()
            tsmap[l[0]] = l[1]

    # Return sequence length and mapping of taxa to sequences
    return m, tsmap

print read_phylip('primate_seqs.phylip')

#####################  STUDENT CODE BELOW THIS LINE  #####################
# Test tree
A = FullBiTree('A')
B = FullBiTree('B')
C = FullBiTree('C')
D = FullBiTree('D')
E = FullBiTree('E', A, B)
F = FullBiTree('F', C, D)
G = FullBiTree('G', E, F)


def write_newick(t):
    """
    :param t: full binary Tree
    :return: newick sequence
    """
    if t.is_leaf():
        return t.get_name()

    else:
        return '(' + write_newick(t.get_left_child()) + ',' + write_newick(t.get_right_child()) + ')'

# print write_newick(G)


def compute_nni_neighborhood(t):
    nni = set()
    parent = dict()

    # set up child of root node
    parent[t.get_left_child().get_name()] = t
    parent[t.get_right_child().get_name()] = t
    stack = [t.get_left_child(), t.get_right_child()]

    while stack:
        node = stack.pop()

        if not node.is_leaf():
            # find children and set their parent
            left_child = node.get_left_child()
            right_child = node.get_right_child()
            parent[left_child.get_name()] = node
            parent[right_child.get_name()] = node

            # find other child of parent
            parent_node = parent[node.get_name()]

            if parent_node.get_left_child().get_name() != node.get_name():
                step = parent_node.get_left_child()
            else:
                step = parent_node.get_left_child()

            # create NNI by switching child1 and step
            t_copy = deepcopy(t)
            step_copy = deepcopy(step)
            left_copy = deepcopy(left_child)
            right_copy = deepcopy(right_child)

            # # find the children of step node if exists
            # if not step_copy.is_leaf():
            #     step_copy_child1 = step_copy.get_left_child()
            #     step_copy_child2 = step_copy.get_right_child()
            #     step_copy.remove_children()
            # else:
            #     step_copy_child1 = None
            #     step_copy_child2 = None
            #

            #
            # # find the children of the left child if it exists
            # if not left_copy.is_leaf():
            #     left_copy_child1 = left_copy.get_left_child()
            #     left_copy_child2 = left_copy.get_right_child()
            #     left_copy.remove_children()
            # else:
            #     left_copy_child1 = None
            #     left_copy_child2 = None
            #
            # # add the nodes to the
            # if left_copy_child1:
            #     step_copy.set_children(left_copy_child1, left_copy_child2)
            #
            # if step_copy_child1:
            #     left_copy.set_children(step_copy_child1, step_copy_child2)

            nni.add(tree_switch_node(t_copy, left_copy, step_copy))
            t_copy = deepcopy(t)
            nni.add(tree_switch_node(t_copy, right_copy, step_copy))

    return nni


def tree_switch_node(t, node1, node2):
    """
    :param t: tree
    :param node1: node to be switched
    :param node2: node to be switched
    :return: tree with the nodes switched
    """

    if t.get_name() == node1.get_name():
        return node2

    elif t.get_name() == node2.get_name():
        return node1

    elif t.is_leaf():
        return t

    else:
        left_child = t.get_left_child()
        right_child = t.get_right_child()
        t.remove_children()
        t.set_children(tree_switch_node(left_child, node1, node2), tree_switch_node(right_child, node1, node2))
        return t

# A3 = FullBiTree('A3')
# B3 = FullBiTree('B3')
# C3 = FullBiTree('C3')
# D3 = FullBiTree('D3', B3, C3)
# E3 = FullBiTree('E3', A3, D3)
# D3_switch = FullBiTree('D3')
# A3_switch = FullBiTree('A3', B3, C3)

# switched = tree_switch_node(E3, D3_switch, A3_switch)
# print switched
# nni_set = compute_nni_neighborhood(G)
# for nni in nni_set:
#     print nni

def random_tree(sequences):
    taxas = sequences.keys()
    node_num = len(taxas) - 1

    # initialize tree with one node
    start = taxas.pop(random.randrange(len(taxas)))
    # print start
    tree = FullBiTree(start)
    tree.set_node_property('taxon', start)
    tree.set_node_property('sequence', sequences[start])

    while taxas:
        prob = random.uniform(0, 1)

        # create leaf out of random taxa
        rand_taxa = taxas.pop(random.randrange(len(taxas)))
        tree_a = FullBiTree(rand_taxa)
        tree_a.set_node_property('taxon', rand_taxa)
        tree_a.set_node_property('sequence', sequences[rand_taxa])

        # add a pair of leaves to tree
        if prob >= 0.5 and len(taxas) >= 2:
            rand_taxa = taxas.pop(random.randrange(len(taxas)))
            tree_b = FullBiTree(rand_taxa)
            tree_b.set_node_property('taxon', rand_taxa)
            tree_b.set_node_property('sequence', sequences[rand_taxa])
            tree_ab = FullBiTree(node_num, tree_a, tree_b)
            node_num -= 1
            tree = FullBiTree(node_num, tree_ab, tree)
            node_num -= 1

        # add one leaf to tree
        else:
            tree = FullBiTree(node_num, tree_a, tree)
            node_num -= 1

    return tree

# primate = read_phylip('primate_seqs.phylip')[1]
# print random_tree(primate)


def bottom_up_ps(tree, sequence_key, set_dict):
    """
    :param tree: DNA tree
    :param sequence_key: key to dna sequence
    :param set_dict: dictionary of node name to list of dna sets
    :return: nothing, just modifies the set_dict
    """
    if tree.is_leaf():
        dna = tree.get_node_property(sequence_key)
        for idx in range(len(dna)):
            set_dict[tree.get_name()].append(set(dna[idx]))

    else:
        bottom_up_ps(tree.get_left_child(), sequence_key, set_dict)
        bottom_up_ps(tree.get_right_child(), sequence_key, set_dict)
        left_child_sets = set_dict[tree.get_left_child().get_name()]
        right_child_sets = set_dict[tree.get_right_child().get_name()]

        for idx in range(len(left_child_sets)):
            if left_child_sets[idx].intersection(right_child_sets[idx]):
                set_dict[tree.get_name()].append(left_child_sets[idx].intersection(right_child_sets[idx]))

            else:
                set_dict[tree.get_name()].append(left_child_sets[idx].union(right_child_sets[idx]))


def top_down_ps(tree, sequence_key, set_dict, m):
    """
    :param tree: DNA tree
    :param sequence_key: key to DNA sequence
    :param set_dict: dictionary of node name to list of dna sets
    :param m: length of DNA sequence
    :return: modifies tree by adding sequences to all internal nodes
    """
    stack = [tree]
    while stack:
        node = stack.pop()
        if not node.get_left_child().is_leaf():
            left_child = node.get_left_child()
            right_child = node.get_right_child()

            dna = node.get_node_property(sequence_key)

            # calculates the dna sequence of the left child
            left_dna = ''
            left_sets = set_dict[left_child.get_name()]
            for idx in range(m):
                if dna[idx] in left_sets[idx]:
                    left_dna += dna[idx]

                else:
                    random_dna = list(left_sets[idx]).pop(random.randrange(len(left_sets[idx]))) # pop random value
                    left_dna += random_dna

            left_child.set_node_property(sequence_key, left_dna)

            # calculates the dna sequence of the right child
            right_dna = ''
            right_sets = set_dict[right_child.get_name()]
            for idx in range(m):
                if dna[idx] in right_sets[idx]:
                    right_dna += dna[idx]

                else:
                    random_dna = list(right_sets[idx]).pop(random.randrange(len(right_sets[idx])))  # pop random value
                    right_dna += random_dna

            right_child.set_node_property(sequence_key, right_dna)

            node.set_children(left_child, right_child)
            stack.append(left_child)
            stack.append(right_child)


def tree_sequence_dict(tree, dna_dict, sequence_key):
    dna_dict[tree.get_name()] = tree.get_node_property(sequence_key)
    if not tree.is_leaf():
        tree_sequence_dict(tree.get_left_child(), dna_dict, sequence_key)
        tree_sequence_dict(tree.get_right_child(), dna_dict, sequence_key)


def compute_genetic_distance(genome_a, genome_b):
    """
    :param genome_a: sequence of genome for patient A
    :param genome_b: sequence of genome for patient B
    :return: the hamming distance
    """
    hamming_dist = 0

    # calculate number of different genomes
    for idx in range(len(genome_a)):
        if genome_a[idx] != genome_b[idx]:
            hamming_dist += 1

    return hamming_dist


def calculate_ps(dna_dict):
    ps_score = 0
    nodes = dna_dict.keys()

    while len(nodes) > 1:
        curr_node = nodes.pop(random.randrange(len(nodes)))

        for other_node in nodes:
            ps_score += compute_genetic_distance(dna_dict[curr_node], dna_dict[other_node])

    return ps_score


def compute_ps(tree, sequence_key, m):
    set_dict = defaultdict(list)

    # modify set_dict to contain information about internal nodes
    bottom_up_ps(tree, sequence_key, set_dict)

    # set dna sequence of root node
    dna = ''
    for dna_set in set_dict[tree.get_name()]:
        dna += random.choice(list(dna_set))

    tree.set_node_property(sequence_key, dna)

    # set dna sequence for internal nodes
    top_down_ps(tree, sequence_key, set_dict, m)

    # compute PS score
    dna_dict = dict()
    tree_sequence_dict(tree, dna_dict, sequence_key)  # modifies dna dict to have node name to dna sequence

    return calculate_ps(dna_dict)


# A = FullBiTree('A')
# A.set_node_property('sequence', 'GGTA')
# B = FullBiTree('B')
# B.set_node_property('sequence', 'ACGT')
# C = FullBiTree('C')
# C.set_node_property('sequence', 'ACTG')
# D = FullBiTree('D')
# D.set_node_property('sequence', 'CATA')
# E = FullBiTree('E', A, B)
# F = FullBiTree('F', C, D)
# G = FullBiTree('G', E, F)
# print compute_ps(G, 'sequence', 4)

primate = read_phylip('primate_seqs.phylip')[1]
len_m = len(primate[random.choice(primate.keys())])
print compute_ps(random_tree(primate), 'sequence', len_m)


def infer_evolutionary_tree(seqfile, outfile, numrestarts):
    sequences = read_phylip(seqfile)[1]
    tree = random_tree(sequences)
    m = len(sequences[random.choice(sequences.keys())])
    ps = compute_ps(tree, 'sequence', m)

    while numrestarts > 0:
        nni_set = compute_nni_neighborhood(tree)
        score_dict = dict()

        # finds ps scores of nni neighbors
        for nni in nni_set:
            score = compute_ps(nni, 'sequence', m)
            score_dict[score] = nni

        max_score = max(score_dict.keys())

        if max_score > ps:
            ps = max_score
            tree = score_dict[max_score]

        else:
            tree = random_tree(sequences)
            ps = compute_ps(tree, 'sequence', m)
            numrestarts -= 1

    return ps

# print infer_evolutionary_tree('primate_seqs.phylip', 'test', 5)