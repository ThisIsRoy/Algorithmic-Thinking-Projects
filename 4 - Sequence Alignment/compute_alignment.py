import math
import random
from collections import deque
from collections import defaultdict

# Test data:
# M = {'A': {'A': 5, 'C': 2, 'G': 2, 'T': 2, '-': -2},
# 	 'C': {'A': 2, 'C': 5, 'G': 2, 'T': 2, '-': -2},
# 	 'G': {'A': 2, 'C': 2, 'G': 5, 'T': 2, '-': -2},
# 	 'T': {'A': 2, 'C': 2, 'G': 2, 'T': 5, '-': -2},
# 	 '-': {'A': -4, 'C': -4, 'G': -4, 'T': -4, '-': None}}
#
# X = 'AC'
# Y = 'TAG'
# X2 = 'AGT'
# Y2 = 'ACC'

def compute_global_alignment_scores(X, Y, M):
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

	return S

# Test cases:
# print compute_global_alignment_scores(X, Y, M)
# expect output to match given solution in pdf
# print compute_global_alignment_scores(X2, Y2, M)


def compute_alignment(X, Y, M, S):
	"""
	:param X: DNA sequence
	:param Y: DNA sequence
	:param M: scoring matrix
	:param S: optimal score matrix
	:return: optimal alignments X' and Y' that maximize score
	"""
	x_opt = ''
	y_opt = ''
	curr_row = len(X)
	curr_col = len(Y)

	while curr_row != 0 or curr_col != 0:
		curr_score = S[curr_row][curr_col]
		chose_path = False

		# check if path is moving up
		if curr_row > 0:
			vertical = S[curr_row - 1][curr_col]
			vert_score = vertical + M[X[curr_row - 1]]['-']

			if vert_score == curr_score:
				x_opt += X[curr_row - 1]
				y_opt += '-'
				curr_row -= 1
				chose_path = True

		# check if path is moving leftward
		if curr_col > 0 and not chose_path:
			horizontal = S[curr_row][curr_col - 1]
			horz_score = horizontal + M['-'][Y[curr_col - 1]]

			if horz_score == curr_score:
				x_opt += '-'
				y_opt += Y[curr_col -1]
				curr_col -= 1
				chose_path = True

		# check if path is diagonal
		if curr_row > 0 and curr_col > 0 and not chose_path:
			diagonal = S[curr_row - 1][curr_col - 1]
			diag_score = diagonal + M[X[curr_row - 1]][Y[curr_col - 1]]

			if diag_score == curr_score:
				x_opt += X[curr_row - 1]
				y_opt += Y[curr_col - 1]
				curr_row -= 1
				curr_col -= 1

	return x_opt[::-1], y_opt[::-1]

# Test cases:
# S = compute_global_alignment_scores(X, Y, M)
# print compute_alignment(X, Y, M, S)
# print compute_alignment(X2, Y2, M, S)


def global_alignment(X, Y, M):
	"""
	:param X: DNA sequence
	:param Y: DNA sequence
	:param M: scoring matrix
	:return: optimal sequences from X and Y
	"""
	S = compute_global_alignment_scores(X, Y, M)
 	return compute_alignment(X, Y, M, S)

# Test cases:
# print global_alignment(X, Y, M)
# print global_alignment(X2, Y2, M)


# Test data:
# M_local = {'A': {'A': 10, 'C': 2, 'G': 2, 'T': 2, '-': -4},
# 	 'C': {'A': 2, 'C': 10, 'G': 2, 'T': 2, '-': -4},
# 	 'G': {'A': 2, 'C': 2, 'G': 10, 'T': 2, '-': -4},
# 	 'T': {'A': 2, 'C': 2, 'G': 2, 'T': 10, '-': -4},
# 	 '-': {'A': -4, 'C': -4, 'G': -4, 'T': -4, '-': None}}
#
# X_local = 'ACC'
# Y_local = 'TTTACACGG'


def compute_local_alignment_scores(X, Y, M):
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
			horizontal  = S[row_idx][col_idx - 1] + M['-'][Y[col_idx - 1]]
			vertical = S[row_idx - 1][col_idx] + M[X[row_idx - 1]]['-']
			max_score = max([diagonal, horizontal, vertical])
			S[row_idx][col_idx] = max_score if max_score > 0 else 0

	return S

# Test cases:
# print compute_local_alignment_scores(X_local, Y_local, M_local)
# print compute_local_alignment_scores(Y_local, X_local, M_local)


def compute_local_alignment(X, Y, M, S):
	"""
	:param X: DNA sequence
	:param Y: DNA sequence
	:param M: scoring matrix
	:param S: optimal score matrix
	:return: optimal subsequences from X and Y
	"""
	max_score = -float('inf')
	x_opt = ''
	y_opt = ''

	# find maximum score from optimal score matrix
	for row in range(len(S)):
		for col in range(len(S[row])):
			if S[row][col] > max_score:
				max_score = S[row][col]
				curr_row = row
				curr_col = col

	while S[curr_row][curr_col] != 0:
		curr_score = S[curr_row][curr_col]
		chose_path = False

		# check if path is moving up
		if curr_row > 0 and not chose_path:
			vertical = S[curr_row - 1][curr_col]
			vert_score = vertical + M[X[curr_row - 1]]['-']

			if vert_score == curr_score:
				x_opt += X[curr_row - 1]
				y_opt += '-'
				curr_row -= 1
				chose_path = True

		# check if path is moving leftward
		if curr_col > 0 and not chose_path:
			horizontal = S[curr_row][curr_col - 1]
			horz_score = horizontal + M['-'][Y[curr_col - 1]]

			if horz_score == curr_score:
				x_opt += '-'
				y_opt += Y[curr_col - 1]
				curr_col -= 1
				chose_path = True

		# check if path is diagonal
		if curr_row > 0 and curr_col > 0 and not chose_path:
			diagonal = S[curr_row - 1][curr_col - 1]
			diag_score = diagonal + M[X[curr_row - 1]][Y[curr_col - 1]]

			if diag_score == curr_score:
				x_opt += X[curr_row - 1]
				y_opt += Y[curr_col - 1]
				curr_row -= 1
				curr_col -= 1
				chose_path = True

	return x_opt[::-1], y_opt[::-1]

# Test cases:
# S = compute_local_alignment_scores(X_local, Y_local, M_local)
# print compute_local_alignment(X_local, Y_local, M_local, S)
# print compute_local_alignment(Y_local, X_local, M_local, S)

def local_alignment(X, Y, M):
	"""
	:param X: DNA sequence
	:param Y: DNA sequence
	:param M: scoring matrix
	:return: optimal local alignments of the X and Y DNA
	"""
	S = compute_local_alignment_scores(X, Y, M)
	return compute_local_alignment(X, Y, M, S)

# Test cases:
# print local_alignment(X_local, Y_local, M_local)
# print local_alignment(Y_local, X_local, M_local)