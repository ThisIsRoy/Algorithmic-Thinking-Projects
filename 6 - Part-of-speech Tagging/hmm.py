#################   DO NOT EDIT THESE IMPORTS #################
import math
import random
import numpy
from collections import *
from dikshanary import *

#################   PASTE PROVIDED CODE HERE AS NEEDED   #################
class HMM:
	"""
	Simple class to represent a Hidden Markov Model.
	"""

	def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
		self.order = order
		self.initial_distribution = initial_distribution
		self.emission_matrix = emission_matrix
		self.transition_matrix = transition_matrix


def read_pos_file(filename):
	"""
	Parses an input tagged text file.
	Input:
	filename --- the file to parse
	Returns:
	The file represented as a list of tuples, where each tuple
	is of the form (word, POS-tag).
	A list of unique words found in the file.
	A list of unique POS tags found in the file.
	"""
	file_representation = []
	unique_words = set()
	unique_tags = set()
	f = open(str(filename), "r")
	for line in f:
		if len(line) < 2 or len(line.split("/")) != 2:
			continue
		word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
		tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
		file_representation.append((word, tag))
		unique_words.add(word)
		unique_tags.add(tag)
	f.close()
	return file_representation, unique_words, unique_tags


#####################  STUDENT CODE BELOW THIS LINE  #####################
training_data, unique_words, unique_tags = read_pos_file('testdata_tagged.txt')
# print 'training data', training_data
# print 'words, ', words
# print 'tags, ', tags
# training_data, unique_words, unique_tags = read_pos_file('training.txt')

def compute_counts(training_data, order):
	"""
	:param training_data: a list of tuples of words and tags
	:param order: the order of the markov chain
	:return: the token number, word count, tag count, two tag count, three tag count
	"""
	data_len = len(training_data)
	tag_word_count = dikshanary()
	tag_count = dikshanary()
	two_tag_count = defaultdict(lambda: defaultdict(int))
	three_tag_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

	for idx, tup in enumerate(training_data):
		# Compute C(t, w)
		tag_word_count[tup] += 1

		# Compute C(t)
		tag_count[tup[1]] += 1

		# Compute (t_i-1, t_i)
		if idx < data_len - 1:
			two_tag_count[tup[1]][training_data[idx+1][1]] += 1
			# two_tag_count[(tup[1], training_data[idx + 1][1])] += 1

		# Compute (t_i-2, t_i-1, t_i)
		if order == 3 and idx < data_len - 2:
			three_tag_count[tup[1]][training_data[idx+1][1]][training_data[idx + 2][1]] += 1
			# three_tag_count[(tup[1], training_data[idx + 1][1], training_data[idx + 2][1])] += 1

	if order == 2:
		return tuple([len(training_data)]), tag_word_count, tag_count, two_tag_count

	elif order == 3:
		return tuple([len(training_data)]), tag_word_count, tag_count, two_tag_count, three_tag_count

	else:
		print 'OOPSIE WOOPSIE!! Uwu We made a fucky wucky!! A wittle fucko boingo! The code monkeys at our headquarters are working VEWY HAWD to fix this!'

# third order training data
# print compute_counts(training_data, 3)
# num_token, W, C1, C2, C3 = compute_counts(training_data, 3)
# print 'num tokens', num_token
# print 'w', W
# print 'C1', C1
# print 'C2', C2
# print 'C3', C3

# second order training data
num_token, W, C1, C2 = compute_counts(training_data, 2)


def compute_initial_distribution(training_data, order):
	"""
	:param training_data: a list of tuples of words and tags
	:param order: the order of the markov chain
	:return: the token number, word count, tag count, two tag count, three tag count
	"""
	data_len = len(training_data)
	start_num = 1.0

	# initialize dictionary of each word mapped to 0
	# pi_dict = dict([(word[1], 0) for word in training_data])

	# compute second order HMM
	if order == 2:
		pi_dict = dikshanary()
		pi_dict[training_data[0][1]] += 1

		for idx, tup in enumerate(training_data[1:-1]):

			if tup[0] == '.':
				pi_dict[training_data[idx + 2][1]] += 1
				start_num += 1

		# divide each count by total number of starting words
		for word in pi_dict:
			pi_dict[word] /= start_num

	# compute third order HMM
	elif order == 3:
		pi_dict = defaultdict(lambda: dikshanary())
		pi_dict[training_data[0][0]][training_data[1][0]] += 1

		for idx, tup in enumerate(training_data[2:]):
			if tup[0] == '.' and idx != data_len - 3:
				pi_dict[training_data[idx + 2][0]][training_data[idx + 3][0]] += 1
				start_num += 1

		# divide each count by total number of starting words
		for word, inner_dict in pi_dict.items():
			for word2 in inner_dict:
				pi_dict[word][word2] /= start_num

	return pi_dict

# initial = compute_initial_distribution(training_data, 2)
# print sum(initial.values()), 'should be equal to 1'
# for key, value in initial.items():
# 	if value != 0:
# 		print key, value


def compute_emission_probabilities(unique_words, unique_tags, W, C):
	emission = defaultdict(lambda: defaultdict(float))

	for tag in unique_tags:
		for word in unique_words:
			if W[(word, tag)] != 0:
				emission[tag][word] = float(W[(word, tag)]) / C[tag]

	return emission

# emission = compute_emission_probabilities(unique_words, unique_tags, W, C)
# for key, val in emission.items():
# 	for key2, val2 in val.items():
# 		if val2 > 0:
# 			print key2, val2


def compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order):
	lambdas = [0.0 , 0.0 , 0.0]
	num_tokens = num_tokens[0]

	# implement lambda calculations
	if order == 2:
		for bigram1 in C2:
			for bigram2, bigram_count in C2[bigram1].items():
				if bigram_count > 0:
					t_i = bigram2
					t_i_minus1 = bigram1

					alpha_0 = float(C1[t_i]) / num_tokens
					alpha_1 = float(C2[t_i_minus1][t_i]) / (C1[t_i_minus1] - 1) if C1[t_i_minus1] - 1 > 0 else 0

					alphas = [alpha_0, alpha_1]
					alpha_i = numpy.argmax(alphas)
					lambdas[alpha_i] += bigram_count

	elif order == 3:
		for trigram1 in C3:
			for trigram2 in C3[trigram1]:
				for trigram3, trigram_count in C3[trigram1][trigram2].items():

					if trigram_count > 0:
						t_i = trigram3
						t_i_minus1 = trigram2
						t_i_minus2 = trigram1

						# print 'c1', C1[t_i]
						alpha_0 = (float(C1[t_i]) - 1.0) / num_tokens if C1[t_i] - 1 != 0 else 0
						alpha_1 = (float(C2[t_i_minus1][t_i]) - 1.0) / (float(C1[t_i_minus1]) - 1.0) if C1[t_i_minus1] - 1 != 0 else 0
						# print 'c2', C2[(t_i_minus2, t_i_minus1)]
						alpha_2 = (trigram_count - 1.0) / (C2[t_i_minus2][t_i_minus1] - 1.0) if C2[t_i_minus2][t_i_minus1] - 1 != 0 else 0

						# find index of maximum alpha
						alphas = [alpha_0, alpha_1, alpha_2]
						alpha_i = numpy.argmax(alphas)
						lambdas[alpha_i] += trigram_count

	# print lambdas
	lambda_sum = sum(lambdas)
	return [lam / lambda_sum for lam in lambdas]

# print compute_lambdas(unique_tags, num_token, C1, C2, C3, 3)
# print compute_lambdas(None, num_token, C1, C2, C3, 3)

def build_hmm(training_data, unique_tags, unique_words, order, use_smoothing):
	if order == 2:
		num_token, W, C1, C2 = compute_counts(training_data, order)
	else:
		num_token, W, C1, C2, C3 = compute_counts(training_data, order)

	initial_dist = compute_initial_distribution(training_data, order)
	emission_matrix = compute_emission_probabilities(unique_words, unique_tags, W, C1)

	if use_smoothing:
		lambdas = compute_lambdas(unique_tags, num_token, C1, C2, C3, order)
	else:
		if order == 2:
			lambdas = [0, 1, 0]

		elif order == 3:
			lambdas = [0, 0, 1]


	if order == 2:
		# construct second order transition matrix
		transition_matrix = defaultdict(lambda: defaultdict(float))
		for bigram1 in C2:
			for bigram2 in C2[bigram1]:
				transition_matrix[bigram1][bigram2] = lambdas[1] * C2[bigram1][bigram2] / C1[bigram1] + lambdas[0] * C1[
					bigram2] / num_token[0]


	elif order == 3:
		# construct third order transition matrix
		transition_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
		for trigram1 in C3:
			for trigram2 in C3[trigram1]:
				for trigram3 in C3[trigram1][trigram2]:
					transition_matrix[trigram1][trigram2][trigram3] = lambdas[2] * C3[trigram1][trigram2][trigram3] / \
																	  C2[trigram1][trigram2] + lambdas[1] * \
																	  C2[trigram2][trigram3] / C1[trigram2] + lambdas[
																		  0] * C1[trigram3] / num_token[0]

	return HMM(order, initial_dist, emission_matrix, transition_matrix)

test_hmm = build_hmm(training_data, unique_tags, unique_words, 2, False)


def update_hmm_bigram(hmm, sentence):
	# missing_words = numpy.setdiff1d(	)
	hmm_words = set()
	for word_dict in hmm.emission_matrix.values():
		hmm_words.add(word_dict.keys())

	missing_words = numpy.setdiff1d(sentence, hmm_words)
	epsilon = 0.00001



def bigram_viterbi(hmm, sentence):
	v_matrix = defaultdict(list)
	bp_matrix = defaultdict(list)
	z = [0 for _ in range(len(sentence))]
	states = hmm.transition_matrix.keys()
	print 'words', hmm.emission_matrix.values()
	# print 'initial distribution', hmm.initial_distribution
	# print 'emission matix', hmm.emission_matrix
	# print 'states', states

	# set up first word in v matrix
	for state in states:
		# print hmm.initial_distribution[state], 'initial dist'
		# print state, 'state'
		# print sentence[0], 'word'
		# print hmm.emission_matrix[state][sentence[0]], 'emission matrix'
		v_matrix[state].append(hmm.initial_distribution[state] * hmm.emission_matrix[state][sentence[0]])
		bp_matrix[state].append(None)

	# use DP to calculate rest of words
	for word_idx in range(1, len(sentence)):
		for state in states:
			prev_col = [v_matrix[state2][word_idx - 1] * hmm.transition_matrix[state2][state] for state2 in states]
			v_matrix[state].append(hmm.emission_matrix[state][sentence[word_idx]] * max(prev_col))
			bp_matrix[state].append(states[numpy.argmax(prev_col)])

	# calculate last row of Z matrix to find state with maximum value
	v_max_key = [(key, v_matrix[key][-1]) for key in states]
	z[-1] = max(v_max_key, key = lambda item: item[1])[0]

	# construct rest of z array
	for index in reversed(range(len(z) - 1)):
		z[index] = bp_matrix[z[index + 1]][index + 1]

	return [(sentence[i], z[i]) for i in range(len(sentence))]

test_sentence = ['The', 'United', 'Nationals','Security','Council', 'was', 'a', 'series','of','domestic','programs','enacted','in','the','United','States','between','1933','and','1936',',','and','a','few','that','came','later','.']
# test_sentence = open('testdata_untagged.txt', 'r').read().split(' ')
# print test_sentence
print bigram_viterbi(test_hmm, test_sentence)


def trigram_viterbi(hmm, sentence):
	pass
