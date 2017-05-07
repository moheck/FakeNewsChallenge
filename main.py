from dataset import DataSet
import csv
import string
import cPickle as pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# read train_bodies.csv and write lemmatized_bodies.csv
def lemmatize_bodies():
	lemmatized_bodies = []
	bodies_file = csv.reader(open('csv/train_bodies.csv'))
	fields = bodies_file.next()
	for row in bodies_file:
		lemmas = to_lemmas(row[1])
		lemmatized_bodies.append({'Body ID': row[0], 'articleBody': lemmas})
	
	with open('csv/lemmatized_bodies.csv', 'wb') as output_file:
		dict_writer = csv.DictWriter(output_file, fields)
		dict_writer.writeheader()
		dict_writer.writerows(lemmatized_bodies)

# return a string of all lemmas from text
def to_lemmas(text):
	wordnet_lemmatizer = WordNetLemmatizer()
	t = remove_non_ascii(text)
	tokens = word_tokenize(t)
	return_string = ""

	for token in tokens:
		if(token in ENGLISH_STOP_WORDS):
			continue
		lemma = wordnet_lemmatizer.lemmatize(token)
		return_string += lemma + " "

	return return_string

# remove all non-ascii characters from txt
def remove_non_ascii(txt):
	return ''.join([i if ord(i) < 128 else ' ' for i in txt])

# return True if string x contains a refuting word
def has_negation(x):
	tok = remove_non_ascii(x)
	tokns = word_tokenize(tok)

	_refuting_words = [
		'fake',
		'fraud',
		'hoax',
		'false',
		'deny', 
		'denies',
		'refute',
		'not',
		'despite',
		'nope',
		'doubt', 
		'doubts',
		'bogus',
		'debunk',
		'pranks',
		'retract'
	]

	for tk in tokns:
		if(tk.lower() in _refuting_words):
			return True
	return False

# return True if headline and body negate eachother
def negation_feature(hl, bdy):
	hl_has_negation = has_negation(hl)
	bdy_has_negation = has_negation(bdy)
	if(hl_has_negation != bdy_has_negation):
		return 1
	else:
		return 0

# returns:
# similarity - tfidf score of headline to entire body
# sentences - dictionary containing every sentence, in its original form, that has > 0 tfidf similarity.
#				each sentence is a dict with keys for Sentence, tfidf Score, and Negates value
# max_head_body_sim - sentence with highest similarity to headline
def similarity_feature(head, bod, orig_bod):
	body_sentence_tokens = sent_tokenize(bod)
	orig_body_sentence_tokens = sent_tokenize(remove_non_ascii(orig_bod))
	h_lemmas = to_lemmas(head)
	vect = TfidfVectorizer(min_df=1)
	tfidf = vect.fit_transform([h_lemmas, bod])
	matrix = (tfidf * tfidf.T).A
	similarity = matrix[0][1]
	max_head_body_sim = {}
	max_score = 0.0
	negating_sentence_length = 0.0
	sum_of_negating_sentence_scores = 0.0

	sentences = []
	for (sentence, orig_sentence) in zip(body_sentence_tokens, orig_body_sentence_tokens):
		tf = vect.fit_transform([h_lemmas, sentence])
		mtx = (tf * tf.T).A
		sim = mtx[0][1]
		if(sim > 0.0):
			is_negation_of_headline = negation_feature(head, orig_sentence)
			if(is_negation_of_headline == 1):
				negating_sentence_length += 1
				sum_of_negating_sentence_scores += sim
			sentences.append({'Sentence': orig_sentence, 'Score': sim, 'Negates': is_negation_of_headline})

		if(sim > max_score):
			max_score = sim
			max_head_body_sim['Sentence'] = orig_sentence
			max_head_body_sim['Score'] = sim
			max_head_body_sim['Negates'] = is_negation_of_headline

	if(negating_sentence_length == 0):
		negation_average = 0
	else:
		negation_average = sum_of_negating_sentence_scores/negating_sentence_length

	return similarity, sentences, max_head_body_sim, negation_average

def save_object(obj, filename):
	with open(filename, 'wb') as output:
		pickle.dump(obj, output, -1)

# run features on the training set and implement SVM
def train():
	unrelated_vs_all = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
	disagree_vs_all = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
	agree_vs_all = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
	# create the training set with lemmatized bodies
	training_set = DataSet("csv/train_stances_csc483583.csv", "csv/lemmatized_bodies.csv")
	# create an original set that has original bodies
	orig_set = DataSet("csv/train_stances_csc483583.csv", "csv/train_bodies.csv")
	stances = training_set.stances
	articles = training_set.articles
	orig_articles = orig_set.articles

	similarity_vectors = []
	similarity_labels = []
	agree_labels = []

	negation_vectors = []
	negation_labels = []

	count = 0
	stanceVal = 0

	for stance in stances:
		count += 1
		print("Training article number: " + str(count))
		headline = stance['Headline']
		bodyID = stance['Body ID']
		#get lemmatized body from DataSet created with lemmatized_bodies.csv
		body_lemmas = articles[bodyID]
		#get the original body from DataSet created with train_bodies.csv
		orig_body = orig_articles[bodyID]
		stance = stance['Stance']
		#get the scores from the features
		similarity_score, similar_sentences, max_similarity, negation_avg = similarity_feature(headline, body_lemmas, orig_body)
		neg = max_similarity.get('Negates')
		if(neg == None):
			neg = 0
		
		max_score = max_similarity.get('Score')
		if(max_score == None):
			max_score = 0.0

		similarity_vectors.append([similarity_score, max_score])
		if(stance == 'unrelated'):
			similarity_labels.append(1)
		else:
			similarity_labels.append(2)

		if(stance == 'agree'):
			agree_labels.append(1)
		else:
			agree_labels.append(2)

		negation_vectors.append([negation_avg])
		if(stance == 'disagree'):
			negation_labels.append(1)
		else:
			negation_labels.append(2)

	np_sim_vectors = np.array(similarity_vectors)
	np_sim_labels = np.array(similarity_labels)
	unrelated_vs_all.fit(np_sim_vectors, np_sim_labels)
	save_object(unrelated_vs_all, 'unrelated_vs_all.pkl')

	np_neg_vectors = np.array(negation_vectors)
	np_neg_labels = np.array(negation_labels)
	disagree_vs_all.fit(np_neg_vectors, np_neg_labels)
	save_object(disagree_vs_all, 'disagree_vs_all.pkl')

	np_agree_labels = np.array(agree_labels)
	agree_vs_all.fit(np_sim_vectors, np_agree_labels)
	save_object(agree_vs_all, 'agree_vs_all.pkl')

# create a dataset with the test data and classify a stance for each article.
# then write the results to gold.csv
def test():
	with open('unrelated_vs_all.pkl', 'rb') as input:
		unrelated_vs_all = pickle.load(input)

	with open('disagree_vs_all.pkl', 'rb') as input:
		disagree_vs_all = pickle.load(input)

	with open('agree_vs_all.pkl', 'rb') as input:
		agree_vs_all = pickle.load(input)
	# create the test set with lemmatized bodies
	test_set = DataSet("csv/test_stances_csc483583.csv", "csv/lemmatized_bodies.csv")
	# create an original set that has original bodies
	orig_set = DataSet("csv/test_stances_csc483583.csv", "csv/train_bodies.csv")
	stances = test_set.stances
	articles = test_set.articles
	orig_articles = orig_set.articles
	gold = []
	count = 0

	for stance in stances:
		stance_result = ""
		headline = stance['Headline']
		bodyID = stance['Body ID']
		#get lemmatized body from DataSet created with lemmatized_bodies.csv
		body_lemmas = articles[bodyID]
		#get the original body from DataSet created with train_bodies.csv
		orig_body = orig_articles[bodyID]
		count += 1
		print("classifying article id: " + str(bodyID))
		print("article count: " + str(count))
		similarity_score, similar_sentences, max_similarity, negation_average = similarity_feature(headline, body_lemmas, orig_body)
		neg = max_similarity.get('Negates')
		if(neg == None):
			neg = 0
		max_score = max_similarity.get('Score')
		if(max_score == None):
			max_score = 0.0
		# predict stance_result using SVM
		unrelated_vs_all_result = unrelated_vs_all.predict([[similarity_score, max_score]])
		disagree_vs_all_result = disagree_vs_all.predict([[negation_average]])
		agree_vs_all_result = agree_vs_all.predict([[similarity_score, max_score]])
		if(unrelated_vs_all_result == 1):
			stance_result = 'unrelated'
		elif(disagree_vs_all_result == 1):
			stance_result = 'disagree'
		elif(agree_vs_all_result == 1):
			stance_result = 'agree'
		else:
			stance_result = 'discuss'

		gold.append({'Headline': headline, 'Body ID': bodyID, 'Stance': stance_result})

	keys = gold[0].keys()
	with open('csv/gold.csv', 'wb') as output_file:
		dict_writer = csv.DictWriter(output_file, keys)
		dict_writer.writeheader()
		dict_writer.writerows(gold)

if __name__ == "__main__":
	lemmatize_bodies()
	train()
	test()
