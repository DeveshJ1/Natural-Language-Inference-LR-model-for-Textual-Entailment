import json
import collections
import argparse
import random
import numpy as np
from util import *

random.seed(42)

def extract_unigram_features(ex):
    """Return unigrams in the hypothesis and the premise.
    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of BoW featurs of x.
    Example:
        "I love it", "I hate it" --> {"I":2, "it":2, "hate":1, "love":1}
    """
    # BEGIN_YOUR_CODE
    words= ex['sentence1'] + ex['sentence2']
    return dict(collections.Counter(words)) 
   #question 8 code for testing 
    #hypothesis_words = ex['sentence2']  # Extract only hypothesis words
    #return dict(collections.Counter(hypothesis_words))

    # END_YOUR_CODE
#redo!!!!!!!!!!!!!!!!!!!!!!!!
def extract_custom_features(ex):
    """Design your own features.
    """
    # BEGIN_YOUR_CODE
    #dictionary of features to be returned
    features=collections.defaultdict(int)
    #distinguish premise and hypothesis words
    premise,hypothesis=ex['sentence1'], ex['sentence2']
    #establishing the unigrams of premise and hypothesis
    for word in premise:
        #we are prepending with premise in order to distinguish between premise and hypothesis unigrams 
        features[f'premise:{word}']+=1
    for word in hypothesis:
        #prepending with hypothesis for same reason 
        features[f'hypothesis:{word}']+=1
    #establishing the bigrams for premise and hypothesis
    for i in range(len(premise)-1):
        #again prepending with premise_bigram to distinguish premise and hypothesis bigrams 
        features[f'premise_bigram:{premise[i]}_{premise[i+1]}']+=1
    for i in range(len(hypothesis)-1):
        features[f'hypothesis_bigram{hypothesis[i]}_{hypothesis[i+1]}']+=1
    #finally distinguishing unigrams in both premise and bigram
    for word in set(hypothesis):
        features[f'both:{word}']= 1 if word in premise else 0
    return dict(features)
    # END_YOUR_CODE

def learn_predictor(train_data, valid_data, feature_extractor, learning_rate, num_epochs):
    """Running SGD on training examples using the logistic loss.
    You may want to evaluate the error on training and dev example after each epoch.
    Take a look at the functions predict and evaluate_predictor in util.py,
    which will be useful for your implementation.
    Parameters:
        train_data : [{gold_label: {0,1}, sentence1: [str], sentence2: [str]}]
        valid_data : same as train_data
        feature_extractor : function
            data (dict) --> feature vector (dict)
        learning_rate : float
        num_epochs : int
    Returns:
        weights : dict
            feature name (str) : weight (float)
    """
    # BEGIN_YOUR_CODE
    #establish dictionary of weights to be returned
    weights= collections.defaultdict(float)
    #loop through number of epochs to be considered
    for epoch in range(num_epochs):
        #shuffle the training data each time 
        random.shuffle(train_data)
        #for each example training data 
        for ex in train_data:
            #get the features according to argument given for method of feature extraction
            features=feature_extractor(ex)
            #obtain the label
            y=ex["gold_label"]
            #predict the label uses the predict function from utils.py 
            f=predict(weights,features)
            #get the gradient scale 
            scale=(f-y) * learning_rate
            #increment the weights according to the gradient using the increment function from utils.py
            increment(weights, features, -scale)  
    #return our dictionary of weights      
    return dict(weights) 
        
    # END_YOUR_CODE

def count_cooccur_matrix(tokens, window_size=4):
    """Compute the co-occurrence matrix given a sequence of tokens.
    For each word, n words before and n words after it are its co-occurring neighbors.
    For example, given the tokens "in for a penny , in for a pound",
    the neighbors of "penny" given a window size of 2 are "for", "a", ",", "in".
    Parameters:
        tokens : [str]
        window_size : int
    Returns:
        word2ind : dict
            word (str) : index (int)
        co_mat : np.array
            co_mat[i][j] should contain the co-occurrence counts of the words indexed by i and j according to the dictionary word2ind.
    """
    # BEGIN_YOUR_CODE
    #to avoid duplicates in the co_occurence matrix i first convert it to a set of tokens
    #then we sort this set and turn it to a list
    unique_tokens = list(sorted(set(tokens)))  #the reason for sorting this set is to negate randomness and enforce deterministic behvaior--> ensure reproducability of result 
    #create our dictionary to return later, going through set of unique tokens and establish an index for each unique word to be used in the co occurence matrix
    word2ind = {word: index for index, word in enumerate(unique_tokens)}
    #establish the size if words to initalize our matrix
    vocab_size = len(unique_tokens)
    #initalize our matrix 
    co_mat = np.zeros((vocab_size, vocab_size), dtype=int)
    
    # loop through all the original tokens  
    for i in range(len(tokens)):
        #get the word we are currently on which acts as the "center" word
        center_word = tokens[i]
        #get the index of that word for our co occurence matrix from our established dictionary
        center_index = word2ind[center_word]
        #establish the starting and ending point to consider using the window size argument--> pretty trivial math
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        #loop through start to end to go through each token
        for j in range(start, end):
            #if we are at the token we currently are using as the center word we skip this 
            if j == i:
                continue  
            #if not we consider it as one of the words within the window size 
            neighbor_word = tokens[j]
            neighbor_index = word2ind[neighbor_word]
            co_mat[center_index][neighbor_index] += 1
    return word2ind, co_mat
    # END_YOUR_CODE

def cooccur_to_embedding(co_mat, embed_size=50):
    """Convert the co-occurrence matrix to word embedding using truncated SVD. Use the np.linalg.svd function.
    Parameters:
        co_mat : np.array
            vocab size x vocab size
        embed_size : int
    Returns:
        embeddings : np.array
            vocab_size x embed_size
    """
    # BEGIN_YOUR_CODE
    #use the svd function as said in the homework directions
    #according to its documentation it returns 3 values U, S, and Vh where Vh doesn't really concern us
    U, S, Vh = np.linalg.svd(co_mat, full_matrices=False,hermitian=True)#set hermitian to true since co-occurence matrix should be symmetric
    # multiply by singular values
    embeddings = U[:, :embed_size] * S[:embed_size]  
    return embeddings    
# END_YOUR_CODE

def top_k_similar(word_ind, embeddings, word2ind, k=10, metric='dot'):
    """Return the top k most similar words to the given word (excluding itself).
    You will implement two similarity functions.
    If metric='dot', use the dot product.
    If metric='cosine', use the cosine similarity.
    Parameters:
        word_ind : int
            index of the word (for which we will find the similar words)
        embeddings : np.array
            vocab_size x embed_size
        word2ind : dict
        k : int
            number of words to return (excluding self)
        metric : 'dot' or 'cosine'
    Returns:
        topk-words : [str]
    """
    # BEGIN_YOUR_CODE
    #get the number of words
    vocab_size = embeddings.shape[0]
    #establish helper dictionary relating index and word 
    ind2word = {index: word for word, index in word2ind.items()}
    #helper variable storing the embedding of our target word
    target_embedding = embeddings[word_ind]
    #check the argument given for metric to choose similarity measure 
    #if dot product
    if metric == 'dot':
        #trivially compute dot producti between embedding of all words and for our target word embedding 
        scores = np.dot(embeddings, target_embedding)
    #if cosine similarity
    elif metric == 'cosine':
        #cosine formula is dot product of two vectors divided multiplication of their norm 
        #get the norm of the embedding of target word 
        norm_vec = np.linalg.norm(target_embedding)
        #get the norm of all embeddings 
        norms = np.linalg.norm(embeddings, axis=1)
        #calculate cosine similarity
        scores = np.dot(embeddings, target_embedding) / (norms * norm_vec + 1e-8)
    #metric is not supported 
    else:
        raise ValueError("Metric not supported")
    #similarity score of word itself should be a minimum since it should not be considered 
    scores[word_ind] = -np.inf  # Exclude self
    #get the top k indicies which had the highest similarity in descending order 
    topk_indices = np.argsort(scores)[-k:][::-1]  
    #return the topk words using the indicies retrieved and our established helper dictionary 
    return [ind2word[index] for index in topk_indices]    
# END_YOUR_CODE
