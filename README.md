# Natural-Language-Inference-LR-model-for-Textual-Entailment
Goal: build a logistic regression model for textual entailment

Pipeline: Given a premise sentence, and a hypothesis sentence, we would like to predict whether the hypothesis is entailed by the premise, i.e. if the premise is true, then the hypothesis must be true.

Implementation Order(submission.py):

1-extract_unigram_features: takes in a premise and hypothesis as sentence x and returns a BoW feature vector for x. 

2-learn_predictor: Implements SGD on training data using logistic loss using the unigram feature extractor. Achieved error of 0.2442 on training set and 0.3559 on testing set. 

3-extract_custom_features: takes in premise and hypothesis as sentences. Forms features in multiple ways: individual words in the premise and hypothesis were used as individual features, words that are in both premise and hypothesis formed a feature, and bigrams of premise and hypothesis were used as features. Achieved error of 0.0757 on training set and 0.2728 on the test set. Outperformed unigram feature implementation

*Tried to change unigram feature implementation to only extract features from hypothesis. This increased the training error slightly to 0.2468 but test error decreased to 0.2838. So using custom features was still ideal. 

*The next focus was to implement functions to compute dense word vectors from a word co-occurrence matrix using SVD, and explore similarities between words. We estimate word vectors using the corpus Emma by Jane Austen from nltk. Take a look at the function read_corpus in util.py which downloads the corpus.

1-count_coocur_matrix: construct the word co-occurrence matrix using a window size of 4(considering 4 words before and after the center word). 

2-cooccur_to_embedding: Performs dimensionality reduction, specifically SVD, on the co-occurrence matrix to get the dense word vectors or embeddings. 

3-top_k_similar: a similarity function that finds the most similar words to a given word using either dot product or cosine similarity. Cosine similarity ended up outperforming dot product since dot product biases high frequency words so the results were stop words whereas cosine similarity improved on this result. 




