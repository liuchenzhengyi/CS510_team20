import nltk
import numpy as np


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def mean_reciprocal_rank(r):
    r = np.asarray(r).nonzero()[0]
    return 1. / (r[0] + 1) if r.size else 0.


# used for sentence-level ranking
def split_sentences(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    sentences = [sent for sent in sentences if len(sent)>10]
    return sentences

# calculate fairness
def fairness(rank, relevance):
    rank = np.array(rank)
    numQ, numD = rank.shape
    # Document-side exposure
    exposure = np.zeros(numD)
    for i in range(numD):
        exposure[rank[:,i]] += 1/(i+1)
    exposure = exposure / np.sum(exposure)
    relevance = np.sum(relevance, axis = 0)
    relevance = relevance / np.sum(relevance)
    fairness = np.sqrt(np.sum(exposure**2 - relevance**2))
    return fairness
