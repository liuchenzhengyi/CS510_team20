import pickle
import networkx
import random
import json
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
import nltk
from bm25 import BM25
from OT import *
from utils import *


random.seed(10)


if __name__ == "__main__":
    start_time = time.time()

    with open("graph_GCN.pickle", "rb") as f:
        graph = pickle.load(f)

    n = 100
    querys = random.sample(list(graph.nodes), n)

    candidate_list = []
    for i in querys:
        candidate_list += list(graph.adj[i])
    candidate_list += random.sample(list(graph.nodes), 10000)
    id_list = list(set(candidate_list))
    # the embedding are pre-processed by sentenceBERT model
    query_emb = np.array([graph.nodes[i]["graph_embedding"] for i in querys])
    candidate_emb = np.array([graph.nodes[i]["graph_embedding"] for i in id_list])
    text_list = [graph.nodes[i]["text"] for i in id_list]

    text_list = [nltk.word_tokenize(sent) for sent in text_list]
    with open("word_corpus.json", "r") as f:
        word_corpus = json.load(f)
    bm25_model = BM25(text_list, word_corpus)

    # cost = np.exp(-np.dot(query_emb, candidate_emb.T))  # cost matrix
    cost = np.exp(-cosine_similarity(query_emb, candidate_emb))
    mu = [np.ones(len(querys))/len(querys), np.ones(len(id_list))/len(id_list)]
    epi = 1e-3
    alpha = 0.9
    ot_score = log_ot(cost, mu, epi, alpha, 1, 50)
    ot_rank_list = []

    lm_mrr = []
    lm_ndcg = []
    lm_rank_list = []
    bm_mrr = []
    bm_ndcg = []
    bm_rank_list = []
    ot_mrr = []
    ot_ndcg = []
    for i in range(len(querys)):
        if len(graph.adj[querys[i]]) < 3:
            continue

        q_embedding = graph.nodes[querys[i]]['graph_embedding']
        q_text = graph.nodes[querys[i]]['text']
        ground_truth = list(graph.adj[querys[i]])
        true_ranking = [id_list.index(j) for j in ground_truth]
        
        # ranking based on SBERT
        lm_ranking = cosine_similarity([q_embedding], candidate_emb)
        # lm_ranking = np.dot(q_embedding, candidate_emb.T)
        lm_ranking = lm_ranking.reshape(-1).argsort()[::-1]
        lm_rank_list.append(lm_ranking)
        
        # ranking based on bm25
        bm_ranking = bm25_model.ranked(nltk.word_tokenize(q_text), len(id_list))
        bm_ranking = np.array(bm_ranking)
        bm_rank_list.append(bm_ranking)

        # ranking based on OT
        ot_ranking = ot_score[i].argsort()[::-1]
        ot_rank_list.append(ot_ranking)

        # evaluation
        ylabel = np.zeros(len(id_list))
        ylabel[true_ranking] = 1

        lm_res = ylabel[lm_ranking]
        lm_mrr.append(mean_reciprocal_rank(lm_res))
        lm_ndcg.append(ndcg_at_k(lm_res, len(lm_res)))
        bm_res = ylabel[bm_ranking]
        bm_mrr.append(mean_reciprocal_rank(bm_res))
        bm_ndcg.append(ndcg_at_k(bm_res, len(bm_res)))
        ot_res = ylabel[ot_ranking]
        ot_mrr.append(mean_reciprocal_rank(ot_res))
        ot_ndcg.append(ndcg_at_k(ot_res, len(ot_res)))

        # # get the id of the ranked paper in the graph
        # ground_truth = list(graph.adj[i])
        # lm_ranking = [id_list[j] for j in lm_ranking]
        # bm_ranking = [id_list[j] for j in bm_ranking]
        # print(ground_truth)
        # print(lm_ranking[:10])
        # print(bm_ranking[:10])
        # print()

    relevance = np.dot(query_emb, candidate_emb.T)
    lm_fairness = fairness(lm_rank_list, relevance)
    bm_fairness = fairness(bm_rank_list, relevance)
    ot_fairness = fairness(ot_rank_list, relevance)
    print("=================SentenceBERT=================")
    print('Best Test Mean Reciprocal Rank(MRR):  %.4f' % np.average(lm_mrr))
    print('Normalized Discounted Cumulative Gain(NDCG): %.4f' % np.average(lm_ndcg))
    print('Fairness: %.4f' % lm_fairness)
    print()
    print("================BM25=================")
    print('Best Test Mean Reciprocal Rank(MRR):  %.4f' % np.average(bm_mrr))
    print('Normalized Discounted Cumulative Gain(NDCG): %.4f' % np.average(bm_ndcg))
    print('Fairness: %.4f' % bm_fairness)
    print()
    print("================OT=================")
    print('Best Test Mean Reciprocal Rank(MRR):  %.4f' % np.average(ot_mrr))
    print('Normalized Discounted Cumulative Gain(NDCG): %.4f' % np.average(ot_ndcg))
    print('Fairness: %.4f' % ot_fairness)

    end_time = time.time()
    print("Running time: %.2f" % (end_time - start_time) )