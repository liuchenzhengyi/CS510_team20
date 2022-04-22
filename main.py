import pickle
import networkx
import random
import json
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
import nltk
from bm25 import BM25
from utils import *


random.seed(10)


if __name__ == "__main__":
    start_time = time.time()

    with open("graph.pickle", "rb") as f:
        graph = pickle.load(f)

    n = 100
    querys = random.sample(list(graph.nodes), n)

    candidate_list = []
    for i in querys:
        candidate_list += list(graph.adj[i])
    candidate_list += random.sample(list(graph.nodes), 10000)
    id_list = list(set(candidate_list))
    # the embedding are pre-processed by sentenceBERT model
    embedding_list = [json.loads(graph.nodes[i]["embedding"]) for i in id_list]
    text_list = [graph.nodes[i]["text"] for i in id_list]

    text_list = [nltk.word_tokenize(sent) for sent in text_list]
    with open("word_corpus.json", "r") as f:
        word_corpus = json.load(f)
    bm25_model = BM25(text_list, word_corpus)

    lm_mmr = []
    lm_ndcg = []
    bm_mmr = []
    bm_ndcg = []
    for i in querys:
        if len(graph.adj[i]) < 3:
            continue

        q_embedding = json.loads(graph.nodes[i]['embedding'])
        q_text = graph.nodes[i]['text']
        ground_truth = list(graph.adj[i])
        true_ranking = [id_list.index(j) for j in ground_truth]
        
        # ranking based on SBERT
        lm_ranking = cosine_similarity([q_embedding], embedding_list)
        lm_ranking = lm_ranking.reshape(-1).argsort()[::-1]
        
        # ranking based on bm25
        bm_ranking = bm25_model.ranked(nltk.word_tokenize(q_text), len(id_list))
        bm_ranking = np.array(bm_ranking)

        # evaluation
        ylabel = np.zeros(len(id_list))
        ylabel[true_ranking] = 1

        lm_res = ylabel[lm_ranking]
        lm_mmr.append(mean_reciprocal_rank(lm_res))
        lm_ndcg.append(ndcg_at_k(lm_res, len(lm_res)))
        bm_res = ylabel[bm_ranking]
        bm_mmr.append(mean_reciprocal_rank(bm_res))
        bm_ndcg.append(ndcg_at_k(bm_res, len(bm_res)))

        # # get the id of the ranked paper in the graph
        # ground_truth = list(graph.adj[i])
        # lm_ranking = [id_list[j] for j in lm_ranking]
        # bm_ranking = [id_list[j] for j in bm_ranking]
        # print(ground_truth)
        # print(lm_ranking[:10])
        # print(bm_ranking[:10])
        # print()

    print("=================SentenceBERT=================")
    print('Best Test Mean Reciprocal Rank(MRR):  %.4f' % np.average(lm_mmr))
    print('Normalized Discounted Cumulative Gain(NDCG): %.4f' % np.average(lm_ndcg))
    print()
    print("================BM25=================")
    print('Best Test Mean Reciprocal Rank(MRR):  %.4f' % np.average(bm_mmr))
    print('Normalized Discounted Cumulative Gain(NDCG): %.4f' % np.average(bm_ndcg))
    print()

    end_time = time.time()
    print("Running time: %.2f" % (end_time - start_time) )