from sentence_transformers import SentenceTransformer
import json
import pandas as pd


if __name__ == "__main__":
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    with open("node.csv", "r") as f:
        df = pd.read_csv(f)

    sentences = []
    for _, row in df.iterrows():
        sentences.append(list(row)[1])
    embeddings = model.encode(sentences)
    embeddings = [json.dumps(e.tolist()) for e in embeddings]

    df["embedding"] = embeddings
    df.to_csv("node_with_embedding.csv")