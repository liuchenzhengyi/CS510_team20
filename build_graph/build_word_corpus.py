from nltk import word_tokenize
import pandas as pd 
import json

if __name__ == "__main__":
    with open("node.csv", "r") as f:
        nodes = pd.read_csv(f)

    frequency = {}
    for _, row in nodes.iterrows():
        temp_node = list(row)
        text = temp_node[1]
        words = word_tokenize(text)
        for word in words:
            if len(word) > 1:
                if word not in frequency:
                    frequency[word] = 0
                frequency[word] += 1
    
    word_corpus = [k for k, _ in sorted(frequency.items(), key=lambda item: item[1])[-10000:]]

    with open("word_corpus.json", "w") as f:
        json.dump(word_corpus, f)
        
    