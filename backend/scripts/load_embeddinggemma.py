from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m")

sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [4, 4]
