import numpy as np
from sentence_transformers import SentenceTransformer

def generate_and_save_embeddings():
    # 1. Load a set of sample queries
    sample_queries = [
        "What is the capital of France?",
        "How to bake a chocolate cake",
        "Latest advancements in quantum computing",
        "The weather is lovely today."
    ]

    # 2. Initialize the Sentence Transformer model
    # 'all-MiniLM-L6-v2' is a lightweight, general-purpose model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Compute embeddings
    print("Computing embeddings...")
    embeddings = model.encode(sample_queries)

    # 4. Save the embeddings to a .npy file
    output_filename = "embeddings.npy"
    np.save(output_filename, embeddings)
    print(f"Embeddings successfully saved to {output_filename}")

if __name__ == "__main__":
    generate_and_save_embeddings()