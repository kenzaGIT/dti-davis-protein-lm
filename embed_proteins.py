from bio_embeddings.embed import (
    SeqVecEmbedder,
    ProtTransBertBFDEmbedder,
    ProtTransT5XLU50Embedder
)
from bio_embeddings.utilities import read_fasta
import numpy as np
import os

# Paths
FASTA = "/work/data/proteins.fasta"
OUT = "/work/embeddings/bio_embeddings"
os.makedirs(OUT, exist_ok=True)

# Load sequences
sequences = read_fasta(FASTA)

def run_embedder(name, embedder):
    print(f"▶ Running {name}")
    X = []
    ids = []

    for seq_id, seq in sequences.items():
        emb = embedder.embed(seq)
        X.append(emb.mean(axis=0))  # per-protein embedding
        ids.append(seq_id)

    np.save(f"{OUT}/{name}.npy", np.array(X))
    np.save(f"{OUT}/{name}_ids.npy", np.array(ids))
    print(f"✅ {name} done")

# Embedders
run_embedder("seqvec", SeqVecEmbedder())
run_embedder("protbert", ProtTransBertBFDEmbedder())
run_embedder("prott5", ProtTransT5XLU50Embedder())
