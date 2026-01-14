# ===============================
# 1. Imports
# ===============================
import os
import numpy as np
import pandas as pd

from tdc.multi_pred import DTI

from rdkit import Chem
from rdkit.Chem import MACCSkeys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, matthews_corrcoef

from bio_embeddings.embed import (
    ProtBertEmbedder,
    ProtTransT5Embedder,
    SeqVecEmbedder
)

# ===============================
# 2. Paths
# ===============================
DATA_DIR = "data"
EMB_DIR = "embeddings"
FEAT_DIR = "features"
RES_DIR = "results"

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ===============================
# 3. Load Davis dataset
# ===============================
print("Loading Davis dataset...")
data = DTI(name="Davis")
df = data.get_data()

print("Dataset size:", df.shape)
print("Columns:", df.columns)

y = df["Y"].values
np.save(os.path.join(RES_DIR, "y_true.npy"), y)

# ===============================
# 4. Ligand features (MACCS)
# ===============================
def compute_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167)
    return np.array(MACCSkeys.GenMACCSKeys(mol))

ligand_path = os.path.join(FEAT_DIR, "X_ligand_maccs.npy")

if os.path.exists(ligand_path):
    X_ligand = np.load(ligand_path)
else:
    print("Computing MACCS fingerprints...")
    X_ligand = np.vstack(df["Drug"].apply(compute_maccs))
    np.save(ligand_path, X_ligand)

print("Ligand shape:", X_ligand.shape)

# ===============================
# 5. Protein one-hot (baseline)
# ===============================
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def protein_onehot(seq):
    vec = np.zeros(len(AMINO_ACIDS))
    if isinstance(seq, str):
        for aa in seq:
            if aa in AMINO_ACIDS:
                vec[AMINO_ACIDS.index(aa)] += 1
    return vec / max(len(seq), 1)

onehot_path = os.path.join(FEAT_DIR, "X_protein_onehot.npy")

if os.path.exists(onehot_path):
    X_protein_onehot = np.load(onehot_path)
else:
    print("Computing protein one-hot features...")
    X_protein_onehot = np.vstack(df["Target"].apply(protein_onehot))
    np.save(onehot_path, X_protein_onehot)

# ===============================
# 6. Evaluation function
# ===============================
def evaluate(model, X, y, name):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)

    y_te_bin = (y_te <= 1000).astype(int)
    y_pr_bin = (y_pred <= 1000).astype(int)
    mcc = matthews_corrcoef(y_te_bin, y_pr_bin)

    print(f"\n=== {name} ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"MCC:  {mcc:.4f}")

    np.save(os.path.join(RES_DIR, f"y_pred_{name}.npy"), y_pred)

# ===============================
# 7. Models
# ===============================
rf = RandomForestRegressor(
    n_estimators=200,
    n_jobs=-1,
    random_state=42
)

mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128),
    max_iter=200,
    random_state=42
)

# ===============================
# 8. Baseline experiment
# ===============================
evaluate(
    rf,
    np.hstack([X_ligand, X_protein_onehot]),
    y,
    "onehot_rf"
)

# ===============================
# 9. Unique proteins
# ===============================
print("\nExtracting unique protein sequences...")
unique_proteins = df["Target"].unique()
print("Unique proteins:", len(unique_proteins))

# ===============================
# 10. ProtBERT (bio_embeddings)
# ===============================
protbert_path = os.path.join(EMB_DIR, "protein_protbert_dict.npy")

if os.path.exists(protbert_path):
    protein_protbert = np.load(protbert_path, allow_pickle=True).item()
else:
    print("Computing ProtBERT embeddings...")
    embedder = ProtBertEmbedder()
    protein_protbert = {}

    for i, seq in enumerate(unique_proteins):
        protein_protbert[seq] = embedder.embed(seq).mean(axis=0)
        print(f"ProtBERT {i+1}/{len(unique_proteins)}")

    np.save(protbert_path, protein_protbert)

X_protein_protbert = np.vstack(df["Target"].map(protein_protbert).values)

evaluate(
    rf,
    np.hstack([X_ligand, X_protein_protbert]),
    y,
    "protbert_rf"
)

# ===============================
# 11. ProtT5 (bio_embeddings)
# ===============================
prott5_path = os.path.join(EMB_DIR, "protein_prott5_dict.npy")

if os.path.exists(prott5_path):
    protein_prott5 = np.load(prott5_path, allow_pickle=True).item()
else:
    print("Computing ProtT5 embeddings...")
    embedder = ProtTransT5Embedder()
    protein_prott5 = {}

    for i, seq in enumerate(unique_proteins):
        protein_prott5[seq] = embedder.embed(seq).mean(axis=0)
        print(f"ProtT5 {i+1}/{len(unique_proteins)}")

    np.save(prott5_path, protein_prott5)

X_protein_prott5 = np.vstack(df["Target"].map(protein_prott5).values)

evaluate(
    rf,
    np.hstack([X_ligand, X_protein_prott5]),
    y,
    "prott5_rf"
)

# ===============================
# 12. SeqVec (bio_embeddings)
# ===============================
seqvec_path = os.path.join(EMB_DIR, "protein_seqvec_dict.npy")

if os.path.exists(seqvec_path):
    protein_seqvec = np.load(seqvec_path, allow_pickle=True).item()
else:
    print("Computing SeqVec embeddings...")
    embedder = SeqVecEmbedder()
    protein_seqvec = {}

    for i, seq in enumerate(unique_proteins):
        protein_seqvec[seq] = embedder.embed(seq).mean(axis=0)
        print(f"SeqVec {i+1}/{len(unique_proteins)}")

    np.save(seqvec_path, protein_seqvec)

X_protein_seqvec = np.vstack(df["Target"].map(protein_seqvec).values)

evaluate(
    rf,
    np.hstack([X_ligand, X_protein_seqvec]),
    y,
    "seqvec_rf"
)

print("\n✅ ALL GROUP 1 EXPERIMENTS COMPLETED SUCCESSFULLY.")
