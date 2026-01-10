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

import torch
from transformers import AutoTokenizer, AutoModel
import esm

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
# 10. ProtBERT embeddings (CPU safe)
# ===============================
protbert_path = os.path.join(EMB_DIR, "protein_protbert_dict.npy")

if os.path.exists(protbert_path):
    protein_bert = np.load(protbert_path, allow_pickle=True).item()
else:
    print("Computing ProtBERT embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Rostlab/prot_bert", do_lower_case=False
    )
    model = AutoModel.from_pretrained("Rostlab/prot_bert")
    model.eval()

    protein_bert = {}

    for i, seq in enumerate(unique_proteins):
        seq_spaced = " ".join(list(seq))
        inputs = tokenizer(seq_spaced, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model(**inputs)
        protein_bert[seq] = out.last_hidden_state.mean(dim=1).squeeze().numpy()
        print(f"ProtBERT {i+1}/{len(unique_proteins)}")

    np.save(protbert_path, protein_bert)

X_protein_bert = np.vstack(df["Target"].map(protein_bert).values)

evaluate(
    rf,
    np.hstack([X_ligand, X_protein_bert]),
    y,
    "protbert_rf"
)

# ===============================
# 11. ESM-2 embeddings (CPU safe)
# ===============================
esm_path = os.path.join(EMB_DIR, "protein_esm2_dict.npy")

if os.path.exists(esm_path):
    protein_esm = np.load(esm_path, allow_pickle=True).item()
else:
    print("Computing ESM-2 embeddings...")
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()

    protein_esm = {}

    for i, seq in enumerate(unique_proteins):
        data = [("p", seq)]
        _, _, tokens = batch_converter(data)
        with torch.no_grad():
            res = esm_model(tokens, repr_layers=[6])
        emb = res["representations"][6][0, 1:len(seq)+1].mean(0).numpy()
        protein_esm[seq] = emb
        print(f"ESM-2 {i+1}/{len(unique_proteins)}")

    np.save(esm_path, protein_esm)

X_protein_esm = np.vstack(df["Target"].map(protein_esm).values)

evaluate(
    rf,
    np.hstack([X_ligand, X_protein_esm]),
    y,
    "esm2_rf"
)

evaluate(
    mlp,
    np.hstack([X_ligand, X_protein_esm]),
    y,
    "esm2_mlp"
)

print("\n✅ ALL GROUP 1 EXPERIMENTS COMPLETED SUCCESSFULLY.")
