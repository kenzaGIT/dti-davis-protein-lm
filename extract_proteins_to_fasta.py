from tdc.multi_pred import DTI
import os

# ===============================
# Load Davis dataset
# ===============================
print("Loading Davis dataset...")
data = DTI(name="Davis")
df = data.get_data()

# ===============================
# Extract unique protein sequences
# ===============================
proteins = df["Target"].unique()
print(f"Unique proteins: {len(proteins)}")

# ===============================
# Save to FASTA
# ===============================
os.makedirs("data", exist_ok=True)
fasta_path = "data/proteins.fasta"

with open(fasta_path, "w") as f:
    for i, seq in enumerate(proteins):
        f.write(f">protein_{i}\n{seq}\n")

print(f"✅ FASTA file written to {fasta_path}")