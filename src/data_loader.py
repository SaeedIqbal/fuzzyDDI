import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_drugcombdb(path):
    df = pd.read_csv(path)
    print(f"Loaded DrugCombDB with {len(df)} records")
    return df

def load_drugbank(path):
    df = pd.read_csv(path)
    print(f"Loaded DrugBank with {len(df)} records")
    return df

def load_twosides(path):
    df = pd.read_csv(path)
    print(f"Loaded TWOSIDES with {len(df)} records")
    return df

def load_drkg_embeddings(path):
    df = pd.read_csv(path)
    entities = df['entity'].tolist()
    embeddings = df.drop(columns=['entity']).values
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    embedding_dict = {
        row['entity']: row.values[1:].astype(np.float32)
        for _, row in df.iterrows()
    }
    print(f"Loaded DRKG embeddings for {len(embedding_dict)} entities")
    return embedding_dict

# ---------------------------
# Example Usage
# ---------------------------

if __name__ == "__main__":
    base_path = "/home/phd/data/fuzzyDDI"

    # Load datasets
    drugcombdb_df = load_drugcombdb(os.path.join(base_path, "drugcombdb.csv"))
    drugbank_df = load_drugbank(os.path.join(base_path, "drugbank.csv"))
    twosides_df = load_twosides(os.path.join(base_path, "twosides.csv"))
    drkg_embeddings = load_drkg_embeddings(os.path.join(base_path, "drkg_embeddings.csv"))

    # Print sample data
    print("\nDrugCombDB Sample:")
    print(drugcombdb_df.head(2))

    print("\nDrugBank Sample:")
    print(drugbank_df[['drug1', 'drug2', 'label']].head(2))

    print("\nTWOSIDES Sample:")
    print(twosides_df[['drug1', 'drug2', 'side_effect_1', 'side_effect_2']].head(2))

    print("\nSample DRKG Embedding for DrugBank::DB00159:")
    print(drkg_embeddings.get("DrugBank::DB00159", None))