from vectordb import build_vector_db, model
import numpy as np
from pathlib import Path

directory = Path.home() / "path-to-code"
index, metadata = build_vector_db(directory)


# Search example
query = "the read power point file isn't working"
query_emb = model.encode(query)
D, I = index.search(np.array([query_emb], dtype=np.float32), k=3)

for idx in I[0]:
    print(metadata[idx]['file_name'], metadata[idx]['code'])
    print("-----")