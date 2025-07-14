# app/vectorizer_test.py

import requests
import numpy as np


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

payload = {
    "texts": [
        # Base Query 1
        "Nebraska precipitation-data 2010-2020 10mx0m resolution",
        "Nebraska rainfall records from 2010 to 2020 at 10m spatial resolution",  # semantic match
        "Nebraska elevation data 2010-2020 10m resolution",  #  lexically similar but unrelated (elevation vs precipitation)
        "Nebraska precipitation trends 1900-2000 10m resolution",  #  semantically similar but different time range
        
        # Base Query 2
        "Nebraska precipitation 10m res",
        "Precipitation dataset for Nebraska with 10m resolution",  # semantic match
        "Nebraska temperature 10m resolution",  # same region/resolution, wrong data type
        "Kansas precipitation 10m res",  # similar variable, wrong location
        
        # Base Query 3
        "Nebraska pelevation tiff 20m resolution",
        "Nebraska elevation raster in TIFF format at 20m resolution",  # semantic match (pelevation typo corrected)
        "Nebraska land cover tiff 20m resolution",  # same file type/resolution but wrong variable
        "South Dakota elevation tiff 20m resolution",  # same data type, wrong state
        
        # Base Query 4
        "Iowa CAFO-data 2010-2020 10m resolution",
        "Iowa Confined Animal Feeding Operations (CAFO) spatial data 2010-2020 10m resolution",  # semantic match
        "Iowa CAFO-data 1990-2000 10m resolution",  # similar but different time period
        "Nebraska CAFO-data 2010-2020 10m resolution",  # same data type, wrong state
    ]
}


resp = requests.post(" http://0.0.0.0:8000/embed", json=payload)
resp = resp.json()
resp = resp["embeddings"]
search_queries_embs = resp[::4]
search_queries_text = payload['texts'][::4]

for query, q_idx in search_queries_embs:
	top_index = -1
	top_score = -1
	for emb, idx in resp:
		if emb == query: continue
		similarity = cosine_similarity(query, emb)
		if similarity > top_score:
			top_score = similarity
			top_index = idx
	print(f"Top similarity with search rating: {search_queries_text[q_idx]}: {payload['texts'][top_index]}" )
	