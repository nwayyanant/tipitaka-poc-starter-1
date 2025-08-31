
# search_weaviate_labse_hybridfix.py
# Usage examples:
#   python search_weaviate_labse_hybridfix.py --url http://localhost:8080 --grpc-port 50051 --collection Window --mode vector --query "mettā and compassion" --k 5
#   python search_weaviate_labse_hybridfix.py --url http://localhost:8080 --grpc-port 50051 --collection Window --mode bm25   --query "Buddha" --k 5
#   python search_weaviate_labse_hybridfix.py --url http://localhost:8080 --grpc-port 50051 --collection Window --mode hybrid --query "four noble truths" --k 5 --alpha 0.5

import argparse
from typing import List
import numpy as np

from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/LaBSE"
_model = SentenceTransformer(MODEL_NAME)

MODEL_NAME = "sentence-transformers/LaBSE"

def get_client(url: str, grpc_port: int) -> WeaviateClient:
    cp = ConnectionParams.from_url(url, grpc_port=grpc_port)
    client = WeaviateClient(cp)
    client.connect()
    return client

def encode_query_labse(text: str) -> np.ndarray:
    vec = _model.encode([text], normalize_embeddings=False)  # keep consistent with stored vectors
    return vec.astype(np.float32)[0]

def pick_return_props(coll_name: str) -> List[str]:
    if coll_name == "Window":
        return ["window_id","text","chunk_id","path","h1","h2","h3","h4","h5","h6"]
    if coll_name == "Sentence":
        return ["sentence_id","sentence_text","chunk_id","path","h1","h2","h3","h4","h5","h6"]
    if coll_name == "Subchunk":
        return ["subchunk_id","subchunk_text","chunk_id"]
    if coll_name == "Chunk":
        return ["chunk_id","chunk_text"]
    return []

def short_text(s: str, n: int = 180) -> str:
    s = s or ""
    s = " ".join(s.split())  # collapse whitespace
    return s[:n] + ("..." if len(s) > n else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--grpc-port", type=int, default=50051)
    ap.add_argument("--collection", default="Window", choices=["Window","Sentence","Subchunk","Chunk"])
    ap.add_argument("--mode", default="vector", choices=["vector","bm25","hybrid"])
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5, help="hybrid alpha (0..1) higher favors vector")
    args = ap.parse_args()

    client = get_client(args.url, args.grpc_port)
    try:
        coll = client.collections.get(args.collection)
        props = pick_return_props(args.collection)

        if args.mode == "vector":
            qvec = encode_query_labse(args.query)
            res = coll.query.near_vector(
                near_vector=qvec,
                limit=args.k,
                return_properties=props
            )
        elif args.mode == "hybrid":
            qvec = encode_query_labse(args.query)
            res = coll.query.hybrid(
                query=args.query,   # BM25 side
                vector=qvec,        # vector side (since vectorizer=none)
                alpha=args.alpha,
                limit=args.k,
                return_properties=props
            )
        else:  # bm25
            res = coll.query.bm25(
                query=args.query,
                limit=args.k,
                return_properties=props
            )

        print(f"[results] {len(res.objects)} objects")
        for i, o in enumerate(res.objects or [], start=1):
            p = o.properties or {}
            kind = args.collection.lower()
            if args.collection == "Window":
                idx = f"[{kind}] {p.get('window_id','')} | chunk={p.get('chunk_id','')}"
                text = p.get("text","")
                trail = " > ".join(x for x in [p.get("h1"),p.get("h2"),p.get("h3"),p.get("h4"),p.get("h5"),p.get("h6")] if x)
                suffix = f" | Headings: {trail}" if trail else ""
            elif args.collection == "Sentence":
                idx = f"[{kind}] {p.get('sentence_id','')} | chunk={p.get('chunk_id','')}"
                text = p.get("sentence_text","")
                suffix = ""
            elif args.collection == "Subchunk":
                idx = f"[{kind}] {p.get('subchunk_id','')} | chunk={p.get('chunk_id','')}"
                text = p.get("subchunk_text","")
                suffix = ""
            else:
                idx = f"[{kind}] {p.get('chunk_id','')}"
                text = p.get("chunk_text","")
                suffix = ""

            print(f"{i:>2}. {idx}{suffix}")
            print(f"    {short_text(text)}")
    finally:
        client.close()

if __name__ == "__main__":
    main()
