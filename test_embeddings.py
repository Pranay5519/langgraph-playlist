from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# -------- Dense Embedding --------
emb = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cpu"}
)

hindi_text_doc = "इनसाइड जेएनएआई कैन बी रिप्रेजेंटेड हियर। इन आठ में से किसी एक लेयर में फिट कर सकते हो। अगर वो इन आठ में से किसी एक"
english_question = "How many layers are present in this architecture?"
hindi_question = "इस आर्किटेक्चर में कितनी लेयर मौजूद हैं?"

# ---- E5 prefix ----
doc = "passage: " + hindi_text_doc
queries = [
    "query: " + english_question,
    "query: " + hindi_question
]

doc_emb = emb.embed_documents([doc])[0]
query_embs = emb.embed_documents(queries)

dense_sim_eng = cosine_similarity([query_embs[0]], [doc_emb])[0][0]
dense_sim_hin = cosine_similarity([query_embs[1]], [doc_emb])[0][0]

# -------- BM25 --------
# simple whitespace tokenizer (works ok for Hindi)
tokenized_corpus = [hindi_text_doc.split()]
bm25 = BM25Okapi(tokenized_corpus)

tokenized_eng_query = english_question.split()
tokenized_hin_query = hindi_question.split()

bm25_eng = bm25.get_scores(tokenized_eng_query)[0]
bm25_hin = bm25.get_scores(tokenized_hin_query)[0]

print("===== Dense Similarity =====")
print("English ↔ Hindi Doc:", dense_sim_eng)
print("Hindi ↔ Hindi Doc:", dense_sim_hin)

print("\n===== BM25 Score =====")
print("English ↔ Hindi Doc:", bm25_eng)
print("Hindi ↔ Hindi Doc:", bm25_hin)




