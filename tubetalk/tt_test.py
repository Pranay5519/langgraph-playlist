
from retriever import create_retriever_from_url
retriever_result = create_retriever_from_url("https://www.youtube.com/watch?v=e-GR3PlEOVU", doc_language="hindi")

if isinstance(retriever_result, tuple):
    retriever_obj = retriever_result[0]
else:
    retriever_obj = retriever_result

docs = retriever_obj("How many layers are present in this Architecture?") 

print(docs)
