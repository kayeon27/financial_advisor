from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings

def get_embedder(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
   
   #retourne un objet qui permet de transformer des textes(phrases, paragraphe) en vecteurs en capturant le sens
   return HuggingFaceEmbeddings(model_name=model_name)