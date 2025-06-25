from langchain.vectorstores import Chroma

#Ce fichier contient des fonctions gerer une base de donnée vectorielle (embeddings) avec Chroma

#Creation de la base de donnée vectorielle Chroma
def init_vectorstore(docs, embedder, persist_dir: str, collection_name: str):
    store = Chroma.from_documents(
        documents=docs, #liste de livre(chunks de textes)
        embedding=embedder,#fonction chargé de calculer l'embedding de chaque chunks
        persist_directory=persist_dir, #chemin où chroma vas sotcker l'index et les vecteurs
        collection_name=collection_name
    )
    #sauvegarde la base de donnée vectorielle
    #pour qu'elle soit persistante et puisse être rechargée ultérieurement
    store.persist()
    return store


def load_vectorstore(embedder, persist_dir: str, collection_name: str):
    return Chroma(
        embedding_function=embedder,
        persist_directory=persist_dir,
        collection_name=collection_name
    )
