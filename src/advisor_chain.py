from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


#construction d'un système de question-réponse basé sur des documents financiers
def build_fin_advisor_chain(llm, retriever, template_str : str):
    #definir le prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"], 
        template = template_str)
    #créer une chaîne de question-réponse avec le modèle de langage, le prompt et le récupérateur
    return RetrievalQA.from_chain_type(
        llm=llm, #modele
        chain_type="stuff", #type de chaine de traitement ('fourrer')
        chain_type_kwargs={"prompt": prompt}, #passer le prompt
        retriever=retriever,# recherche et récupère les infos pertinentes dans les docs
        return_source_documents=True #retourne les documents sources en plus de la réponse
        )