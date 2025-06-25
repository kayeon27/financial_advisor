# src/llm_client.py
from pydantic import PrivateAttr
from langchain.chat_models.base import BaseChatModel
from huggingface_hub import InferenceClient
from langchain.schema import AIMessage, HumanMessage
from typing import List, Optional

class HFChatModel(BaseChatModel):
    # On déclare un attribut privé pour le client HuggingFace
    _client: InferenceClient = PrivateAttr()

    repo_id: str
    token: Optional[str]
    temperature: float = 0.1 

    def __init__(self, repo_id: str, token: Optional[str] = None, temperature: float = 0.1):
        # On appelle le constructeur de BaseModel / BaseChatModel
        super().__init__(repo_id=repo_id, token=token, temperature=temperature)
        # Puis on initialise notre client
        object.__setattr__(self, "_client", InferenceClient(token=token, provider='hf-inference'))

    def _generate(self, messages: List[HumanMessage], stop=None, **kwargs):
        # On combine tous les messages en un seul prompt
        prompt = self._format_messages_to_prompt(messages)
        
        # Appel à l'endpoint text-generation
        response = self._client.text_generation(
            prompt,
            model=self.repo_id,
            temperature=self.temperature,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            do_sample=True,
            return_full_text=False
        )
        
        return AIMessage(content=response)

    def _format_messages_to_prompt(self, messages: List[HumanMessage]) -> str:
        # On formate les messages en un prompt unique
        prompt = ""
        for i, message in enumerate(messages):
            if i == 0:
                # Premier message - on ajoute un contexte de système
                prompt += f"<|system|>\nTu es un conseiller financier expert. Réponds de manière claire et professionnelle. Réponds uniquement dans la langue de la question posée. \n<|user|>\n{message.content}\n<|assistant|>\n"
            else:
                prompt += f"{message.content}\n<|assistant|>\n"
        return prompt

    @property
    def _llm_type(self) -> str:
        return "hf_chat_model"
