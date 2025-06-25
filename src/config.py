import os
from dotenv import load_dotenv
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
PERSIST_DIR = os.getenv("PERSIST_DIR","./Data/Docs/chroma")
COLLECTION = os.getenv("COLLECTION","financial_advice")
RAW_CSV = os.getenv("RAW_CSV","./Data/Finance_data.csv")