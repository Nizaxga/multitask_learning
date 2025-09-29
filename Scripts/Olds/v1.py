import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import mteb

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
Client = OpenAI(api_key=os.getenv("API_KEY"))
