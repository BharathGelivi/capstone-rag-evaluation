import os
import uvicorn
from dotenv import load_dotenv
from src.env_check import ensure_llm_credentials_or_exit
from configs.api import HOST, PORT

if __name__ == "__main__":
    load_dotenv()
    ensure_llm_credentials_or_exit()

    # Launch FastAPI application using Uvicorn
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    uvicorn.run("src.api:app", host=HOST, port=PORT, reload=debug)
