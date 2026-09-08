import os
import uvicorn
from dotenv import load_dotenv
from src.env_check import ensure_llm_credentials_or_exit
from configs.api import HOST, UI_PORT

if __name__ == "__main__":
    load_dotenv()
    ensure_llm_credentials_or_exit()

    debug = os.environ.get("DEBUG", "false").lower() == "true"
    uvicorn.run("src.api_ui:app", host=HOST, port=UI_PORT, reload=debug, workers=1)
