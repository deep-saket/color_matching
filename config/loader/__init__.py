import os
import dotenv
from config.loader.ConfigLoader import ConfigLoader

dotenv.load_dotenv()

project_root = os.environ.get("PROJECT_ROOT")
settings = ConfigLoader(os.path.join(project_root, "config/files/settings.yml")).get_config()
prompts = ConfigLoader(os.path.join(project_root, "config/files/prompts.yml")).get_config()

artifacts_dir = os.path.join(project_root, settings["shared"]["artifacts_dir"])
os.makedirs(artifacts_dir, exist_ok=True)