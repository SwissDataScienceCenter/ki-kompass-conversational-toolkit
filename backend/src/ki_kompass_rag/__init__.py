from pathlib import Path

from dotenv import load_dotenv

__version__ = "0.1.0"

load_dotenv(Path(__file__).parents[3] / ".env.local")
