from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://huggingface.co/datasets/bitext/"
    "Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)

ROOT = Path(__file__).resolve().parent.parent
DESTINATION = ROOT / "data" / "raw" / "customer_support.csv"

DESTINATION.parent.mkdir(parents=True, exist_ok=True)

if DESTINATION.exists():
    print(f"Dataset already exists: {DESTINATION}")
else:
    print("Downloading Bitext customer-support dataset...")
    urlretrieve(URL, DESTINATION)
    print(f"Saved to: {DESTINATION}")
