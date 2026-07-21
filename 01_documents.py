import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/bitext/"
    "customer-support-llm-chatbot-training-dataset/main/data/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)

# One canonical knowledge-base article per intent: the longest (most detailed)
# real agent response for that intent.
_raw = pd.read_csv(DATA_URL)
_longest_response_idx = _raw.groupby("intent")["response"].apply(
    lambda responses: responses.str.len().idxmax()
)
_canonical = _raw.loc[_longest_response_idx].sort_values("intent").reset_index(drop=True)

documents = [
    {
        "id": row["intent"],
        "title": row["intent"].replace("_", " ").title(),
        "category": row["category"],
        "is_current": True,
        "text": row["response"],
    }
    for _, row in _canonical.iterrows()
]
