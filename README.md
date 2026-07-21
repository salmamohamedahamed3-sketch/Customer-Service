# Simple RAG Lab — Customer Support

Same lab sequence as the reference lab, applied to a real dataset: the
[Bitext Customer Support LLM Chatbot Training Dataset](https://github.com/bitext/customer-support-llm-chatbot-training-dataset)
(26,872 instruction/response pairs across 27 intents). `01_documents.py` downloads the
dataset and builds one knowledge-base article per intent automatically — no manual data
entry.

```text
01_documents.py
02_preprocessing.py
03_chunking.py
04_vector_representation.py
05_create_chroma_store.py
06_retrieve_context.py
07_prompting.py
streamlit_app.py
```

Final retrieval:

```text
hybrid = 0.4 * BM25 + 0.6 * all-MiniLM-L6-v2 embeddings
```

Run manually:

```bash
python -m pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"
cp .env.example .env   # then fill in OPENROUTER_API_KEY
python 05_create_chroma_store.py
streamlit run streamlit_app.py
```
