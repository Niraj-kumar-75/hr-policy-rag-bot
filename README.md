# HR-Policy RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that answers employee queries accurately based on company HR policy documents.

## 🚀 Features
* **Document Ingestion:** Loads and processes HR policy PDFs into vector embeddings.
* **Semantic Search:** Uses ChromaDB to retrieve the most relevant policy sections.
* **Context-Aware Answers:** Powered by Google Gemini API to provide grounded responses without hallucination.

## 🛠️ Tech Stack
* **Language:** Python
* **Vector Database:** ChromaDB
* **LLM:** Google Gemini API
* **Environment Management:** `python-dotenv`

## ⚙️ Local Setup & Run

1. **Clone the repository:**
   
   git clone https://github.com/Niraj-kumar-75/hr-policy-rag-bot.git
   cd hr-policy-rag-bot

   Install dependencies:
   pip install -r requirements.txt

   Configure environment variables:
   Create a .env file in the root directory (refer to .env.example) and add:
   GOOGLE_API_KEY=your_google_api_key_here

   Run the application:
   streamlit run app.py

