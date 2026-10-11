import os,google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

loader = TextLoader("data/policy.txt")
docs = loader.load()
policy_text = docs[0].page_content

question = "How many sick leaves do i get?"
prompt = f"Context:\n{policy_text}\n\nQuestion:{question}\nAnswer based ONLY on context:"

model = genai.GenerativeModel("models/gemini-3.8-flash")
response = model.generate_content(prompt)

print("QUESTION:", question)
print("ANSWER:", response.text)
