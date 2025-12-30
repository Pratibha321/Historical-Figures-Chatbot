# Historical-Figures-Chatbot

A local AI-powered chatbot that answers questions about historical figures using information strictly extracted from a provided PDF document. The application uses LangChain, Ollama (local LLM), Chroma vector database, and a Gradio-based UI.

🎯 Project Objective

Build a PDF-based Question Answering chatbot

Ensure answers are generated only from the uploaded document

Avoid hallucinations by returning “I don’t know based on the provided document” when data is missing

Run completely offline using a local LLM


▶️ How to Run the Project
1️⃣ Install Dependencies
pip install langchain chromadb gradio pypdf ollama

2️⃣ Install & Start Ollama
ollama pull llama3
ollama pull granite-embedding
ollama serve

3️⃣ Run the Application
python app.py

4️⃣ Open in Browser

http://127.0.0.1:7860
