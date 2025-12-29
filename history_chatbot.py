import os
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory



# LangSmith Configuration (via Environment Variables)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv(
    "LANGCHAIN_PROJECT", "Historical-Figures-Chatbot"
)



# PDF Ingestion as mentioned in req pdf

PDF_PATH = "historical_figures.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# Split PDF into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

docs = text_splitter.split_documents(documents)



# Vector Store (Chroma + Ollama Embeddings)
embeddings = OllamaEmbeddings(
    model="granite-embedding:latest"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="historical_figures"
)

retriever = vectorstore.as_retriever()



# LLM (Local Ollama)
llm = Ollama(
    model="llama3",
    temperature=0.2
)


# Custom Prompt Template (PDF-only answers)
prompt_template = """
You are HistoryBot, an expert on historical figures.
Answer the question strictly using the provided context.
If the answer is not present in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=False
)


# In-Memory Chat History
chat_history = ChatMessageHistory()

# Chat Functions 

def chat(user_input):
    if not user_input.strip():
        return "", []

    response = qa_chain.run(user_input)

    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(response)

    formatted_history = []
    for msg in chat_history.messages:
        if msg.type == "human":
            formatted_history.append(("User", msg.content))
        else:
            formatted_history.append(("HistoryBot", msg.content))

    return "", formatted_history


def clear_history():
    chat_history.clear()
    return "", []

 # Gradio UI
with gr.Blocks(title="Historical Figures Chatbot") as demo:
    gr.Image("history_banner.png",show_label=False)
    
    gr.Markdown(
        "### 🏛️ HistoryBot\n"
        "**Hello, I am HistoryBot, your expert on historical figures. "
        "How can I assist you today?**"
    )

    chatbot = gr.Chatbot(label="Conversation")
    user_input = gr.Textbox(
        label="Your Question",
        placeholder="Ask about a historical figure..."
    )

    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear History")

    submit_btn.click(
        fn=chat,
        inputs=user_input,
        outputs=[user_input, chatbot]
    )

    clear_btn.click(
        fn=clear_history,
        outputs=[user_input, chatbot]
    )

if __name__ == "__main__":
    demo.launch()
