import os
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import streamlit as st # type: ignore
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser

# === Load keys ===
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

# === Directories ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
vec_dir = os.path.join(script_dir, "vectorstores")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(vec_dir, exist_ok=True)

# === Scraper ===
def scrape_page(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f"Failed to fetch the page. Please try with different URL"

        soup = BeautifulSoup(response.content, "html.parser")
        content = []
        for tag in soup.find_all(["title", "h1", "h2", "h3", "h4", "h5", "h6", "p"]):
            text = tag.get_text(strip=True)
            if text and len(text) > 3:
                content.append(text)

        unique_text = list(dict.fromkeys(content))
        parsed_url = urlparse(url)
        url_path = parsed_url.path.strip("/").replace("/", "_") or "home"
        domain = parsed_url.netloc.replace(".", "_")
        filename = f"{domain}_{url_path}.txt"
        file_path = os.path.join(data_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(unique_text))

        return file_path, "Scraping complete."
    except Exception as e:
        return "", str(e)

# === Embed & Save ===
def embed_and_save(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embedding = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_api_key,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embedding)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(vec_dir, base_name)
    vectorstore.save_local(output_path)
    return output_path

# === Load chatbot ===
def get_chatbot(vector_path):
    embedding = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_api_key,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(vector_path, embedding, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=google_api_key,
        temperature=0.3,
        top_k=4
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10
    )

    history_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    answer_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", """
You are a helpful and conversational assistant.
Use the following retrieved context to help answer the user's question.
If it's not in context or chat history, rely on general knowledge.
Avoid fabricating facts.

Context:
{context}

Question: {input}
""")
    ])

    hist_retriever = create_history_aware_retriever(llm, retriever, history_prompt)
    combine_docs_chain = answer_prompt | llm | StrOutputParser()
    rag_chain = create_retrieval_chain(hist_retriever, combine_docs_chain)

    return rag_chain, memory

# === Streamlit UI ===
st.set_page_config(page_title="Web Q&A Chatbot", layout="centered")
st.title("🔍 Web Q&A Chatbot")
st.write(script_dir, data_dir, vec_dir)

url = st.text_input("Enter a URL to chat about:")
submit = st.button("Process")

if submit and url:
    if not url.startswith("http"):
        st.error("Invalid URL. Please include http/https.")
    else:
        with st.spinner("Scraping webpage..."):
            file_path, message = scrape_page(url)
        if file_path:
            st.success(message)
            with st.spinner("Embedding and saving..."):
                vector_path = embed_and_save(file_path)
            st.session_state["vector_path"] = vector_path
            st.success("Ready to chat!")
        else:
            st.error(message)

if "vector_path" in st.session_state:
    rag_chain, memory = get_chatbot(st.session_state["vector_path"])
    st.subheader("Ask questions about the page:")
    user_input = st.text_input("You:")
    if user_input:
        with st.spinner("Thinking..."):
            chat_history = memory.chat_memory.messages
            result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(result["answer"])
            st.markdown(f"**🤖:** {result['answer']}")
