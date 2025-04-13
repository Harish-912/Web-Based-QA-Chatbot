from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import time
import sys
import os

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Chatbot_with_RAG_and_Webscraping"

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",  # or gemini-pro if you want
    google_api_key=google_api_key,
    temperature=0.3,
    top_k=4
)

embedding_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_api_key,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local("../vectorstores/dxfactor", embedding_model, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever()
retriever

# 🧠 Memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5,
)

# 🔧 Step 1: Create history-aware retriever
history_aware_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    history_aware_prompt
)

# 📝 Step 2: Your custom answer prompt
answer_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", """
You are a helpful and conversational assistant.
Use the following retrieved context to help answer the user's question, if it's relevant.
You may also use information from the chat history to personalize your response.
If the answer isn't in the context or history, respond naturally based on your general knowledge — but avoid making up facts.

Context:
{context}

Question: {input}
""")
])

# 🛠 Step 3: Build combine_docs_chain manually
combine_docs_chain = answer_prompt | llm | StrOutputParser()

# 🔄 Step 4: Final RAG chain
rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=combine_docs_chain
)

# 🔁 Ask with memory tracking
def ask_with_history(user_input):
    chat_history = memory.chat_memory.messages
    response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(response["answer"])
    return response["answer"]

# 🖥 Typing effect
def print_typing_effect(text, delay=0.01):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def show_history():
    print("\n--- Chat History ---")
    for msg in memory.chat_memory.messages:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}: {msg.content}")
    print("--- End of History ---\n")

def run_chatbot():
    print("Welcome to the RAG Chatbot with Memory!")
    print("Type 'HISTORY' to view chat history, 'CLEAR' to clear history, or 'STOP' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.upper() in ('STOP', 'EXIT', 'QUIT'):
            print("Exiting the chatbot. Goodbye!")
            break
        elif user_input.upper() == 'HISTORY':
            show_history()
            continue
        elif user_input.upper() == 'CLEAR':
            memory.clear()
            print("Chat history has been cleared.\n")
            continue

        start_time = time.time()
        answer = ask_with_history(user_input)
        end_time = time.time()

        print_typing_effect(f"AI: {answer}")
        print(f"(Response time: {end_time - start_time:.2f} seconds)\n")

if __name__ == "__main__":
    run_chatbot()