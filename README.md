# 🔍 Web Q&A Chatbot

A web-based chatbot that scrapes content from any URL, converts it into embeddings using Hugging Face and FAISS, and answers user queries using a history-aware RAG pipeline powered by Google's Gemini LLM. Deployed on AWS EC2 using Docker.

---

## 📸 Demo

![Screenshot](Screenshots/Screenshot%202025-04-14%20003413.png)

---

## 🚀 Features

- 🌐 Scrape content from any public webpage
- 📚 Embeds and stores data with FAISS
- 🧠 Conversational LLM with memory and retrieval (RAG)
- 🔎 Powered by Gemini-Flash and HuggingFace embeddings
- 🐳 Deployed on AWS EC2 using Docker

---

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **Backend**: LangChain, Gemini LLM
- **Embeddings**: Hugging Face Inference API (`all-MiniLM-L6-v2`)
- **Vector Store**: FAISS
- **Web Scraping**: BeautifulSoup
- **Deployment**: AWS EC2, Docker

---

## 🛠️ Setup & Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/web-qa-chatbot.git
cd web-qa-chatbot
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Create a .env file in the root directory and add your keys:
```bash
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACE_API_KEY=your_hf_api_key
```
4. Run the app
```bash
streamlit run app.py
```

---

## 🌐 How to Use

- Enter a valid URL in the input box.
- The app scrapes the content and stores embeddings.
- You can now chat with the bot and ask questions about the page content!

---

## 🧪 Limitations / Future Improvements

- Currently supports only basic HTML scraping (JavaScript content not handled)
- Could add PDF/YouTube/Multiple URLs support
- Could integrate OpenAI/Gemini Pro for better accuracy

