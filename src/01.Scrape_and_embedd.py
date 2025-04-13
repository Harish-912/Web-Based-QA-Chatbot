import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
import time
import os

def Scrape_and_clean(url_to_scrape):
    wait_time = 2.5  # seconds

    # === Create output filename based on URL ===
    parsed_url = urlparse(url_to_scrape)
    url_path = parsed_url.path.strip("/").replace("/", "_") or "home"
    domain = parsed_url.netloc.replace(".", "_")
    filename = f"{domain}_{url_path}.txt"

    # === Save location ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, filename)

    try:
        # === Get HTML content ===
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        response = requests.get(url_to_scrape, headers=headers, timeout=10)
        time.sleep(wait_time)

        if response.status_code != 200:
            print(f"Failed to fetch page. Status code: {response.status_code}")
            return None

        soup = BeautifulSoup(response.content, "html.parser")

        # === Extract visible text content ===
        content = []
        for tag in soup.find_all(["title", "h1", "h2", "h3", "h4", "h5", "h6", "p"]):
            text = tag.get_text(strip=True)
            if text and len(text) > 3:
                content.append(text)

        # === Remove duplicates ===
        seen = set()
        unique_content = []
        for line in content:
            if line not in seen:
                seen.add(line)
                unique_content.append(line)

        # === Save to file ===
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"URL: {url_to_scrape}\n\n")
            f.write("\n".join(unique_content))

        print(f"\n✅ Scraping complete. Saved to: {output_file}")
        return output_file  # Return the full path

    except Exception as e:
        print(f"There was an error during scraping: {e}")
        return None


def embed_url_content(file_path):
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_api_key:
        raise ValueError("❌ Hugging Face API key not found in environment variables.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found at {file_path}")

    # Read file
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)
    print(f"📄 Total chunks created: {len(chunks)}")

    # Load embedding model
    embedding_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_api_key,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Embed and create vector store
    vectorstore = FAISS.from_texts(chunks, embedding_model)

    # Save vector store
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.abspath(os.path.join(script_dir, "..", "vectorstores"))
    os.makedirs(save_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(save_dir, base_name)
    vectorstore.save_local(output_file)

    print(f"✅ Embedding complete and vector store saved to: {output_file}")

# === RUN BOTH ===
url_to_scrape = input("Enter the URL to scrape and embed: ").strip()
scraped_file_path = Scrape_and_clean(url_to_scrape)

if scraped_file_path:
    embed_url_content(scraped_file_path)
