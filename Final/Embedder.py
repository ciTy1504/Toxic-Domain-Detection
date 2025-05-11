import json
import os
import pickle
import numpy as np
from Scraper import scraper
from Config_Loader import load_config
import re
import tiktoken
from openai import OpenAI

client = OpenAI()

def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Lỗi embed: {e}")
        return None

def clean_text(text, config):
    MAX_TOKEN = config["max_token"]
    tokenizer = tiktoken.get_encoding(config.get("tokenizer_encoding", "cl100k_base"))

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'[^\w\s.,:;!?@-]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    tokens = tokenizer.encode(text)
    if len(tokens) > MAX_TOKEN:
        first = tokens[:3000]
        mid = tokens[len(first) // 2 - 1000: len(first) // 2 + 1000]
        tail = tokens[-3000:]
        tokens = first + mid + tail
        text = tokenizer.decode(tokens)

    return text

#Nếu đã có file emb thì load ra, chưa có mới gọi API
def check_exist_emb(file_path, content):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        embedding = get_embedding(content)
        with open(file_path, "wb") as f:
            pickle.dump(embedding, f)
        return embedding

def prepare_emb(domain):
    config = load_config()
    flag, raw_json = scraper(domain, config)

    if flag in ("Err", "Ambiguous"):
        print(f"Bỏ qua {domain} vì flag = {flag}")
        return False, raw_json

    domain_val = raw_json.get("domain", "")
    headers = raw_json.get("headers", {})
    raw_text = raw_json.get("text", "")

    if isinstance(headers, dict):
        headers_text = "\n".join([f"{k}: {v}" for k, v in headers.items()])
    else:
        headers_text = str(headers)

    if isinstance(raw_text, list):
        raw_text = " ".join(raw_text)

    text = clean_text(raw_text, config)
    domain_headers = f"{domain}\n{headers_text}".strip()
    domain_content = f"{domain}\n{text}".strip()

    # Kiểm tra nếu text & content đều rỗng, và headers có chứa 'error', còn nếu chỉ 1 cái lỗi thì kệ
    if (not text.strip() and not domain_content.strip()) and ("error" in headers_text.lower()):
        print(f"Bỏ qua {domain_val} vì Lỗi cả 3")
        return False, raw_json

    output_base_path = config["output_base_path"]
    emb_path = os.path.join(output_base_path, domain)
    os.makedirs(emb_path, exist_ok=True)

    try:
        # Load hoặc tạo mới emb cho domain, headers, content
        domain_emb = check_exist_emb(os.path.join(emb_path, "domain.pkl"), domain)
        headers_emb = check_exist_emb(os.path.join(emb_path, "headers.pkl"), domain_headers)
        content_emb = check_exist_emb(os.path.join(emb_path, "content.pkl"), domain_content)
    except Exception as e:
        print(f"Lỗi khi embedding {domain_val}: {e}")
        return False, raw_json

    try:
        domain_emb_np = np.array(domain_emb, dtype=np.float32)
        headers_emb_np = np.array(headers_emb, dtype=np.float32)
        content_emb_np = np.array(content_emb, dtype=np.float32)
    except Exception as e:
        print(f"Lỗi khi chuyển đổi embedding sang NumPy array cho {domain_val}: {e}")
        return False, None

    # Một phát trả về cả 3 emb + 1 json
    return True,{
        "raw_json": raw_json,
        "domain_emb": domain_emb_np,
        "headers_emb": headers_emb_np,
        "content_emb": content_emb_np
    }

if __name__ == "__main__":
    test_domain = "google.com"
    data = prepare_emb(test_domain)
    print(json.dumps(data[1]["raw_json"], indent=4, ensure_ascii=False))
    print(data[1]["domain_emb"].shape)
