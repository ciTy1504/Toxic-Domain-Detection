import os
import re
import numpy as np
import tiktoken
import pickle
import json
from openai import OpenAI

client = OpenAI()
tokenizer = tiktoken.get_encoding("cl100k_base")
MAX_TOKEN = 8000

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

def clean_text(text):
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

def check_exist_emb(file_path, content):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        embedding = get_embedding(content)
        with open(file_path, "wb") as f:
            pickle.dump(embedding, f)
        return embedding

def prepare_emb(domain, base_path = "output"):
    domain_name = domain

    domain_dir = os.path.join(base_path, domain_name)
    os.makedirs(domain_dir, exist_ok=True)
    json_file_path = os.path.join(base_path, domain_name, f"{domain_name}.json")

    # Kiểm tra sự tồn tại của file JSON
    if not os.path.exists(json_file_path):
        print(f"Không tìm thấy file JSON: {json_file_path}")
        return

    # Mở và đọc file JSON
    with open(json_file_path, "r", encoding="utf-8-sig") as f:
        raw_json = json.load(f)

    # Lấy thông tin từ file JSON
    domain_val = raw_json.get("domain", "")
    headers = raw_json.get("headers", {})
    raw_text = raw_json.get("text", "")

    if isinstance(headers, dict):
        headers_text = "\n".join([f"{k}: {v}" for k, v in headers.items()])
    else:
        headers_text = str(headers)

    if isinstance(raw_text, list):
        raw_text = " ".join(raw_text)

    text = clean_text(raw_text)
    domain_headers = f"{domain}\n{headers_text}".strip()
    domain_content = f"{domain}\n{text}".strip()

    try:
        # Lưu embedding vào thư mục domain đã tạo
        domain_emb = check_exist_emb(os.path.join(domain_dir, "domain.pkl"), domain_content)
        headers_emb = check_exist_emb(os.path.join(domain_dir, "headers.pkl"), domain_headers)
        content_emb = check_exist_emb(os.path.join(domain_dir, "content.pkl"), text)
    except Exception as e:
        print(f"Lỗi khi embedding {domain_val}: {e}")
        return None

    try:
        domain_emb_np = np.array(domain_emb, dtype=np.float32)
        headers_emb_np = np.array(headers_emb, dtype=np.float32)
        content_emb_np = np.array(content_emb, dtype=np.float32)
    except Exception as e:
        print(f"Lỗi khi chuyển đổi embedding sang NumPy array cho {domain_val}: {e}")
        return None

    return {
        "raw_json": raw_json,
        "domain_emb": domain_emb_np,
        "headers_emb": headers_emb_np,
        "content_emb": content_emb_np
    }


if __name__ == "__main__":
    custom_output_path = "clean/multi"
    test_domain = "google.com"
    data = prepare_emb(test_domain, base_path=custom_output_path)
    if data: print(json.dumps(data, indent=4, ensure_ascii=False))
