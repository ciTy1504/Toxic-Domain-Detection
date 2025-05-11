import os
import pandas as pd
import math
import numpy as np
from collections import Counter
from sklearn.compose import ColumnTransformer
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle
import torch.optim as optim
import collections
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from Embedder import prepare_emb
import torch
import torch.nn as nn

black_words = [
    "porn", "xxx", "boobs", "nude", "sex", "adult", "cam", "webcam", "live", "milf", "teen", "anal", "bdsm",
    "hentai", "lesbian", "gay", "fetish", "creampie", "hardcore", "pussy", "dildo", "strip", "escort", "pornhub",
    "xvideos", "redtube", "bet", "casino", "win", "lotto", "jackpot", "poker", "slots", "betting", "stake", "crypto",
    "bitcoin", "blockchain", "ethereum", "forex", "trading", "binary", "invest", "money", "rich", "profit", "bonus",
    "update", "verify", "security", "login", "account", "support", "download", "install", "alert", "warning", "system",
    "firewall", "malware", "protection", "secure", "bank", "paypal", "unlock", "recovery", "free", "gift", "prize",
    "winner", "offer", "deal", "freegift", "promo", "discount", "coupon", "sale", "cheap", "limited", "exclusive",
    "now", "urgent", "winbig", "instant", "redeem", "xyz", "top", "vip", "pro", "online", "info", "click", "store", "fun",
    "icu", "link", "website", "page", "trade", "shop", "love", "party", "wtf", "sexy", "host", "tech", "cloud", "media",
    "tube", "stream", "video", "app", "mobi", "site", "biz", "date", "review", "space", "tk", "ga", "cf", "gq", "hotsex",
    "adultfriend", "camgirl", "livechat", "erotic", "horny", "one-night", "sexchat", "hooker", "meetsex",
    "naked", "pornstars", "spicy", "singles", "fap", "flirt", "nudegirls", "nudes", "stripchat", "camshow",
    "betonline", "bet365", "fastcash", "casinogames", "slotmachine", "spinwin", "cryptobets", "gambling", "earncash", "freemoney",
    "virus", "hacker", "ransom", "infected", "trojan", "identitycheck", "yourbank", "resetpassword", "emailverify",
    "lockedaccount", "vpnsecure", "devicealert", "confirmemail", "verifylogin", "phishalert", "secureportal", "resetlink", "documentupdate", "databreach",
    "getfree", "only1day", "limiteddeal", "hurryup", "massiveoffer", "grabnow", "freesample", "promocode", "bigdiscount",
    "flashsale", "bigdeal", "instantgift", "winfree", "joinfree", "justforyou", "hurrydeal", "clickwin", "0dollar", "winiphone",
    "webpage", "onlineshop", "lowprice", "supercheap", "instantsale", "grabdeal", "fastservice", "clickbuy", "digitalpromo",
    "autobuy", "smartbuy", "freehosting", "videoplayer", "cheapdeal", "appzone", "cloudapp", "mediahub", "shopzone",
    "webcamera", "adultdating", "webgirls", "findsex", "nightcams", "naughty", "findlover", "datingmatch", "spicydate",
    "bigtits", "hotdating", "sexvideos",
    "người lớn", "phim sex", "ảnh sex", "clip sex", "địt", "lồn", "bướm", "cu", "gái", "gái gọi", "trai", "tìm bạn", "tình dục",
    "hàng nóng", "lộ hàng", "kheo hàng", "quay lén", "địt lén", "thủ dâm", "đồng tính", "les", "livestream",
    "chat sex", "gọi tình", "nhạy cẩm", "dâm đãng", "cờ bạc", "cá cược", "đánh bạc", "đánh bài", "kiếm tiền",
    "sồng bạc", "xổ số", "lô đề", "vietlott", "số cầu", "dự đoán", "tài xỉu", "bầu cua", "xóc đĩa", "game bài", "bài tiến lên",
    "bóng đá", "tỷ lệ bóng đá", "kèo nhà cái", "nhà cái", "cầu đỏ", "trúng thưởng", "quay hũ", "nổ hũ", "đổi thưởng", "thẻ cào",
    "tài chính", "đầu tư", "làm giàu", "tiền ảo", "chứng khoán", "cổ phiếu", "ngoại hối",
    "vay tiền", "vay nhanh", "vay online", "app vay", "vay không thế chấp", "tín dụng", "tín dụng đen",
    "lợi nhuận", "lãi suất", "thu nhập", "việc làm", "việc làm online", "tuyển dụng", "bán thời gian",
    "bảo mật", "an ninh", "an toàn", "xác minh", "xác nhận", "kích hoạt", "đăng nhập", "tài khoản", "mật khẩu",
    "cập nhật", "thông báo", "cảnh báo", "hỗ trợ", "kỹ thuật", "tổng đài", "ngân hàng", "thanh toán", "chuyển khoản",
    "momo", "zalopay", "vnpay", "khôi phục", "đổi mật khẩu", "lấy lại mật khẩu", "tài khoản bị khóa", "khóa tài khoản",
    "sos", "khẩn cấp", "mã otp", "otp", "chứng minh nhân dân", "cccd", "cmnd", "thông tin", "thông tin cá nhân",
    "tải về", "cài đặt", "phần mềm", "ứng dụng", "tiện ích", "quét virus", "diệt virus", "bẻ khóa", "hack", "crack",
    "mã hóa", "ransomware", "miễn phí", "quà tặng", "tặng quà", "nhận quà", "phần thưởng", "giải thưởng", "trúng",
    "khuyến mãi", "giảm giá", "mã giảm giá", "voucher", "ưu đãi", "giá sốc", "giá rẻ", "siêu rẻ",
    "giới hạn", "duy nhất", "hôm nay", "đặc biệt", "độc quyền", "nhận tiền", "thẻ cào", "tham gia", "đăng ký",
    "ngày", "ngay lập tức", "gặp", "mua", "bán", "đặt hàng", "trực tuyến", "nhận", "bấm",
    "cửa hàng", "liên kết", "trang web", "tin tức", "cộng đồng", "diễn đàn", "dịch vụ", "tổng hợp", "hot"
]
common_tlds = {'com', 'org', 'net', 'gov', 'edu'}
country_tlds = {'vn', 'uk', 'de', 'jp', 'fr', 'ca', 'au',
               'cn', 'ru', 'jp', 'kr', 'vn', 'de', 'fr', 'it', 'br', 'in',
               'pl', 'ua', 'ir', 'tr', 'tw', 'th', 'id', 'hk', 'cz', 'sk',
               'hu', 'ro', 'bg', 'by', 'rs', 'kz', 'sa', 'ae', 'pk', 'bd'}

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='.*pandas.*')

PATH_BLACK = "Emb/black"
PATH_CLEAN_MULTI = "Emb/clean/multi"
PATH_CLEAN_VN = "Emb/clean/vn"

RANDOM_STATE = 15052004

class DHML(Dataset):
    def __init__(self, embeddings_domain, embeddings_header, embeddings_content, normal_features, labels):
        self.domain = torch.tensor(np.array(embeddings_domain)).float()
        self.domain_header = torch.tensor(np.array(embeddings_header)).float()
        self.domain_content = torch.tensor(np.array(embeddings_content)).float()
        self.normal_features = torch.tensor(np.array(normal_features)).float()
        self.labels = torch.tensor(np.array(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.domain[idx], self.domain_header[idx], self.domain_content[idx], self.normal_features[idx]), self.labels[idx]

def entropy(s):
    if not s: return 0.0
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log2(count/lns) for count in p.values())

def safe_get(data, keys, default=None):
    if not isinstance(keys, list):
        keys = [keys]
    temp = data
    for key in keys:
        if isinstance(temp, dict):
            temp = temp.get(key)
        if temp is None:
            return default
    return temp

def ip_to_binary_vector(ip_string):
    if pd.isna(ip_string) or not isinstance(ip_string, str):
        return [np.nan] * 32

    octets = ip_string.split('.')

    binary_vector = []
    for octet in octets:
        octet_int = int(octet)
        binary_octet = format(octet_int, '08b')
        binary_vector.extend([int(bit) for bit in binary_octet])
    return binary_vector if len(binary_vector) == 32 else [np.nan] * 32

def get_tld(domain):
    if pd.isna(domain) or not isinstance(domain, str) or '.' not in domain: return None
    parts = domain.split('.')
    if len(parts) < 2 or not parts[0]: return parts[-1].lower() if parts[-1] else None
    if len(parts) > 2 and len(parts[-2]) <= 3 and len(parts[-1]) == 2:
         return f"{parts[-2]}.{parts[-1]}".lower()
    return parts[-1].lower()

def count_black_words(s, words_list):
    if not isinstance(s, str): return 0
    s = s.lower()
    return sum(s.count(word) for word in words_list)

def get_domain_part(domain):
     if pd.isna(domain) or not isinstance(domain, str) or '.' not in domain: return domain
     parts = domain.split('.')
     if len(parts) < 2 or not parts[0]: return None
     tld = get_tld(domain)
     if tld and domain.endswith(f".{tld}"):
          domain_no_tld = domain[:-len(tld)-1]
          return domain_no_tld
     else:
          return parts[0] if len(parts) == 2 else '.'.join(parts[:-1])

def extract_features(df):
    features = pd.DataFrame()

    # Các features liên quan đến domain
    features['length'] = df['domain'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    features['num_subdomains'] = df['domain'].apply(lambda x: str(x).count('.') if pd.notna(x) else 0)
    features['num_dashes'] = df['domain'].apply(lambda x: str(x).count('-') if pd.notna(x) else 0)

    df_safe = df.copy()
    df_safe['tld'] = df_safe['domain'].apply(get_tld)
    df_safe['domain_part'] = df_safe['domain'].apply(get_domain_part)

    features['num_black_words_in_domain'] = df_safe['domain'].apply(lambda d: count_black_words(d, black_words))
    features['is_country_tld'] = df_safe['tld'].apply(lambda x: 1 if isinstance(x, str) and len(x) == 2 and x in country_tlds else 0)
    features['is_common_tld'] = df_safe['tld'].apply(lambda x: 1 if isinstance(x, str) and x in common_tlds else 0)
    features['entropy'] = df_safe['domain_part'].apply(lambda x: entropy(str(x)) if pd.notna(x) else 0)

    # Header Features
    if 'headers' in df.columns:
        df_safe['headers_lower'] = df_safe['headers'].apply(lambda h: {k.lower(): v for k, v in h.items()} if isinstance(h, dict) else {})
        features['has_security'] = df_safe['headers_lower'].apply(lambda h: 1 if isinstance(h, dict) and ('content-security-policy' in h or 'x-frame-options' in h) else 0)
        features['is_error_header'] = df_safe['headers'].apply(lambda h: 1 if isinstance(h, dict) and list(h.keys()) == ['error'] else 0) # Dùng df gốc vì kiểm tra key 'error'
        features['has_cookie'] = df_safe['headers_lower'].apply(lambda h: 1 if isinstance(h, dict) and 'set-cookie' in h else 0)
        features['has_cache_control'] = df_safe['headers_lower'].apply(lambda h: 1 if isinstance(h, dict) and 'cache-control' in h else 0)
    else:
        print("Lỗi header")

    # IP Features (liên quan đến ip và geo)
    if 'ip' in df.columns and 'geo' in df.columns:
        features['num_ipv4'] = df_safe['ip'].apply(lambda x: len(safe_get(x, 'ipV4', [])))
        features['num_ipv6'] = df_safe['ip'].apply(lambda x: len(safe_get(x, 'ipV6', [])))

        def get_first_ipv4_string(ip_data):
            ipv4_list = safe_get(ip_data, 'ipV4', [])
            if ipv4_list and isinstance(ipv4_list, list) and len(ipv4_list) > 0 and isinstance(ipv4_list[0], str):
                return ipv4_list[0]
            return None
        first_ipv4_strings = df_safe['ip'].apply(get_first_ipv4_string)
        binary_ip_vectors = first_ipv4_strings.apply(ip_to_binary_vector)
        ipv4_feature_names = [f'ipv4_bit_{i}' for i in range(32)]
        try:
            ipv4_df = pd.DataFrame(binary_ip_vectors.tolist(), index=df_safe.index, columns=ipv4_feature_names).astype(float)
        except Exception as e_ip_expand:
            print(f"Lỗi IP {e_ip_expand}")
            ipv4_df = binary_ip_vectors.apply(pd.Series)
            if ipv4_df.empty:
                 print("Lỗi ko có ipV4")
                 ipv4_df = pd.DataFrame(np.nan, index=df_safe.index, columns=ipv4_feature_names)
            else:
                 ipv4_df = ipv4_df.rename(columns={i: name for i, name in enumerate(ipv4_feature_names)})

            ipv4_df = ipv4_df[ipv4_feature_names]
            ipv4_df = ipv4_df.astype(float)

        features = pd.concat([features, ipv4_df], axis=1)
    else:
        ip_geo_cols = ['num_ipv4', 'num_ipv6'] + [f'ipv4_bit_{i}' for i in range(32)]
        for col in ip_geo_cols:
             if col.startswith('ipv4_bit_'): features[col] = np.nan
             else: features[col] = 0

    # DNS Features
    if 'dns' in df.columns:
        features['ttl'] = df_safe['dns'].apply(lambda x: safe_get(x, 'ttl', 0))
        features['has_mx'] = df_safe['dns'].apply(lambda x: 1 if safe_get(x, 'mx_records', []) else 0)
        features['num_ns'] = df_safe['dns'].apply(lambda x: len(safe_get(x, 'ns_records', [])))
        features['fluxiness_score'] = pd.to_numeric(df_safe['dns'].apply(lambda x: safe_get(x, 'dns_fluxiness_score', 0)), errors='coerce')
    else:
        dns_cols = ['ttl', 'has_mx', 'num_ns', 'fluxiness_score']
        for col in dns_cols:
            if col == 'has_mx': features[col] = 0
            elif col == 'num_ns': features[col] = 0
            else: features[col] = np.nan

    # Content Features
    if 'content' in df.columns and 'text' in df.columns:
        features['num_links'] = pd.to_numeric(df_safe['content'].apply(lambda x: safe_get(x, 'num_links')), errors='coerce')
        features['num_scripts'] = pd.to_numeric(df_safe['content'].apply(lambda x: safe_get(x, 'num_scripts')), errors='coerce')
        features['num_iframes'] = pd.to_numeric(df_safe['content'].apply(lambda x: safe_get(x, 'num_iframes')), errors='coerce')
        features['has_meta_refresh'] = df_safe['content'].apply(lambda x: 1 if safe_get(x, 'has_meta_refresh', False) else 0)
    else:
        content_cols = ['num_links', 'num_scripts', 'num_iframes', 'has_meta_refresh']
        for col in content_cols:
             if col == 'has_meta_refresh': features[col] = 0
             else: features[col] = np.nan

    return features

def load_data_combined():
    all_data = {
        "domain_emb": [],
        "headers_emb": [],
        "content_emb": [],
        "json_raw": [],
        "label": [],
        "identifier": []
    }
    processed_identifiers = set()
    source_map = {}

    # Xử lý dữ liệu BLACK
    if not os.path.exists(PATH_BLACK):
        print(f"{PATH_BLACK}' không có")
    else:
        for domain_name in os.listdir(PATH_BLACK):
            domain_full_path = os.path.join(PATH_BLACK, domain_name)
            if not os.path.isdir(domain_full_path):
                print(f"Lỗi PATH_BLACK: {domain_name}")
                continue

            identifier = f"black/{domain_name}".strip('/')
            label_int = 1

            # Giả sử prepare_emb(domain_name, parent_folder_of_domain)
            raw_data = prepare_emb(domain_name, PATH_BLACK)

            if raw_data is None or not all(
                    k in raw_data for k in ['raw_json', 'domain_emb', 'headers_emb', 'content_emb']):
                print(f"Lỗi load {identifier}")
                continue

            all_data["domain_emb"].append(raw_data['domain_emb'])
            all_data["headers_emb"].append(raw_data['headers_emb'])
            all_data["content_emb"].append(raw_data['content_emb'])
            all_data["json_raw"].append(raw_data['raw_json'])
            all_data["label"].append(label_int)
            all_data["identifier"].append(identifier)
            processed_identifiers.add(identifier)

            source_map[identifier] = "black"

            print(f"{domain_name} ok")

    # -Xử lý dữ liệu CLEAN
    clean_sources_paths = []
    if 'PATH_CLEAN_MULTI' in globals() and PATH_CLEAN_MULTI:  # Kiểm tra biến có được định nghĩa
        clean_sources_paths.append(PATH_CLEAN_MULTI)
    if 'PATH_CLEAN_VN' in globals() and PATH_CLEAN_VN:
        clean_sources_paths.append(PATH_CLEAN_VN)

    if not clean_sources_paths:
        print("PATH_CLEAN_MULTI, PATH_CLEAN_VN lỗi")

    for path_to_source_type in clean_sources_paths:
        if not os.path.exists(path_to_source_type):
            print(f"{path_to_source_type}' không có")
            continue

        source_name = os.path.basename(path_to_source_type)

        for domain_name in os.listdir(path_to_source_type):
            domain_full_path = os.path.join(path_to_source_type, domain_name)
            if not os.path.isdir(domain_full_path):
                continue

            identifier = f"clean/{source_name}/{domain_name}".strip('/')
            label_int = 0

            raw_data = prepare_emb(domain_name, path_to_source_type)

            if raw_data is None or not all(
                    k in raw_data for k in ['raw_json', 'domain_emb', 'headers_emb', 'content_emb']):
                print(f"Warning: Invalid data from prepare_emb for {identifier}. Skipping.")
                continue

            all_data["domain_emb"].append(raw_data['domain_emb'])
            all_data["headers_emb"].append(raw_data['headers_emb'])
            all_data["content_emb"].append(raw_data['content_emb'])
            all_data["json_raw"].append(raw_data['raw_json'])
            all_data["label"].append(label_int)
            all_data["identifier"].append(identifier)
            processed_identifiers.add(identifier)

            source_map[identifier] = source_name

            print(f"{domain_name} ok")

    print(f"Load xong {len(all_data['label'])} samples")
    return all_data, source_map

CACHE_FILE = "cache/full_data_cache.pkl"
os.makedirs("cache", exist_ok=True)

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        full_cache = pickle.load(f)
    loaded_data = full_cache["loaded_data"]
    source_map = full_cache["source_map"]
    print("Đã tải dữ liệu từ cache")
else:
    print(f"Không tìm thấy cache tại {CACHE_FILE}")
    loaded_data, source_map = load_data_combined()
    full_cache = {
        "loaded_data": loaded_data,
        "source_map": source_map
    }

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(full_cache, f)

json_df_raw = pd.DataFrame(loaded_data["json_raw"], index=loaded_data["identifier"])

extracted_features_df = extract_features(json_df_raw)

identifiers = loaded_data['identifier']
labels = loaded_data['label']
domain_embs = loaded_data['domain_emb']
header_embs = loaded_data['headers_emb']
content_embs = loaded_data['content_emb']

label_counts = pd.Series(labels).value_counts()
print("Mỗi mẫu:\n", label_counts)
can_stratify = all(count >= 2 for count in label_counts) and len(label_counts) > 1

label_source_combo = [
    f"{labels[i]}_{source_map.get(identifiers[i], 'unknown')}"
    for i in range(len(labels))
]

indices = list(range(len(labels)))

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=label_source_combo
)

temp_label_source = [label_source_combo[i] for i in temp_indices]
val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=temp_label_source
)

X_train_dom = [domain_embs[i] for i in train_indices]
X_val_dom   = [domain_embs[i] for i in val_indices]
X_test_dom  = [domain_embs[i] for i in test_indices]

X_train_hdr = [header_embs[i] for i in train_indices]
X_val_hdr   = [header_embs[i] for i in val_indices]
X_test_hdr  = [header_embs[i] for i in test_indices]

X_train_cnt = [content_embs[i] for i in train_indices]
X_val_cnt   = [content_embs[i] for i in val_indices]
X_test_cnt  = [content_embs[i] for i in test_indices]

y_train = [labels[i] for i in train_indices]
y_val   = [labels[i] for i in val_indices]
y_test  = [labels[i] for i in test_indices]

X_train_features_extracted = extracted_features_df.iloc[train_indices]
X_val_features_extracted = extracted_features_df.iloc[val_indices]
X_test_features_extracted = extracted_features_df.iloc[test_indices]

def count_source_dist(indices, identifiers, source_map):
    sources = [source_map.get(identifiers[i], 'black') for i in indices]
    return collections.Counter(sources)

print(f"Train {len(y_train)}: ", count_source_dist(train_indices, identifiers, source_map))
print(f" Val {len(y_val)}: ", count_source_dist(val_indices, identifiers, source_map))
print(f"Test {len(y_test)}: ", count_source_dist(test_indices, identifiers, source_map))


all_cols = extracted_features_df.columns.tolist()

numeric_features = [
    'length', 'num_subdomains', 'num_dashes', 'entropy', 'num_ipv4', 'num_ipv6',
    'ttl', 'num_ns', 'fluxiness_score', 'num_links', 'num_scripts', 'num_iframes',
    'num_black_words_in_domain'
] + [f'ipv4_bit_{i}' for i in range(32)]

binary_features = [
    'is_country_tld', 'is_common_tld', 'has_security', 'is_error_header',
    'has_cookie', 'has_cache_control', 'has_mx', 'has_meta_refresh'
]

numeric_features_present = [f for f in numeric_features if f in all_cols]
binary_features_present = [f for f in binary_features if f in all_cols]

print(f"\nFeatures của normal:")
print(f"  Numeric ({len(numeric_features_present)}): {numeric_features_present}")
print(f"  Binary ({len(binary_features_present)}): {binary_features_present}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
])

transformers_list = []
if numeric_features:
    transformers_list.append(('num', numeric_transformer, numeric_features))
if binary_features:
    binary_pipeline = Pipeline(steps=[
        ('type_caster', 'passthrough'),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])
    transformers_list.append(('bin', binary_pipeline, binary_features))

preprocessor = ColumnTransformer(
    transformers=transformers_list,
    remainder='drop'
)

preprocessor.fit(X_train_features_extracted)

with open("quadra_preprocessor.pkl", "wb") as f_prep:
    pickle.dump(preprocessor, f_prep)
print("Đã lưu quadra_preprocessor.pkl")

X_train_normal_processed = preprocessor.transform(X_train_features_extracted)
X_val_normal_processed = preprocessor.transform(X_val_features_extracted)
X_test_normal_processed = preprocessor.transform(X_test_features_extracted)

processed_feature_names = preprocessor.get_feature_names_out()
processed_feature_names_cleaned = [
    name.replace('[', '_').replace(']', '').replace('<', '_lt_')
    for name in processed_feature_names
]
train_dataset = DHML(X_train_dom, X_train_hdr, X_train_cnt, X_train_normal_processed, y_train)
val_dataset = DHML(X_val_dom, X_val_hdr, X_val_cnt, X_val_normal_processed, y_val)
test_dataset = DHML(X_test_dom, X_test_hdr, X_test_cnt, X_test_normal_processed, y_test)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"X_train_dom size: {len(X_train_dom)} x {X_train_dom[0].shape}")
print(f"X_train_hdr size: {len(X_train_hdr)} x {X_train_hdr[0].shape}")
print(f"X_train_cnt size: {len(X_train_cnt)} x {X_train_cnt[0].shape}")
print(f"X_train_normal size: {len(X_train_normal_processed)} x {X_train_normal_processed[0].shape}")

class QuadraKill(nn.Module):
    def __init__(self, input_dim, normal_dim, domain_dim=128, domain_header_dim=256, domain_content_dim=384, normal_dim_out=384):
        super(QuadraKill, self).__init__()

        self.FC_domain = nn.Sequential(
            nn.Linear(input_dim, domain_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.FC_domain_header = nn.Sequential(
            nn.Linear(input_dim, domain_header_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.FC_domain_content = nn.Sequential(
            nn.Linear(input_dim, domain_content_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.FC_normal = nn.Sequential(
            nn.Linear(normal_dim, normal_dim_out),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        total_input_dim = domain_dim + domain_header_dim + domain_content_dim + normal_dim_out

        self.classifier = nn.Sequential(
            nn.LayerNorm(total_input_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(total_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2)
        )

    def forward(self, domain, header, content, normal_features):
        d = self.FC_domain(domain)
        d_h = self.FC_domain_header(header)
        d_c = self.FC_domain_content(content)
        n = self.FC_normal(normal_features)

        x = torch.cat([d, d_h, d_c, n], dim=1)
        out = self.classifier(x)
        return out

emb_size = train_dataset[0][0][0].shape[0]
normal_feats_size= train_dataset[0][0][3].shape[0]

learning_rate = 5e-4
num_epochs = 10
max_grad_norm = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = QuadraKill(input_dim=emb_size, normal_dim = normal_feats_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

early = 5
best_val_loss = float('inf')
best_model = None
counter = 0

def train(model, train_loader):
    model.train()
    total_loss = 0
    for (dom, dom_head, dom_cont, normal), labels in train_loader:
        dom, dom_head, dom_cont, normal, labels = dom.to(device), dom_head.to(device), dom_cont.to(device), normal.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(dom, dom_head, dom_cont, normal)  # thêm normal vào
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_and_metrics(model, val_loader):
    model.eval()
    total_loss = 0
    correct = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for (dom, hdr, cnt, normal), labels in val_loader:
            dom, hdr, cnt, normal, labels = dom.to(device), hdr.to(device), cnt.to(device), normal.to(device), labels.to(device)

            outputs = model(dom, hdr, cnt, normal)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = correct / len(val_loader.dataset)

    precision_0 = precision_score(all_labels, all_predictions, pos_label=0)
    recall_0 = recall_score(all_labels, all_predictions, pos_label=0)
    f1_0 = f1_score(all_labels, all_predictions, pos_label=0)

    precision_1 = precision_score(all_labels, all_predictions, pos_label=1)
    recall_1 = recall_score(all_labels, all_predictions, pos_label=1)
    f1_1 = f1_score(all_labels, all_predictions, pos_label=1)

    return total_loss / len(val_loader), accuracy, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1, all_labels, all_predictions

checkpoint_path = "test.pt"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    final_loss, final_accuracy, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1, all_labels, all_predictions = evaluate_and_metrics(
        model, test_loader)

    target_names = ["Black", "Clean"]
    print(classification_report(all_labels, all_predictions, target_names=target_names, digits=4))
else:
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader)
        val_loss, val_accuracy, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1, all_labels, all_predictions = evaluate_and_metrics(model, val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.5f}  |  Val Loss: {val_loss:.5f}  |  Val Accuracy: {val_accuracy:.5f}")
        print(f"Class 0 - Precision: {precision_0:.5f}  |  Recall: {recall_0:.5f}  |  F1-Score: {f1_0:.5f}")
        print(f"Class 1 - Precision: {precision_1:.5f}  |  Recall: {recall_1:.5f}  |  F1-Score: {f1_1:.5f}")
        print('-' * 60)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, checkpoint_path)
            counter = 0
        else:
            counter += 1

        if counter >= early:
            print(f"Early stopping at epoch {epoch + 1}")
            model.load_state_dict(best_model)
            break

    final_loss, final_accuracy, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1, all_labels, all_predictions = evaluate_and_metrics(
            model, test_loader)

    target_names = ["Black", "Clean"]
    print(classification_report(all_labels, all_predictions, target_names=target_names, digits=4))


if test_indices:  # Đảm bảo test_indices không rỗng
    # Chọn một mẫu từ tập test, ví dụ mẫu đầu tiên
    sample_test_idx_in_original_data = test_indices[1554]  # Index của mẫu trong dữ liệu gốc `loaded_data`
    sample_identifier = identifiers[sample_test_idx_in_original_data]

    print(f"\n--- In tensor cho mẫu Test đầu tiên (từ TrainModel.py) ---")
    print(f"Identifier của mẫu: {sample_identifier}")
    print(f"Label gốc: {labels[sample_test_idx_in_original_data]} (0=Clean, 1=Black)")

    # Lấy lại các embedding và features gốc cho mẫu này
    # (TRƯỚC KHI đưa vào DataLoader và chuyển thành tensor batch)
    domain_emb_sample_np = domain_embs[sample_test_idx_in_original_data]
    header_emb_sample_np = header_embs[sample_test_idx_in_original_data]
    content_emb_sample_np = content_embs[sample_test_idx_in_original_data]

    # Lấy features đã trích xuất cho mẫu này (TRƯỚC KHI qua preprocessor)
    # X_test_features_extracted là DataFrame, chúng ta cần tìm đúng hàng
    # Hoặc lấy trực tiếp từ extracted_features_df
    # extracted_features_df đã có index là identifiers
    normal_features_extracted_sample_series = extracted_features_df.loc[sample_identifier]

    # Áp dụng preprocessor cho MỘT MẪU NÀY
    # Preprocessor mong đợi DataFrame, nên cần tạo DataFrame một hàng
    normal_features_extracted_sample_df = pd.DataFrame([normal_features_extracted_sample_series])
    normal_features_processed_sample_np = preprocessor.transform(normal_features_extracted_sample_df)

    # Chuyển sang tensor và thêm chiều batch (unsqueeze(0)) để giống với cách Predictor.py chuẩn bị
    domain_emb_sample_tensor = torch.tensor(domain_emb_sample_np, dtype=torch.float32).unsqueeze(0)
    header_emb_sample_tensor = torch.tensor(header_emb_sample_np, dtype=torch.float32).unsqueeze(0)
    content_emb_sample_tensor = torch.tensor(content_emb_sample_np, dtype=torch.float32).unsqueeze(0)
    normal_features_sample_tensor = torch.tensor(normal_features_processed_sample_np,
                                                 dtype=torch.float32)  # preprocessor.transform đã trả về 2D (1, num_features)

    print("\n--- Dữ liệu đầu vào cho model (dạng tensor, đã unsqueeze(0) cho embedding) ---")
    print(f"Domain Embedding Tensor ({domain_emb_sample_tensor.shape}):")
    print(domain_emb_sample_tensor)
    print(f"\nHeader Embedding Tensor ({header_emb_sample_tensor.shape}):")
    print(header_emb_sample_tensor)
    print(f"\nContent Embedding Tensor ({content_emb_sample_tensor.shape}):")
    print(content_emb_sample_tensor)
    print(f"\nNormal Features Tensor ({normal_features_sample_tensor.shape}):")
    print(normal_features_sample_tensor)

    # (Tùy chọn) Lấy output của model cho mẫu này để so sánh
    model.eval()  # Đảm bảo model ở eval mode
    with torch.no_grad():
        # Đưa tensor lên device
        domain_emb_sample_tensor_dev = domain_emb_sample_tensor.to(device)
        header_emb_sample_tensor_dev = header_emb_sample_tensor.to(device)
        content_emb_sample_tensor_dev = content_emb_sample_tensor.to(device)
        normal_features_sample_tensor_dev = normal_features_sample_tensor.to(device)

        output_sample = model(domain_emb_sample_tensor_dev,
                              header_emb_sample_tensor_dev,
                              content_emb_sample_tensor_dev,
                              normal_features_sample_tensor_dev)

        import torch.nn.functional as F

        softmax_output_sample = F.softmax(output_sample, dim=1)
        predicted_class_sample = torch.argmax(softmax_output_sample, dim=1)
        predicted_prob_sample = softmax_output_sample[0][predicted_class_sample].item()

        label_map = {0: "Clean", 1: "Black"}  # Dựa trên label_counts (0=Clean, 1=Black)
        predicted_label_str_sample = label_map.get(predicted_class_sample.item(), "Unknown")

        print(f"\n--- Dự đoán của model cho mẫu này (từ TrainModel.py) ---")
        print(f"Raw Output (logits): {output_sample.cpu().numpy()}")
        print(f"Softmax Output: {softmax_output_sample.cpu().numpy()}")
        print(f"Predicted Class Index: {predicted_class_sample.item()}")
        print(f"Predicted Label: {predicted_label_str_sample}")
        print(f"Predicted Probability: {predicted_prob_sample:.4f}")

