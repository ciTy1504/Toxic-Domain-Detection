import os
from Config_Loader import load_config
import pandas as pd
import math
import numpy as np
from collections import Counter
import pickle
import torch
from Embedder import prepare_emb

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

#Các hàm tiện ích để extract features
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

#Extract Features
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
    features['is_country_tld'] = df_safe['tld'].apply(
        lambda x: 1 if isinstance(x, str) and len(x) == 2 and x in country_tlds else 0)
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
            ipv4_df = pd.DataFrame(binary_ip_vectors.tolist(), index=df_safe.index,
                                   columns=ipv4_feature_names).astype(float)
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
        features['fluxiness_score'] = pd.to_numeric(
            df_safe['dns'].apply(lambda x: safe_get(x, 'dns_fluxiness_score', 0)), errors='coerce')
    else:
        dns_cols = ['ttl', 'has_mx', 'num_ns', 'fluxiness_score']
        for col in dns_cols:
            if col == 'has_mx':features[col] = 0
            elif col == 'num_ns':features[col] = 0
            else:features[col] = np.nan

    # Content Features
    if 'content' in df.columns and 'text' in df.columns:
        features['num_links'] = pd.to_numeric(df_safe['content'].apply(lambda x: safe_get(x, 'num_links')),errors='coerce')
        features['num_scripts'] = pd.to_numeric(df_safe['content'].apply(lambda x: safe_get(x, 'num_scripts')),errors='coerce')
        features['num_iframes'] = pd.to_numeric(df_safe['content'].apply(lambda x: safe_get(x, 'num_iframes')),errors='coerce')
        features['has_meta_refresh'] = df_safe['content'].apply(lambda x: 1 if safe_get(x, 'has_meta_refresh', False) else 0)
    else:
        content_cols = ['num_links', 'num_scripts', 'num_iframes', 'has_meta_refresh']
        for col in content_cols:
            if col == 'has_meta_refresh':features[col] = 0
            else:features[col] = np.nan

    return features

def processor(domain, config):
    flag, raw_data = prepare_emb(domain)

    if flag:
        if raw_data is None or not all(k in raw_data for k in ['raw_json', 'domain_emb', 'headers_emb', 'content_emb']):
            raise ValueError("Dữ liệu không đầy đủ để xử lý")

        json_df_single = pd.DataFrame([raw_data['raw_json']])
        extracted_single_df = extract_features(json_df_single)

        preprocessor_path = os.path.join(config["model_repo"], config["preprocessor_path"])
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

        X_single_normal_processed = preprocessor.transform(extracted_single_df)

        domain_emb_tensor = torch.tensor(raw_data['domain_emb']).unsqueeze(0)
        header_emb_tensor = torch.tensor(raw_data['headers_emb']).unsqueeze(0)
        content_emb_tensor = torch.tensor(raw_data['content_emb']).unsqueeze(0)
        normal_feat_tensor = torch.tensor(X_single_normal_processed, dtype=torch.float32)

        return True, {
            'domain_emb': domain_emb_tensor,
            'headers_emb': header_emb_tensor,
            'content_emb': content_emb_tensor,
            'features': normal_feat_tensor,
            'raw_json': raw_data['raw_json']
        }
    else:
        return False, raw_data if raw_data else None

if __name__ == "__main__":
    test_domain = "hust.edu.vn"
    config = load_config()  # Giả sử load_config() trả về cấu hình từ một file hoặc một nguồn khác

    flag, result = processor(test_domain, config)

    if flag:
        print("Processor thành công!")
        print("Domain Embedding Tensor: ", result['domain_emb'])
        print("Headers Embedding Tensor: ", result['headers_emb'])
        print("Content Embedding Tensor: ", result['content_emb'])
        print("Normal Features Tensor: ", result['features'])
        print("Raw JSON: ", result['raw_json'])
    else:
        print("Xảy ra lỗi trong quá trình xử lý.")
        print("Dữ liệu không hợp lệ:", result)

