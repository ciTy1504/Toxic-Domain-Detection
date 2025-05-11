import os
import json
import time
import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import dns.resolver
from Config_Loader import load_config
from Content_Checker import check_content
import requests

# Phần scrape content
def setup_driver(chromedriver_path, config):
    options = Options()
    driver_config = config.get("driver", {})

    # Tùy chỉnh trong config.yalm
    if driver_config.get("headless", True):
        options.add_argument("--headless")

    log_level = driver_config.get("log_level", 3)
    options.add_argument(f"--log-level={log_level}")

    user_agent = driver_config.get("user_agent", "")
    if user_agent:
        options.add_argument(f"user-agent={user_agent}")

    # Cài chế độ mặc định
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--incognito")

    service = Service(chromedriver_path)
    try:
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception as e:
        print(f"Lỗi driver: {e}", file=sys.stderr)
        return None

def get_content_and_headers(driver, domain, config):
    text = ""
    user_agent = config.get('user_agent', "Mozilla/5.0")
    scroll_pauses = config.get('scroll_pauses')
    scroll_delay = config.get('scroll_delay')

    # Thực hiện yêu cầu HTTP để lấy header
    try:
        response = requests.get(domain, timeout=3, headers={"User-Agent": user_agent})
        headers = dict(response.headers)
    except Exception as e:
        headers = {"error": str(e)}

    # Tiến hành xử lý với Selenium để crawl content
    try:
        driver.get(domain)
        WebDriverWait(driver, config['selenium_timeout']).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(config['page_load_delay'])

        # Cuộn trang nếu cần
        for _ in range(scroll_pauses):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_delay)

        visible_text = driver.find_element(By.TAG_NAME, "body").text
        lines = [line.strip() for line in visible_text.split("\n") if line.strip()]

        text = " ".join(lines)

        return text, headers, None
    except Exception as e:
        error_msg = f"Selenium crawling error for {domain}: {e}"
        print(error_msg, file=sys.stderr)
        return text, headers, error_msg

# Phần lấy các Features
def get_ip_addresses(domain_name):
    ipV4 = []
    ipV6 = []
    try:
        ipV4_records = dns.resolver.resolve(domain_name, 'A')
        ipV4 = sorted([ip.to_text() for ip in ipV4_records])
    except Exception as e:
        print(f"Không tìm được IPV4 {domain_name}: {e}", file=sys.stderr)

    try:
        ipV6_records = dns.resolver.resolve(domain_name, 'AAAA')
        ipV6 = sorted([ip.to_text() for ip in ipV6_records])
    except Exception as e:
        print(f"Không tìm được IPV6 {domain_name}: {e}", file=sys.stderr)
    return ipV4, ipV6

def get_whois_info(domain):
    import whois
    import datetime

    registrar = None
    registrant = None
    creation_date_str = None
    expiration_date_str = None
    domain_age = None
    creation_date_obj = None
    expiration_date_obj = None

    try:
        w = whois.whois(domain)

        if w and getattr(w, 'domain_name', None):
            registrar = w.registrar
            registrant = getattr(w, 'registrant', None) or \
                         getattr(w, 'registrant_name', None) or \
                         getattr(w, 'org', None)

            creation_date = getattr(w, 'creation_date', None)
            if isinstance(creation_date, list) and creation_date:
                creation_date_obj = creation_date[0]
            elif isinstance(creation_date, datetime.datetime):
                creation_date_obj = creation_date

            expiration_date = getattr(w, 'expiration_date', None)
            if isinstance(expiration_date, list) and expiration_date:
                expiration_date_obj = expiration_date[0]
            elif isinstance(expiration_date, datetime.datetime):
                expiration_date_obj = expiration_date

            if isinstance(creation_date_obj, datetime.datetime):
                 creation_date_str = creation_date_obj.strftime('%Y-%m-%d %H:%M:%S') # Standard format
            if isinstance(expiration_date_obj, datetime.datetime):
                 expiration_date_str = expiration_date_obj.strftime('%Y-%m-%d %H:%M:%S') # Standard format

            now = datetime.datetime.now()
            start_date_for_age = creation_date_obj if isinstance(creation_date_obj, datetime.datetime) else None

            if start_date_for_age:
                 domain_age = (now - start_date_for_age).days

        else:
            print("")
    except whois.parser.PywhoisError as e:
         print(f"")
    except Exception as e:
        print(f"Error during WHOIS lookup for {domain}: {e}")

    return {
        "registrar": registrar if registrar else None,
        "registrant": registrant if registrant else None,
        "registration_date": creation_date_str if creation_date_str else None,
        "expiration_date": expiration_date_str if expiration_date_str else None,
        "domain_age_days": domain_age if domain_age is not None else None
    }

def get_dns_info(domain):
    dns_info = {
        "ttl": None,
        "a_records": [],
        "mx_records": [],
        "ns_records": [],
        "dns_fluxiness_score": None,
        "num_ips_last_week": None
    }

    #Config server gg và cloudfare
    resolver = dns.resolver.Resolver()
    resolver.nameservers = ['8.8.8.8', '1.1.1.1']
    resolver.timeout = 2
    resolver.lifetime = 5

    #Lấy tên A, mail, ns
    record_types = ['A', 'MX', 'NS']
    for record_type in record_types:
        try:
            answers = resolver.resolve(domain, record_type)
            if record_type == 'A':
                dns_info["a_records"] = sorted([rdata.address for rdata in answers])
                if hasattr(answers, 'rrset') and hasattr(answers.rrset, 'ttl'):
                     dns_info["ttl"] = answers.rrset.ttl
            elif record_type == 'MX':
                mx_data = sorted([(rdata.preference, rdata.exchange.to_text(omit_final_dot=True)) for rdata in answers])
                dns_info["mx_records"] = [f"{pref} {exch}." for pref, exch in mx_data]
            elif record_type == 'NS':
                dns_info["ns_records"] = sorted([rdata.target.to_text(omit_final_dot=True) + "." for rdata in answers])

        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN) as e:
            print(f"[{record_type} record] {domain}: Not found ({type(e).__name__}).")
        except (dns.resolver.Timeout, dns.resolver.NoNameservers) as e:
             print(f"[{record_type} record error] {domain}: {e}")
        except Exception as e:
            print(f"[{record_type} record general error] {domain}: {e}")

    if dns_info["a_records"]:
        dns_info["dns_fluxiness_score"] = len(set(dns_info["a_records"]))
    return dns_info

def get_ip_enrichment_info(target_ip):
    import socket
    import ipwhois

    enrichment_info = {
        "as_number": None,
        "is_ip_in_domain": None,
        "is_reverse_dns_valid": None,
        "reverse_ptr": "Not found"
    }

    # Kiểm tra xem có IP hợp lệ được cung cấp không
    if not target_ip or not isinstance(target_ip, str):
        print(f"Warning: No valid target IP provided for enrichment lookup.")
        return enrichment_info

    try:
        # ipwhois hoạt động với cả IPv4 và IPv6
        ip_obj = ipwhois.IPWhois(target_ip)
        # Thêm timeout và retry cho RDAP lookup để tăng độ tin cậy
        result = ip_obj.lookup_rdap(depth=1, retry_count=2, timeout=10)

        if result:
            enrichment_info["as_number"] = result.get('asn')
            enrichment_info["is_ip_in_domain"] = result.get('network') is not None

            # Cố gắng tìm tên ngược từ RDAP
            reverse_name = result.get('name')
            # Thử tìm trong các cấu trúc object phổ biến nếu 'name' không có
            if not reverse_name and 'objects' in result:
                 for key, obj_data in result['objects'].items():
                     contact_name = obj_data.get('contact', {}).get('name')
                     entity_name = None
                     if 'entities' in obj_data and isinstance(obj_data['entities'], list) and obj_data['entities']:
                          # Lấy tên từ entity đầu tiên nếu có
                          entity_name = obj_data['entities'][0].get('contact', {}).get('name')

                     # Ưu tiên contact name hoặc entity name
                     reverse_name = contact_name or entity_name
                     if reverse_name:
                         break # Đã tìm thấy tên, thoát vòng lặp

            if reverse_name:
                 enrichment_info["reverse_ptr"] = reverse_name
                 enrichment_info["is_reverse_dns_valid"] = True
                 print(f"Info: Found reverse DNS via RDAP for {target_ip}: {reverse_name}")
            else:
                 # Nếu RDAP không có tên, đánh dấu là không tìm thấy và thử fallback
                 enrichment_info["reverse_ptr"] = "Not found"
                 enrichment_info["is_reverse_dns_valid"] = False
                 print(f"Info: RDAP did not provide reverse DNS for {target_ip}. Attempting socket lookup.")
                 try:
                    # Đặt timeout cục bộ cho gethostbyaddr
                    original_timeout = socket.getdefaulttimeout()
                    socket.setdefaulttimeout(5) # 5 giây timeout
                    hostname, _, _ = socket.gethostbyaddr(target_ip)
                    socket.setdefaulttimeout(original_timeout) # Khôi phục timeout gốc
                    enrichment_info["reverse_ptr"] = hostname
                    enrichment_info["is_reverse_dns_valid"] = True
                    print(f"Info: Found reverse DNS via socket for {target_ip}: {hostname}")
                 except socket.herror:
                    print(f"Info: No reverse DNS found via socket for {target_ip}.")
                    # Giữ nguyên giá trị mặc định "Not found" và False
                 except socket.timeout:
                     print(f"Error: Socket gethostbyaddr timed out for {target_ip}.")
                     socket.setdefaulttimeout(original_timeout) # Khôi phục timeout gốc
                 except Exception as e_sock:
                    print(f"Error during socket gethostbyaddr for {target_ip}: {e_sock}")
                    socket.setdefaulttimeout(original_timeout) # Khôi phục timeout gốc

        else:
            # Nếu RDAP không trả về kết quả gì cả
            print(f"Warning: IPWhois lookup (RDAP) for {target_ip} returned no result. Attempting socket reverse DNS.")
            # Fallback duy nhất là socket reverse DNS
            try:
                original_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(5)
                hostname, _, _ = socket.gethostbyaddr(target_ip)
                socket.setdefaulttimeout(original_timeout)
                enrichment_info["reverse_ptr"] = hostname
                enrichment_info["is_reverse_dns_valid"] = True
                print(f"Info: Found reverse DNS via socket for {target_ip}: {hostname}")
            except socket.herror:
                print(f"Info: No reverse DNS found via socket for {target_ip}.")
                enrichment_info["reverse_ptr"] = "Not found"
                enrichment_info["is_reverse_dns_valid"] = False
            except socket.timeout:
                print(f"Error: Socket gethostbyaddr timed out for {target_ip}.")
                socket.setdefaulttimeout(original_timeout)
            except Exception as e_sock:
                print(f"Error during socket gethostbyaddr for {target_ip}: {e_sock}")
                socket.setdefaulttimeout(original_timeout)

    # Xử lý các exceptions cụ thể của ipwhois và requests
    except ipwhois.exceptions.IPDefinedError as e:
         print(f"Warning: IPWhois defined error for {target_ip}: {e}")
    except ipwhois.exceptions.ASNRegistryError as e:
         print(f"Warning: IPWhois ASN registry error for {target_ip}: {e}")
    except ipwhois.exceptions.WhoisRateLimitError as e:
         print(f"CRITICAL: IPWhois Rate Limit Hit for {target_ip}: {e}. Consider adding delays.")
         time.sleep(5) # Sleep nếu bị rate limit
    except requests.exceptions.Timeout: # Bắt timeout từ requests (thường dùng bởi ipwhois)
        print(f"Error: IPWhois lookup (RDAP) timed out for {target_ip}.")
    except requests.exceptions.RequestException as e_req: # Bắt các lỗi mạng chung
         print(f"Error: Network error during IPWhois lookup for {target_ip}: {e_req}")
    except Exception as e_ipw_generic: # Bắt các lỗi không mong muốn khác
        print(f"Error during IP enrichment processing for {target_ip}: {e_ipw_generic} ({type(e_ipw_generic).__name__})")
    finally:
        socket.setdefaulttimeout(None)
    return enrichment_info

def get_geo_info(ip, config):
    if not ip:
        print("Warning: No IP provided for geo lookup.")
        return {}

    try:
        import requests
        access_token = config.get('ipinfo_token')

        url = f'https://ipinfo.io/{ip}?token={access_token}'

        response = requests.get(url)

        if response and response.status_code == 200:
            geo_info = response.json()
            return geo_info
        else:
            print(f"Warning: Failed to retrieve GEO info for IP {ip}, Status Code: {response.status_code}",
                  file=sys.stderr)
            return {}
    except Exception as e:
        print(f"Error during geo lookup for {ip}: {e}")
        return {}

def analyze_html(domain):
    from bs4 import BeautifulSoup
    import requests, time, sys

    results = {
        "num_links": 0,
        "num_scripts": 0,
        "num_iframes": 0,
        "has_meta_refresh": False
    }

    if not isinstance(domain, str) or not domain.strip():
        print("Warning: Invalid or empty domain provided for HTML analysis.")
        return results

    urls_to_try = [f"http://{domain}", f"https://{domain}"]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive'
    }

    html_content, final_url = None, None

    for i, url in enumerate(urls_to_try):
        try:
            resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "").lower()
            if "html" in content_type:
                html_content = resp.text
                final_url = resp.url
                break
            else:
                print(f"[WARN] Non-HTML Content-Type: {content_type}")

        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] Timeout while connecting to {url}")
        except requests.exceptions.SSLError as e:
            print(f"[SSL ERROR] {url}: {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"[CONNECTION ERROR] {url}: {e}")
        except requests.exceptions.HTTPError as e:
            print(f"[HTTP ERROR] {url}: {e}")
        except requests.exceptions.RequestException as e:
            print(f"[REQUEST ERROR] {url}: {e}")

        if i == 0:
            time.sleep(0.5)

    if not html_content:
        print(f"[FAILURE] Unable to retrieve valid HTML content from {domain}")
        return results

    try:
        soup = BeautifulSoup(html_content, "html.parser")
        results["num_links"] = len(soup.find_all("a", href=True))
        results["num_scripts"] = len(soup.find_all("script"))
        results["num_iframes"] = len(soup.find_all("iframe"))
        results["has_meta_refresh"] = bool(
            soup.find("meta", attrs={"http-equiv": lambda x: x and x.lower() == "refresh"})
        )
    except Exception as e:
        print(f"[PARSING ERROR] Failed analyzing HTML for {final_url or domain}: {e}")

    return results

#Hàm scrape để gọi
def scraper(domain_name, config):
    from typing import Any

    clean_domain_for_path = domain_name.replace("http://", "").replace("https://", "").split('/')[0].lower()

    output_domain_folder = os.path.join(config['output_base_path'], clean_domain_for_path)
    output_json_path = os.path.join(output_domain_folder, f"{clean_domain_for_path}.json")

    if os.path.exists(output_json_path):
        all_data = json.load(open(output_json_path, encoding="utf-8-sig"))
    else:
        os.makedirs(output_domain_folder, exist_ok=True)

        domain_url_to_crawl = domain_name
        if not domain_url_to_crawl.startswith("http://") and not domain_url_to_crawl.startswith("https://"):
            domain_url_to_crawl = "https://" + domain_url_to_crawl

        # Cấu trúc file JSON
        all_data = {
            "domain": clean_domain_for_path,
            "text": None,
            "headers": {},
            "ip": {
                "ipV4": [],
                "ipV6": [],
                "as_number": None,
                "is_ip_in_domain": None,
                "is_reverse_dns_valid": None,
                "reverse_ptr": None
            },
            "whois": None,
            "dns": None,
            "geo": None,
            "content": dict[str, Any] | None,
            "errors": []
        }

        # 1. Content và Header
        selenium_driver = None
        try:
            selenium_driver = setup_driver(config['chromedriver_path'], config)
            if selenium_driver:
                print(f"Đang xử lý: {domain_url_to_crawl}")
                text_lines, headers_dict, selenium_error = get_content_and_headers(selenium_driver,domain_url_to_crawl,config)
                all_data["text"] = text_lines
                all_data["headers"] = headers_dict
                if selenium_error:
                    all_data["errors"].append(f"Selenium/Content Fetch: {selenium_error}")
            else:
                err_msg = "Selenium/Content Fetch: Failed to setup WebDriver."
                all_data["errors"].append(err_msg)
                all_data["headers"] = {"error": err_msg}
                all_data["text"] = []
        except Exception as e:
            err_msg = f"Selenium/Content Fetch General Error: {str(e)}"
            all_data["errors"].append(err_msg)
            if not all_data["headers"]: all_data["headers"] = {"error": err_msg}
            if not all_data["text"]: all_data["text"] = []
            print(f"Lỗi crawl content header {domain_name}: {e}", file=sys.stderr)
        finally:
            if selenium_driver:
                selenium_driver.quit()

        # 2. IP
        primary_ip_for_lookups = None
        try:
            ipv4, ipv6 = get_ip_addresses(clean_domain_for_path)

            # Xác định IP chính để làm giàu và lấy geo (ưu tiên IP đầu tiên của danh sách IPv4 không trống)
            if ipv4:  # Kiểm tra xem ipv4 có rỗng không
                primary_ip_for_lookups = ipv4[0]
            elif ipv6:  # Nếu không có IPv4 thì chọn IPv6
                primary_ip_for_lookups = ipv6[0].split('%')[0]

            # Kiểm tra nếu primary_ip_for_lookups hợp lệ trước khi thực hiện enrichment
            if primary_ip_for_lookups:
                try:
                    time.sleep(0.5)
                    enrichment_info = get_ip_enrichment_info(primary_ip_for_lookups)

                    # Cập nhật thông tin IP vào all_data
                    all_data["ip"].update({
                        "ipV4": ipv4,
                        "ipV6": ipv6,
                        "as_number": enrichment_info.get("as_number", None),
                        "is_ip_in_domain": enrichment_info.get("is_ip_in_domain", None),
                        "is_reverse_dns_valid": enrichment_info.get("is_reverse_dns_valid", None),
                        "reverse_ptr": enrichment_info.get("reverse_ptr", None)
                    })
                except Exception as e:
                    err_msg = f"IP Enrichment ({primary_ip_for_lookups}): {str(e)}"
                    all_data["errors"].append(err_msg)
                    print(f"Error getting IP enrichment for {primary_ip_for_lookups}: {e}", file=sys.stderr)
            else:
                # Nếu không có IP hợp lệ, cập nhật các trường với giá trị null
                all_data["ip"].update({
                    "ipV4": ipv4,
                    "ipV6": ipv6,
                    "as_number": None,
                    "is_ip_in_domain": None,
                    "is_reverse_dns_valid": None,
                    "reverse_ptr": None
                })
                print(f"Ko có IP {clean_domain_for_path}.", file=sys.stderr)

        except Exception as e:
            all_data["errors"].append(f"IP Address Lookup ({clean_domain_for_path}): {str(e)}")
            print(f"Lỗi IP {clean_domain_for_path}: {e}", file=sys.stderr)

        # 3. WHOIS
        try:
            time.sleep(1)
            all_data["whois"] = get_whois_info(clean_domain_for_path)
        except Exception as e:
            all_data["errors"].append(f"WHOIS Info ({clean_domain_for_path}): {str(e)}")
            print(f"Lỗi WHOIS {clean_domain_for_path}: {e}", file=sys.stderr)

        # 4. DNS
        try:
            all_data["dns"] = get_dns_info(clean_domain_for_path)
        except Exception as e:
            all_data["errors"].append(f"DNS Info ({clean_domain_for_path}): {str(e)}")
            print(f"Lỗi DNS {clean_domain_for_path}: {e}", file=sys.stderr)

        # 5. Geo
        if primary_ip_for_lookups:
            try:
                time.sleep(1)
                all_data["geo"] = get_geo_info(primary_ip_for_lookups, config)
            except Exception as e:
                err_msg = f"Geo Info ({primary_ip_for_lookups}): {str(e)}"
                all_data["errors"].append(err_msg)
                all_data["geo"] = {"error": str(e)}
        else:
            all_data["errors"].append("IP Enrichment/Geo: No primary IP address found for these lookups.")

        # 7. Phân tích html
        try:
            all_data["content"] = analyze_html(clean_domain_for_path)
            if isinstance(all_data["content"], dict) and all_data["content"].get("error"):
                all_data["errors"].append(f"HTML Analysis ({clean_domain_for_path}): {all_data['html_analysis']['error']}")
        except Exception as e:
            err_msg = f"HTML Analysis ({clean_domain_for_path}) General Error: {str(e)}"
            all_data["errors"].append(err_msg)
            print(f"Error getting HTML analysis for {clean_domain_for_path}: {e}", file=sys.stderr)
            all_data["content"] = {"error": str(e)}

        # Lưu JSON
        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=4, default=str)  # default=str for datetime etc.
        except Exception as e:
            print(f"Lỗi lưu {output_json_path}: {e}", file=sys.stderr)

    #Thêm cờ để check lỗi crawl
    flags = ["Ok", "Err", "Ambiguous"]

    text_lines = all_data["text"]
    checker = check_content(text_lines)
    log_file = config["error_base"]

    if checker == "blank" or checker == "short" :
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{domain_name} {checker}\n")
        print(f"content ngắn/rỗng: {len(all_data['text'])}")
        return flags[2], all_data

    if checker:
        print("content lỗi")
        return flags[1], all_data

    return flags[0], all_data

if __name__ == "__main__":
    config = load_config()

    if len(sys.argv) > 1:
        domain_to_process = sys.argv[1]
    else:
        domain_to_process = "yahoo.com"
        if not domain_to_process:
            sys.exit(1)

    print(json.dumps(scraper(domain_to_process, config), ensure_ascii=False, indent=4))