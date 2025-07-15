import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/114.0"
}


def clean_url(url):
    parsed = urlparse(url)
    return parsed.netloc.replace('.', '_')


def extract_text_from_url(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"URL {url} not reachable.")
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        # for unwanted in soup.find_all(['header', 'footer', 'nav', 'aside']):
        #     unwanted.decompose()
        texts = soup.find_all(['h1', 'h2', 'h3', 'p', 'span', 'li', "strong"])
        combined = " ".join(t.get_text(strip=True) for t in texts if t.get_text(strip=True))
        return combined
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def scrape_all_urls(input_file="data/urls.txt", output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, "r") as f:
        urls = f.read().splitlines()

    for i, url in enumerate(urls):
        text = extract_text_from_url(url)
        if text:
            filename = f"{i:03d}_{clean_url(url)}.txt"
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f_out:
                f_out.write(text)
        time.sleep(1)


if __name__ == "__main__":
    scrape_all_urls()
