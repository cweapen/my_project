import requests
import os

def download_wikitext2(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    urls = {
        "train.txt": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1/wiki.train.raw",
        "valid.txt": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1/wiki.valid.raw"
    }
    for name, url in urls.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            text = requests.get(url).text
            with open(path, "w", encoding="utf-8") as f:
                f.write(text[:50000])  # 可选：只取前 5 万字符做小数据集
            print(f"{name} 下载完成")
