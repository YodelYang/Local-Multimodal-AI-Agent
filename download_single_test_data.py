import os
import requests
import json
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "test_downloads")

SINGLE_TEST_DIR = os.path.join(DOWNLOAD_DIR, "single_pdf")

MAIN_GT_FILE = os.path.join(DOWNLOAD_DIR, "ground_truth.json")


TARGET_PAPER = ("GPT-4 Technical Report", "2303.08774", "Large Language Models")

def download_file(url, filepath):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return True
        else:
            print(f"❌ Failed (Status {response.status_code}): {url}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():

    if not os.path.exists(SINGLE_TEST_DIR):
        os.makedirs(SINGLE_TEST_DIR)
    else:
        for filename in os.listdir(SINGLE_TEST_DIR):
            file_path = os.path.join(SINGLE_TEST_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                pass

    title, arxiv_id, category = TARGET_PAPER

    safe_title = "".join([c if c.isalnum() or c in " .-_" else "" for c in title])
    filename = f"{safe_title}.pdf"
    filepath = os.path.join(SINGLE_TEST_DIR, filename)
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    print("-" * 50)

    print(f"⬇️  Downloading [{category}]: {safe_title}...")
    
    if download_file(url, filepath):
        gt_data = {}
        if os.path.exists(MAIN_GT_FILE):
            with open(MAIN_GT_FILE, 'r', encoding='utf-8') as f:
                try:
                    gt_data = json.load(f)
                except json.JSONDecodeError:
                    gt_data = {}
        
        gt_data[filename] = category
        
        with open(MAIN_GT_FILE, 'w', encoding='utf-8') as f:
            json.dump(gt_data, f, indent=4, ensure_ascii=False)


        print("-" * 50)
        print("✅ Single Download Complete.")
    else:
        print("❌ Download failed. Test aborted.")

if __name__ == "__main__":
    main()