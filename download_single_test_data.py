import os
import requests
import json
import shutil

# =================配置区域=================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 或者硬编码: BASE_DIR = "/amax/home/dywang/course_work/Multimodal/LocalAI_Agent"
DOWNLOAD_DIR = os.path.join(BASE_DIR, "test_downloads")
# 创建一个专门用于单独测试的子文件夹
SINGLE_TEST_DIR = os.path.join(DOWNLOAD_DIR, "single_pdf")
# Ground Truth 文件必须是 main.py 能读取到的那个主文件
MAIN_GT_FILE = os.path.join(DOWNLOAD_DIR, "ground_truth.json")

# ================= 单独测试的论文 =================
# Title: GPT-4 Technical Report
# Category: Large Language Models
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
    # 1. 准备目录
    if not os.path.exists(SINGLE_TEST_DIR):
        os.makedirs(SINGLE_TEST_DIR)
        # print(f"📂 Created directory: {SINGLE_TEST_DIR}") # 可选：为了保持输出整洁，这行可以注释掉
    else:
        # 清空文件夹
        for filename in os.listdir(SINGLE_TEST_DIR):
            file_path = os.path.join(SINGLE_TEST_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                pass
                # print(f"⚠️ Failed to delete {file_path}. Reason: {e}")

    title, arxiv_id, category = TARGET_PAPER
    
    # 2. 构建文件名和 URL
    safe_title = "".join([c if c.isalnum() or c in " .-_" else "" for c in title])
    filename = f"{safe_title}.pdf"
    filepath = os.path.join(SINGLE_TEST_DIR, filename)
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    # 3. 开始下载 (修改了这里的输出格式)
    print("-" * 50)
    # === 修改点：格式与批量脚本保持一致 ===
    print(f"⬇️  Downloading [{category}]: {safe_title}...")
    
    if download_file(url, filepath):
        # 4. 更新主 Ground Truth 文件
        gt_data = {}
        if os.path.exists(MAIN_GT_FILE):
            with open(MAIN_GT_FILE, 'r', encoding='utf-8') as f:
                try:
                    gt_data = json.load(f)
                except json.JSONDecodeError:
                    gt_data = {}
        
        # 添加/更新这篇论文的分类信息
        gt_data[filename] = category
        
        with open(MAIN_GT_FILE, 'w', encoding='utf-8') as f:
            json.dump(gt_data, f, indent=4, ensure_ascii=False)
        
        # print(f"📘 Updated Ground Truth in: {MAIN_GT_FILE}") # 可选：隐藏内部细节日志
        
        # 5. 输出测试命令
        print("-" * 50)
        print("✅ Single Download Complete.")
    else:
        print("❌ Download failed. Test aborted.")

if __name__ == "__main__":
    main()