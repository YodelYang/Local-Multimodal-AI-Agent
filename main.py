import os

# ================= 显卡配置 =================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ===========================================

import argparse
import uuid
import json
from tqdm import tqdm
from src.config import PAPERS_DIR, IMAGES_DIR, GT_FILE # 确保 config 导出 GT_FILE
from src.models import ModelManager
from src.database import VectorDB
from src.utils import extract_text_from_pdf, load_image, copy_file, get_files_in_directory

model_manager = ModelManager()
db = VectorDB()

def add_paper(file_path, topics_str):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None, None

    filename = os.path.basename(file_path)
    
    # 1. 尝试提取文本
    text = extract_text_from_pdf(file_path)
    
    if not text:
        text = f"Title: {filename}" 

    # 3. 自动分类
    topics = [t.strip() for t in topics_str.split(",")]
    
    best_topic = model_manager.classify_paper_with_llm(text, filename, topics)
    
    print(f"✅ {filename[:40]:<40} -> \033[92m[{best_topic}]\033[0m")

    # 4. 复制文件 (原文件保留)
    target_dir = os.path.join(PAPERS_DIR, best_topic)
    new_path = copy_file(file_path, target_dir, filename)
    
    # 5. 生成向量并存储
    embedding = model_manager.get_text_embedding(text)
    doc_id = str(uuid.uuid4())
    metadata = {
        "filename": filename,
        "original_name": filename,
        "path": new_path,
        "topic": best_topic,
        "content_snippet": text[:200].replace('\n', ' ') if text else "No preview available"
    }
    db.add_paper(doc_id, embedding, metadata)
    
    return filename, best_topic

def batch_organize(folder_path, topics_str):
    pdf_files = get_files_in_directory(folder_path, [".pdf"])
    print(f"📦 Found {len(pdf_files)} PDFs in {folder_path}...")
    
    # 加载 Ground Truth
    ground_truth = {}
    if os.path.exists(GT_FILE):
        with open(GT_FILE, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        print(f"📘 Loaded Ground Truth for {len(ground_truth)} files.")
    else:
        print("⚠️ Warning: ground_truth.json not found.")

    correct_count = 0
    total_processed = 0
    errors = []
    
    distribution = {t.strip(): 0 for t in topics_str.split(",")}

    for pdf in tqdm(pdf_files, desc="Processing", unit="paper"):
        real_name, predicted_topic = add_paper(pdf, topics_str)
        
        if real_name and predicted_topic:
            total_processed += 1
            if predicted_topic in distribution:
                distribution[predicted_topic] += 1
            
            true_label = ground_truth.get(real_name)
            
            if true_label:
                if predicted_topic == true_label:
                    correct_count += 1
                else:
                    errors.append({
                        "file": real_name,
                        "predicted": predicted_topic,
                        "truth": true_label
                    })

    print("\n" + "="*60)
    print("📊 CLASSIFICATION REPORT")
    print("="*60)
    
    print("\n📂 Category Distribution:")
    for topic, count in distribution.items():
        print(f"   - {topic:<30}: {count}")

    if total_processed > 0 and ground_truth:
        accuracy = (correct_count / total_processed) * 100
        print(f"\n✅ Total Verified: {total_processed}")
        print(f"🎯 Accuracy:       {accuracy:.2f}% ({correct_count}/{total_processed})")
        
        if errors:
            print("\n❌ Incorrect Classifications:")
            print(f"{'Filename':<50} | {'Predicted':<25} | {'Truth'}")
            print("-" * 90)
            for err in errors:
                print(f"{err['file'][:50]:<50} | {err['predicted']:<25} | {err['truth']}")
        else:
            print("\n🎉 Perfect Score! All classifications match ground truth.")
    else:
        print("\nℹ️  No ground truth matching performed.")
    print("="*60)

def index_images(folder_path):
    img_files = get_files_in_directory(folder_path, [".jpg", ".jpeg", ".png", ".webp"])
    print(f"🖼️ Found {len(img_files)} images. Processing...")

    success_count = 0
    for img_path in tqdm(img_files, desc="Indexing Images"):
        filename = os.path.basename(img_path)
        
        image = load_image(img_path)
        if image:
            embedding = model_manager.get_image_embedding(image)
            new_path = copy_file(img_path, IMAGES_DIR, filename)
            img_id = str(uuid.uuid4())
            metadata = {
                "filename": filename,
                "original_name": filename,
                "path": new_path
            }
            db.add_image(img_id, embedding, metadata)
            success_count += 1
            
    print(f"✅ Copied and Indexed {success_count}/{len(img_files)} images.")

def search_paper(query):
    print(f"\n🔎 Searching papers for: '\033[94m{query}\033[0m'")
    query_vec = model_manager.get_text_embedding(query)
    results = db.search_papers(query_vec)
    
    if not results['ids'][0]:
        print("❌ No results found.")
        return

    print("\n" + "="*80)
    print(f"{'SCORE':<8} | {'TOPIC':<25} | {'DOCUMENT NAME'}")
    print("-" * 80)
    
    seen_names = set()
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i]
        score = 1 - dist
        
        name = meta['filename']
        if name in seen_names: continue
        seen_names.add(name)

        snippet = meta['content_snippet'].replace('\n', ' ')
        print(f"{score:.4f}   | {meta['topic']:<25} | \033[1m{name}\033[0m")
        print(f"         > Snippet: {snippet[:120]}...\n")

def search_image(query):
    print(f"\n🔎 Searching images for: '\033[94m{query}\033[0m'")
    query_vec = model_manager.get_clip_text_embedding(query)
    results = db.search_images(query_vec)

    if not results['ids'][0]:
        print("❌ No results found.")
        return

    print("\n" + "="*60)
    print(f"{'SCORE':<8} | {'IMAGE NAME'}")
    print("-" * 60)
    
    seen_names = set()
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i]
        score = 1 - dist
        
        name = meta['filename']
        if name in seen_names: continue
        seen_names.add(name)
        
        print(f"{score:.4f}   | \033[1m{name}\033[0m")
        print(f"         > Path: {meta['path']}\n")

def main():
    parser = argparse.ArgumentParser(description="Local Multimodal AI Agent")
    subparsers = parser.add_subparsers(dest="command")

    p_add = subparsers.add_parser("add_paper")
    p_add.add_argument("path", type=str)
    p_add.add_argument("--topics", type=str, required=True)

    p_batch = subparsers.add_parser("organize_folder")
    p_batch.add_argument("path", type=str)
    p_batch.add_argument("--topics", type=str, required=True)

    p_img = subparsers.add_parser("index_images")
    p_img.add_argument("path", type=str)

    p_search = subparsers.add_parser("search_paper")
    p_search.add_argument("query", type=str)

    p_s_img = subparsers.add_parser("search_image")
    p_s_img.add_argument("query", type=str)

    args = parser.parse_args()

    if args.command == "add_paper":
        add_paper(args.path, args.topics)
    elif args.command == "organize_folder":
        batch_organize(args.path, args.topics)
    elif args.command == "index_images":
        index_images(args.path)
    elif args.command == "search_paper":
        search_paper(args.query)
    elif args.command == "search_image":
        search_image(args.query)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()