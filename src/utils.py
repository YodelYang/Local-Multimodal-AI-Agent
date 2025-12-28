import fitz  # PyMuPDF
from PIL import Image
import os
import shutil
import re

def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_path, max_pages=5):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text += page.get_text()
        
        cleaned_text = clean_text(text)
        if not cleaned_text or len(cleaned_text) < 50:
            return None
        return cleaned_text
    except Exception as e:
        print(f"❌ Error reading PDF {pdf_path}: {e}")
        return None

def load_image(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None

def copy_file(src_path, target_folder, filename):
    """复制文件到目标文件夹 (保留原文件)"""
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    dst_path = os.path.join(target_folder, filename)
    shutil.copy2(src_path, dst_path)
    return dst_path

def get_files_in_directory(directory, extensions):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(tuple(extensions)):
                files.append(os.path.join(root, filename))
    return files