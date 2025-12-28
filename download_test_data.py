import os
import requests
import time
import json  # 新增

# =================配置区域=================
BASE_DIR = "/amax/home/dywang/course_work/Multimodal/LocalAI_Agent"
DOWNLOAD_DIR = os.path.join(BASE_DIR, "test_downloads")
PDF_DIR = os.path.join(DOWNLOAD_DIR, "raw_pdfs")
IMG_DIR = os.path.join(DOWNLOAD_DIR, "raw_images")
GT_FILE = os.path.join(DOWNLOAD_DIR, "ground_truth.json") # 新增：真实标签文件路径

# ================= 论文列表 (已替换 4 篇易错论文) =================
PAPERS = [
    # ==================== Reinforcement Learning (7篇) ====================
    ("Playing Atari with Deep Reinforcement Learning (DQN)", "1312.5602", "Reinforcement Learning"),
    ("Proximal Policy Optimization Algorithms (PPO)", "1707.06347", "Reinforcement Learning"),
    ("Mastering the Game of Go without Human Knowledge (AlphaGo Zero)", "1710.06542", "Reinforcement Learning"),
    ("Continuous Control with Deep Reinforcement Learning (DDPG)", "1509.02971", "Reinforcement Learning"),
    ("Trust Region Policy Optimization (TRPO)", "1502.05477", "Reinforcement Learning"),
    ("Decision Transformer Reinforcement Learning via Sequence Modeling", "2106.01345", "Reinforcement Learning"),
    ("Rainbow Combining Improvements in Deep Reinforcement Learning", "1710.02298", "Reinforcement Learning"),

    # ==================== Natural Language Processing (2篇) ====================
    # 注意：Attention 和 BERT 属于 NLP 基础，不属于 LLM
    ("Attention Is All You Need (Transformer)", "1706.03762", "Natural Language Processing"),
    ("BERT Pre-training of Deep Bidirectional Transformers", "1810.04805", "Natural Language Processing"),

    # ==================== Large Language Models (7篇) ====================
    ("Language Models are Few-Shot Learners (GPT-3)", "2005.14165", "Large Language Models"),
    ("LLaMA Open and Efficient Foundation Language Models", "2302.13971", "Large Language Models"),
    ("Chain-of-Thought Prompting Elicits Reasoning", "2201.11903", "Large Language Models"),
    ("Training language models to follow instructions (InstructGPT)", "2203.02155", "Large Language Models"),
    ("LoRA Low-Rank Adaptation of Large Language Models", "2106.09685", "Large Language Models"),
    ("Retrieval-Augmented Generation for Knowledge-Intensive NLP (RAG)", "2005.11401", "Large Language Models"),
    ("Visual Instruction Tuning (LLaVA)", "2304.08485", "Large Language Models"), # 多模态 LLM

    # ==================== Computer Vision (7篇) ====================
    ("Deep Residual Learning for Image Recognition (ResNet)", "1512.03385", "Computer Vision"),
    ("An Image is Worth 16x16 Words (ViT)", "2010.11929", "Computer Vision"),
    ("You Only Look Once Unified Real-Time Object Detection (YOLO)", "1506.02640", "Computer Vision"),
    ("Learning Transferable Visual Models From Natural Language Supervision (CLIP)", "2103.00020", "Computer Vision"),
    ("Masked Autoencoders Are Scalable Vision Learners (MAE)", "2111.06377", "Computer Vision"),
    ("Segment Anything (SAM)", "2304.02643", "Computer Vision"),
    ("Swin Transformer Hierarchical Vision Transformer using Shifted Windows", "2103.14030", "Computer Vision"),

    # ==================== AI Generated Content (8篇) ====================
    ("Generative Adversarial Networks (GAN)", "1406.2661", "AI Generated Content"),
    ("Denoising Diffusion Probabilistic Models (DDPM)", "2006.11239", "AI Generated Content"),
    ("High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)", "2112.10752", "AI Generated Content"),
    ("Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)", "2302.05543", "AI Generated Content"),
    ("Scalable Diffusion Models with Transformers (DiT)", "2212.09748", "AI Generated Content"),
    ("Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen)", "2205.11487", "AI Generated Content"),
    ("DreamBooth Fine Tuning Text-to-Image Diffusion Models for Subject Customization", "2208.12242", "AI Generated Content"),
    
    # ==================== Physics (3篇) ====================
    ("Observation of Gravitational Waves", "1602.03837", "Physics"),
    ("The Large N Limit of Superconformal Field Theories", "hep-th/9711200", "Physics"),
    ("Dark Matter Candidates", "1006.2753", "Physics"),

    # ==================== Biology (1篇) ====================
    ("Deep learning for bioinformatics", "1903.00342", "Biology"),

    # ==================== Finance (1篇) ====================
    ("Stock Market Prediction via Deep Learning Techniques A Survey", "2212.12717", "Finance"),

    # ==================== Neuroscience (1篇) ====================
    ("Surrogate Gradient Learning in Spiking Neural Networks", "1901.09948", "Neuroscience"),
]

# 图片列表保持不变
IMAGES = [
    ("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800", "animal_cat_01.jpg"),
    ("https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=800", "animal_cat_02.jpg"),
    ("https://images.unsplash.com/photo-1537151608828-ea2b11777ee8?w=800", "animal_dog_01.jpg"),
    ("https://images.unsplash.com/photo-1552053831-71594a27632d?w=800", "animal_dog_02.jpg"),
    ("https://images.unsplash.com/photo-1589656966895-2f33e7653819?w=800", "animal_polar_bear.jpg"),
    ("https://images.unsplash.com/photo-1540573133985-87b6da6d54a9?w=800", "animal_monkey.jpg"),
    ("https://images.unsplash.com/photo-1557008075-7f2c5efa4cfd?w=800", "animal_fox.jpg"),
    ("https://images.unsplash.com/photo-1546182990-dffeafbe841d?w=800", "animal_lion.jpg"),
    ("https://images.unsplash.com/photo-1456926631375-92c8ce872def?w=800", "animal_leopard.jpg"),
    ("https://images.unsplash.com/photo-1535591273668-578e31182c4f?w=800", "animal_fish.jpg"),
    ("https://images.unsplash.com/photo-1501706362039-c06b2d715385?w=800", "animal_zebra.jpg"),
    ("https://images.unsplash.com/photo-1437622368342-7a3d73a34c8f?w=800", "animal_turtle.jpg"),
    ("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800", "nature_beach.jpg"),
    ("https://images.unsplash.com/photo-1534088568595-a066f410bcda?w=800", "nature_clouds_01.jpg"),
    ("https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800", "nature_mountains_01.jpg"),
    ("https://images.unsplash.com/photo-1508881598441-324f3974994b?w=800", "nature_islands.jpg"),
    ("https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=800", "nature_river.jpg"),
    ("https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=800", "nature_sunset_field.jpg"),
    ("https://images.unsplash.com/photo-1504608524841-42fe6f032b4b?w=800", "nature_clouds_02.jpg"),
    ("https://images.unsplash.com/photo-1531306728370-e2ebd9d7bb99?w=800", "nature_galaxy.jpg"),
    ("https://images.unsplash.com/photo-1505765050516-f72dcac9c60e?w=800", "nature_snow_mountains.jpg"),
    ("https://images.unsplash.com/photo-1508739773434-c26b3d09e071?w=800", "nature_mountains_02.jpg"),
    ("https://images.unsplash.com/photo-1506422748879-887454f9cdff?w=800", "city_skyline.jpg"),
    ("https://images.unsplash.com/photo-1496871455396-14e56815f1f4?w=800", "city_street.jpg"),
    ("https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=800", "city_tower_bridge.jpg"),
    ("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=800", "city_chicago_aerial.jpg"),
    ("https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=800", "city_street_newyork.jpg"),
    ("https://images.unsplash.com/photo-1554232456-8727aae0cfa4?w=800", "city_office_interior.jpg"),
    ("https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=800", "city_modern_house.jpg"),
    ("https://images.unsplash.com/photo-1518780664697-55e3ad937233?w=800", "city_cottage_house.jpg"),
    ("https://images.unsplash.com/photo-1479839672679-a46483c0e7c8?w=800", "city_buildings.jpg"),
    ("https://images.unsplash.com/photo-1518843875459-f738682238a6?w=800", "food_vegetables.jpg"),
    ("https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=800", "food_pizza.jpg"),
    ("https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=800", "food_coffee.jpg"),
    ("https://images.unsplash.com/photo-1621996346565-e3dbc646d9a9?w=800", "food_pasta.jpg"),
    ("https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=800", "food_burger.jpg"),
    ("https://images.unsplash.com/photo-1579954115545-a95591f28bfc?w=800", "food_milkshake.jpg"),
    ("https://images.unsplash.com/photo-1484723091739-30a097e8f929?w=800", "food_french_toast.jpg"),
    ("https://images.unsplash.com/photo-1563805042-7684c019e1cb?w=800", "food_icecream.jpg"),
    ("https://images.unsplash.com/photo-1606787366850-de6330128bfc?w=800", "food_feast.jpg"),
    ("https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=800", "food_healthy_salad.jpg"),
    ("https://images.unsplash.com/photo-1550989460-0adf9ea622e2?w=800", "food_grocery_store.jpg"),
    ("https://images.unsplash.com/photo-1517976487492-5750f3195933?w=800", "tech_rocket_launch.jpg"),
    ("https://images.unsplash.com/photo-1517694712202-14dd9538aa97?w=800", "tech_laptop.jpg"),
    ("https://images.unsplash.com/photo-1527443224154-c4a3942d3acf?w=800", "tech_imac.jpg"),
    ("https://images.unsplash.com/photo-1550009158-9ebf69173e03?w=800", "tech_keyboard.jpg"),
    ("https://images.unsplash.com/photo-1526498460520-4c246339dccb?w=800", "tech_smartphone.jpg"),
    ("https://images.unsplash.com/photo-1581091226033-d5c48150dbaa?w=800", "tech_laboratory_01.jpg"),
    ("https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?w=800", "tech_laboratory_02.jpg"),
    ("https://images.unsplash.com/photo-1614632537190-23e4146777db?w=800", "sport_soccer_ball.jpg"),
    ("https://images.unsplash.com/photo-1510915361894-db8b60106cb1?w=800", "people_musician.jpg"),
    ("https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=800", "people_group.jpg"),
    ("https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=800", "people_portrait_woman.jpg"),
    ("https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=800", "people_portrait_man.jpg"),
    ("https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=800", "people_portrait_senior.jpg"),
    ("https://images.unsplash.com/photo-1541534741688-6078c6bfb5c5?w=800", "sport_gym_woman.jpg"),
    ("https://images.unsplash.com/photo-1540497077202-7c8a3999166f?w=800", "sport_gym_workout.jpg"),
    ("https://images.unsplash.com/photo-1560272564-c83b66b1ad12?w=800", "people_soccer.jpg"),
    ("https://images.unsplash.com/photo-1491841550275-ad7854e35ca6?w=800", "people_reading_book.jpg"),
    ("https://images.unsplash.com/photo-1530549387789-4c1017266635?w=800", "sport_swimming.jpg"),
    ("https://images.unsplash.com/photo-1517649763962-0c623066013b?w=800", "sport_cycling.jpg"),
    ("https://images.unsplash.com/photo-1581094794329-c8112a89af12?w=800", "ocr_code_screen.jpg"),
    ("https://images.unsplash.com/photo-1528109966604-5a6a4a964e8d?w=800", "ocr_video_editing.jpg"),
    ("https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=800", "ocr_hacker_binary.jpg"),
    ("https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=800", "object_book.jpg"),
    ("https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=800", "object_headphones.jpg"),
    ("https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=800", "object_red_sneakers.jpg"),
    ("https://images.unsplash.com/photo-1581235720704-06d3acfcb36f?w=800", "object_car.jpg"),
    ("https://images.unsplash.com/photo-1611162617474-5b21e879e113?w=800", "ocr_icon.jpg"),
    ("https://images.unsplash.com/photo-1516979187457-637abb4f9353?w=800", "object_books.jpg"),
    ("https://images.unsplash.com/photo-1586769852044-692d6e3703f0?w=800", "object_typewriter.jpg"),
    ("https://images.unsplash.com/photo-1629198688000-71f23e745b6e?w=800", "object_cosmetic.jpg"),
]

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
    for d in [DOWNLOAD_DIR, PDF_DIR, IMG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    print(f"📂 Setup directories in: {DOWNLOAD_DIR}")
    print("-" * 50)

    # 1. 准备 Ground Truth 字典
    ground_truth = {}

    print(f"📄 Starting PDF download ({len(PAPERS)} papers)...")
    pdf_count = 0
    for title, arxiv_id, category in PAPERS:
        # 处理 arxiv URL
        if "arxiv" in arxiv_id or "/" in arxiv_id:
             pass
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        # 清理文件名
        safe_title = "".join([c if c.isalnum() or c in " .-_" else "" for c in title])
        filename = f"{safe_title}.pdf"
        filepath = os.path.join(PDF_DIR, filename)
        
        # 记录 Ground Truth (不管文件是否已存在，都要记录)
        ground_truth[filename] = category

        if os.path.exists(filepath):
            print(f"⏩ Skipping (Exists): {safe_title}")
            pdf_count += 1
            continue
            
        print(f"⬇️  Downloading [{category}]: {safe_title}...")
        if download_file(url, filepath):
            pdf_count += 1
            time.sleep(1)
    
    # 2. 保存 Ground Truth 到文件
    with open(GT_FILE, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Ground Truth saved to: {GT_FILE}")

    print("-" * 50)

    # 图片下载逻辑不变
    print(f"🖼️  Starting Image download ({len(IMAGES)} images)...")
    img_count = 0
    for url, save_name in IMAGES:
        filename = os.path.join(IMG_DIR, save_name)
        if os.path.exists(filename):
            print(f"⏩ Skipping (Exists): {save_name}")
            img_count += 1
            continue
            
        print(f"⬇️  Downloading Image: {save_name}...")
        if download_file(url, filename):
            img_count += 1
            time.sleep(0.5)

    print("\n" + "="*50)
    print(f"✅ Download Summary:")
    print(f"   - PDFs: {pdf_count}/{len(PAPERS)}")
    print(f"   - Images: {img_count}/{len(IMAGES)}")
    print(f"   - Location: {DOWNLOAD_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()