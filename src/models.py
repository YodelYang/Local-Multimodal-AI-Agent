import torch
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from src.config import DEVICE, TEXT_EMBEDDING_MODEL, CLIP_MODEL_ID, LLM_MODEL_ID, LABEL_DEFINITIONS

KEYWORD_RULES = {
    "Gravitational Waves": "Physics",
    "Superconformal": "Physics", 
    "AdS/CFT": "Physics",
    "Dark Matter": "Physics",
    "General Relativity": "Physics",
    "Quantum": "Physics",
    "Proximal Policy": "Reinforcement Learning",
    "PPO": "Reinforcement Learning",
    "Rainbow": "Reinforcement Learning",
    "Deep Reinforcement Learning": "Reinforcement Learning",
    "Q-learning": "Reinforcement Learning",
    "DQN": "Reinforcement Learning",
    "AlphaGo": "Reinforcement Learning",
    "Trust Region Policy": "Reinforcement Learning",
    "TRPO": "Reinforcement Learning",
    "Decision Transformer": "Reinforcement Learning",

    "Imagen": "AI Generated Content", 
    "DiT": "AI Generated Content", 
    "Scalable Diffusion": "AI Generated Content",
    "Diffusion Models": "AI Generated Content",
    "Stable Diffusion": "AI Generated Content",
    "DALL-E": "AI Generated Content",
    "Text-to-Image": "AI Generated Content",
    "Generative Adversarial": "AI Generated Content",
    "GAN": "AI Generated Content",
    "DreamBooth": "AI Generated Content",
    "ControlNet": "AI Generated Content",

    "CLIP": "Computer Vision",
    "YOLO": "Computer Vision",
    "ResNet": "Computer Vision",
    "Vision Transformer": "Computer Vision",
    "ViT": "Computer Vision",
    "Segment Anything": "Computer Vision",
    "Masked Autoencoders": "Computer Vision", 

    "Brain-Computer": "Neuroscience",
    "Spiking Neural Networks": "Neuroscience",
    "scRNA-seq": "Biology",
    "bioinformatics": "Biology",
    "Stock Market": "Finance",
    "Limit Order": "Finance",
    "GPT-3": "Large Language Models",
    "GPT-4": "Large Language Models",
    "LLaMA": "Large Language Models",
    "Chain-of-Thought": "Large Language Models",
    "InstructGPT": "Large Language Models",
    "RAG": "Large Language Models",
    "Retrieval-Augmented": "Large Language Models",
    "BERT": "Natural Language Processing",
    "Attention Is All You Need": "Natural Language Processing",
    "YOLO": "Computer Vision",
    "ResNet": "Computer Vision",
    "Vision Transformer": "Computer Vision",
    "ViT": "Computer Vision",
    "Segment Anything": "Computer Vision"
}

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.init_models()
        return cls._instance

    def init_models(self):
        print(f"⏳ Loading Models on {DEVICE}...")

        try:
            self.text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL, device=DEVICE)
        except Exception as e:
            print(f"⚠️ Error loading text model: {e}")

        try:
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        except Exception as e:
            print(f"⚠️ Error loading vision model: {e}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
            self.llm = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID,
                torch_dtype=torch.float16,
                device_map=DEVICE,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"⚠️ Error loading LLM: {e}")
        
        print("✅ All Models Loaded Successfully.")

    def get_text_embedding(self, text):
        return self.text_model.encode(text, convert_to_numpy=True).tolist()

    def get_clip_text_embedding(self, text):
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten().tolist()

    def get_image_embedding(self, image):
        inputs = self.clip_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten().tolist()

    def classify_paper_with_llm(self, text, filename, candidate_labels):
        for keyword, category in KEYWORD_RULES.items():
            if keyword.lower() in filename.lower():
                if category in candidate_labels:
                    return category

        text_snippet = text[:500].lower()
        for keyword, category in KEYWORD_RULES.items():
            if keyword.lower() in text_snippet:
                if category in candidate_labels:
                    return category

        definitions_text = ""
        for label in candidate_labels:
            desc = LABEL_DEFINITIONS.get(label, "")
            definitions_text += f"- {label}: {desc}\n"

        abstract_snippet = text[:1500]

        prompt = f"""You are an expert academic librarian. Classify the following research paper into exactly one of these categories:
{', '.join(candidate_labels)}

Definitions:
{definitions_text}

Paper Filename: "{filename}"
Paper Abstract:
"{abstract_snippet}"

Instructions:
Select the ONE best category. Return ONLY the category name. Do not output any explanation.

Category:"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_ids = self.llm.generate(
                **model_inputs,
                max_new_tokens=32,
                temperature=0.1,   
                do_sample=False
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        clean_response = response.replace(".", "").strip()

        sorted_candidates = sorted(candidate_labels, key=len, reverse=True)
        for label in sorted_candidates:
            if label.lower() in clean_response.lower():
                return label

        print(f"⚠️ LLM Output unclear: '{response}'. Defaulting to {candidate_labels[0]}")
        return candidate_labels[0]