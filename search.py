import os
import torch
import cv2
import faiss
import numpy as np
import pickle
from PIL import Image
from rembg import remove
from transformers import AutoImageProcessor, AutoModel

# --- SAFE IMPORT ---
try:
    import imagehash
    HASHING_AVAILABLE = True
except ImportError:
    HASHING_AVAILABLE = False
    print("[WARNING] 'imagehash' library not found. 100% Exact Match feature will be disabled.")

class StructuralSearchEngine:
    def __init__(self, index_path="jewelry.index", map_path="filenames.pkl", hash_path="hashes.pkl"):
        print("[INIT] Loading AI Models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load DINOv2
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        
        # --- THE FIX FOR "SDPA" ERROR IS HERE ---
        # We force 'eager' mode so we can extract attention maps
        self.model = AutoModel.from_pretrained(
            'facebook/dinov2-small',
            attn_implementation="eager" 
        ).to(self.device)
        # ----------------------------------------
        
        self.model.config.output_attentions = True
        
        self.index_path = index_path
        self.map_path = map_path
        self.hash_path = hash_path
        
        self.filenames = []
        self.image_hashes = {} 
        self.index = None

        # Load Database
        if os.path.exists(index_path) and os.path.exists(map_path):
            self.index = faiss.read_index(index_path)
            with open(map_path, "rb") as f:
                self.filenames = pickle.load(f)
            
            if HASHING_AVAILABLE and os.path.exists(hash_path):
                with open(hash_path, "rb") as f:
                    self.image_hashes = pickle.load(f)
            print(f"[SUCCESS] Loaded Database: {len(self.filenames)} rings.")

    def preprocess_structural(self, image_input):
        if isinstance(image_input, str):
            img_pil = Image.open(image_input).convert("RGBA")
        else:
            img_pil = image_input.convert("RGBA")

        img_no_bg = remove(img_pil)
        bbox = img_no_bg.getbbox()
        if bbox:
            img_final = img_no_bg.crop(bbox)
        else:
            img_final = img_no_bg

        img_np = np.array(img_final)
        alpha = img_np[:, :, 3]
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        final_struct = cv2.bitwise_and(dilated, dilated, mask=alpha)
        
        return Image.fromarray(cv2.cvtColor(final_struct, cv2.COLOR_GRAY2RGB))

    def get_embedding(self, image_input):
        struct_img = self.preprocess_structural(image_input)
        inputs = self.processor(images=struct_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        faiss.normalize_L2(emb)
        return emb.astype('float32')

    def calculate_human_score(self, raw_score):
        score = 1 / (1 + np.exp(-15 * (raw_score - 0.72)))
        return score

    def build_index(self, catalog_folder):
        print(f"[INFO] Indexing catalog from: {catalog_folder}...")
        embeddings = []
        self.filenames = []
        self.image_hashes = {} 

        valid_exts = ('.jpg', '.jpeg', '.png')
        files = [f for f in os.listdir(catalog_folder) if f.lower().endswith(valid_exts)]
        
        for i, filename in enumerate(files):
            path = os.path.join(catalog_folder, filename)
            try:
                img = Image.open(path).convert("RGBA")
                
                if HASHING_AVAILABLE:
                    h = str(imagehash.phash(img))
                    self.image_hashes[h] = filename
                
                emb = self.get_embedding(img)
                embeddings.append(emb)
                self.filenames.append(filename)
                
                if i % 10 == 0: print(f"Processed {i}/{len(files)}")
            except Exception as e:
                print(f"Skipping {filename}: {e}")

        if not embeddings: return
        
        dim = embeddings[0].shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.vstack(embeddings))
        
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, "wb") as f:
            pickle.dump(self.filenames, f)
        
        if HASHING_AVAILABLE:
            with open(self.hash_path, "wb") as f:
                pickle.dump(self.image_hashes, f)
            
        print(f"[SUCCESS] Indexed {len(self.filenames)} items.")

    def find_similar(self, query_input, top_k=3):
        if self.index is None: return []
        
        if isinstance(query_input, str):
            img = Image.open(query_input).convert("RGBA")
        else:
            img = query_input.convert("RGBA")

        if HASHING_AVAILABLE and self.image_hashes:
            try:
                query_hash = imagehash.phash(img)
                for db_hash_str, filename in self.image_hashes.items():
                    db_hash = imagehash.hex_to_hash(db_hash_str)
                    if (query_hash - db_hash) <= 5: 
                        return [(filename, 1.0)] 
            except Exception:
                pass

        query_vec = self.get_embedding(img)
        scores, indices = self.index.search(query_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                human_score = self.calculate_human_score(float(score))
                results.append((self.filenames[idx], human_score))
        return results

    def explain_match(self, image_pil):
        """Visualizes attention map with Crash Guard."""
        print("DEBUG: Generating Heatmap...")
        
        try:
            inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
            
            # Run model
            outputs = self.model(**inputs, output_attentions=True)
            
            if outputs.attentions is None:
                print("⚠️ WARNING: Model returned None for attentions. Skipping heatmap.")
                return np.zeros((224, 224, 3), dtype=np.uint8)

            attentions = outputs.attentions[-1] 
            nh = attentions.shape[1]
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
            grid_size = int(np.sqrt(attentions.shape[-1]))
            avg_attn = attentions.mean(0).reshape(grid_size, grid_size).detach().cpu().numpy()
            heatmap = cv2.resize(avg_attn, (224, 224))
            heatmap = np.uint8(255 * (heatmap / heatmap.max()))
            return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        except Exception as e:
            print(f"⚠️ Error inside explain_match: {e}")
            return np.zeros((224, 224, 3), dtype=np.uint8)
        