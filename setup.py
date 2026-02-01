import os
import subprocess
import sys
from PIL import Image, ImageDraw

print("--- STARTING AUTO-SETUP ---")

# 1. Install Libraries
print("Step 1: Checking libraries...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "opencv-python", "numpy", "pillow", "ultralytics", "scikit-learn", "transformers", "torch", "streamlit-cropper", "streamlit-image-coordinates"])

# 2. Fix Streamlit Email Prompt
print("Step 2: Disabling Streamlit Email Prompt...")
folder = ".streamlit"
if not os.path.exists(folder): os.makedirs(folder)
with open(os.path.join(folder, "config.toml"), "w") as f:
    f.write("[browser]\ngatherUsageStats = false\n")

# 3. Create Gemstones
print("Step 3: Creating Gemstone Images...")
def make_stone(name, color):
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((10, 10, 90, 90), fill=color, outline="white")
    img.save(f"stone_{name}.png")

make_stone("ruby", (255, 0, 0, 200))
make_stone("sapphire", (0, 0, 255, 200))
make_stone("diamond", (200, 240, 255, 200))

# 4. Create Catalog Folder
if not os.path.exists("catalog"):
    os.makedirs("catalog")

print("\n✅ SETUP COMPLETE. You can now run 'start.bat'")