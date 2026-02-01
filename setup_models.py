import os
import time

print("⏳ Starting System Check & Model Download...")

# 1. Check Internet & Transformers Download
try:
    print("\n[1/3] Downloading DINOv2 Model (This is big, please wait)...")
    from transformers import AutoImageProcessor, AutoModel
    
    # This forces the download and shows if it hangs
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    print("   ✅ Processor downloaded.")
    
    model = AutoModel.from_pretrained('facebook/dinov2-small')
    print("   ✅ Model downloaded.")

except Exception as e:
    print(f"\n❌ Error downloading DINOv2: {e}")
    print("👉 Check your internet connection.")

# 2. Check YOLO Download
try:
    print("\n[2/3] Downloading YOLO Model...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("   ✅ YOLO downloaded.")

except Exception as e:
    print(f"\n❌ Error downloading YOLO: {e}")

# 3. Check Catalog Size
print("\n[3/3] Checking Catalog Folder...")
if os.path.exists("catalog"):
    num_files = len(os.listdir("catalog"))
    print(f"   ℹ️  You have {num_files} images in your 'catalog' folder.")
    if num_files > 50:
        print("   ⚠️  WARNING: You have many images!")
        print("       The first time you run the app, it will take ~1 second per image to index.")
        print(f"       Estimated wait time: {num_files / 60:.1f} minutes.")
else:
    print("   ℹ️  Catalog folder is empty (Good for first run).")

print("\n✨ Setup Complete! You can now run 'streamlit run app.py'")