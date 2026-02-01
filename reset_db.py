import os
from search import StructuralSearchEngine

# 1. Define the files to delete
files_to_delete = ["jewelry.index", "filenames.pkl", "hashes.pkl"]

print("🧹 Cleaning up old database files...")
for f in files_to_delete:
    if os.path.exists(f):
        os.remove(f)
        print(f"   - Deleted {f}")
    else:
        print(f"   - {f} was already missing")

# 2. Force a Rebuild
print("\n🏗️  Rebuilding Database (this may take a moment)...")
try:
    # Initialize engine (it will see files are missing)
    engine = StructuralSearchEngine()
    # Force build
    engine.build_index("catalog")
    print("\n✅ SUCCESS! 'hashes.pkl' has been created.")
    print("🚀 You can now run 'streamlit run app.py'")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Make sure your 'search.py' is updated with the code from the previous step!")