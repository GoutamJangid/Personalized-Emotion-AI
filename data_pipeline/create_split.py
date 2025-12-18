import os
import shutil
import random

# --- CONFIGURATION ---
SOURCE_DIR = "Friends_Dataset_Master"
TARGET_DIR = "Friends_Dataset_Split"
SPLIT_RATIO = 0.8  # 80% Train, 20% Test
# ---------------------

def create_split():
    # 1. Wipe the old split if it exists (Fresh Start)
    if os.path.exists(TARGET_DIR):
        print(f"🧹 Cleaning up old '{TARGET_DIR}'...")
        shutil.rmtree(TARGET_DIR)
    
    # 2. Create Train/Test directories
    for split in ['train', 'test']:
        for emotion in os.listdir(SOURCE_DIR):
            os.makedirs(os.path.join(TARGET_DIR, split, emotion), exist_ok=True)

    print(f"🚀 Creating 80/20 Split from '{SOURCE_DIR}'...\n")
    
    total_images = 0
    emotion_counts = {}

    # 3. Iterate and Split
    for emotion in os.listdir(SOURCE_DIR):
        source_emotion_path = os.path.join(SOURCE_DIR, emotion)
        
        if not os.path.isdir(source_emotion_path):
            continue
            
        images = [f for f in os.listdir(source_emotion_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images) # Shuffle to ensure random split
        
        split_point = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_point]
        test_imgs = images[split_point:]
        
        # Copy to destinations
        for img in train_imgs:
            shutil.copy(os.path.join(source_emotion_path, img), os.path.join(TARGET_DIR, 'train', emotion, img))
            
        for img in test_imgs:
            shutil.copy(os.path.join(source_emotion_path, img), os.path.join(TARGET_DIR, 'test', emotion, img))
            
        emotion_counts[emotion] = len(images)
        total_images += len(images)
        print(f"   📂 {emotion.ljust(10)}: {len(train_imgs)} train + {len(test_imgs)} test = {len(images)} total")

    print(f"\n✅ DONE! Total Images: {total_images}")
    print(f"📂 Data is ready in: '{TARGET_DIR}'")

    # 4. Check for Dangerously Low Classes
    print("\n⚠️  Health Check:")
    for emotion, count in emotion_counts.items():
        if count < 50:
            print(f"   ❌ WARNING: '{emotion}' has only {count} images. Try to record more if possible!")
        elif count < 100:
            print(f"   ⚠️ CAUTION: '{emotion}' is low ({count}).")
        else:
            print(f"   ✅ '{emotion}' is healthy.")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: '{SOURCE_DIR}' not found. Run ingest_data.py first.")
    else:
        create_split()
