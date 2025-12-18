import os
import zipfile
import shutil
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_DIR = "Raw_Zips"
DEST_DIR = "Friends_Dataset_Master"
VALID_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# ---------------------

def setup_directories():
    """Creates the Master Directory and subfolders for each emotion."""
    if os.path.exists(DEST_DIR):
        print(f"⚠️  Warning: '{DEST_DIR}' already exists.")
        resp = input("    Delete and rebuild it? (y/n): ")
        if resp.lower() == 'y':
            shutil.rmtree(DEST_DIR)
        else:
            print("    Exiting to avoid data loss.")
            exit()
    
    os.makedirs(DEST_DIR)
    for emotion in VALID_EMOTIONS:
        os.makedirs(os.path.join(DEST_DIR, emotion))
    print(f"✅ Created empty Master structure in '{DEST_DIR}'")

def clean_filename(filename):
    """Removes weird characters from filenames."""
    return "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))

def process_zips():
    """Iterates through all zips and extracts valid images to Master."""
    zip_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.zip')]
    
    if not zip_files:
        print(f"❌ No .zip files found in '{SOURCE_DIR}'. Please check the folder.")
        return

    print(f"📦 Found {len(zip_files)} zip files. Starting processing...\n")
    
    total_images_copied = 0

    for zip_name in zip_files:
        zip_path = os.path.join(SOURCE_DIR, zip_name)
        # Use the filename (without .zip) as the unique prefix for this friend
        friend_id = clean_filename(Path(zip_name).stem) 
        
        print(f"   ➡️  Processing: {zip_name} (ID: {friend_id})")

        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                # List all files inside the zip
                for file_info in z.infolist():
                    if file_info.is_dir():
                        continue
                    
                    # Check if the file is an image
                    if not file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue

                    # Analyze the path inside the zip to find the emotion
                    # Example path: "folder/happy/me.jpg" or "happy/img1.jpg"
                    parts = file_info.filename.lower().split('/')
                    
                    detected_emotion = None
                    for part in parts:
                        if part in VALID_EMOTIONS:
                            detected_emotion = part
                            break
                    
                    if detected_emotion:
                        # Extract the file to memory
                        source_bytes = z.read(file_info)
                        
                        # Generate the new unique Master name
                        # Format: Master/happy/FriendID_originalName.jpg
                        original_filename = clean_filename(Path(file_info.filename).name)
                        new_filename = f"{friend_id}_{original_filename}"
                        target_path = os.path.join(DEST_DIR, detected_emotion, new_filename)
                        
                        # Write to Master
                        with open(target_path, 'wb') as f:
                            f.write(source_bytes)
                        
                        total_images_copied += 1
                    # else: skip files that aren't in a valid emotion folder

        except zipfile.BadZipFile:
            print(f"   ❌ ERROR: {zip_name} is corrupted. Skipping.")

    print(f"\n✨ DONE! Processed {len(zip_files)} zips.")
    print(f"📊 Total Images in Master: {total_images_copied}")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: Folder '{SOURCE_DIR}' not found. Please create it and put your zips inside.")
    else:
        setup_directories()
        process_zips()
