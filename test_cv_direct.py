import base64
from cv_engine import scan_frame
import os
from dotenv import load_dotenv
load_dotenv()

image_path = r'C:\Users\saipr\Downloads\Sage Green Ethnic Kurta with Palazzo _ Minimal Elegant Indian Outfit.jpg'

try:
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    print("Testing scan_frame...")
    res = scan_frame(img_b64)
    print("\n--- TEST RESULTS ---")
    
    print("Parsed Items:")
    for i, item in enumerate(res.get("items", [])):
        print(f"Item {i+1}:")
        print(f"  Label: {item.get('label')}")
        print(f"  Description: {item.get('description')}")
        
    has_annotated = "annotated_frame_b64" in res and bool(res["annotated_frame_b64"])
    print(f"\nAnnotated Frame Present & Non-Empty: {has_annotated}")
    
except Exception as e:
    print(f"Error: {e}")
