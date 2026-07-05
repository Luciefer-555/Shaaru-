import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch, base64, re, time
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoModel

print('Loading LocateAnything-3B...')
t0 = time.time()
processor = AutoProcessor.from_pretrained(
    'nvidia/LocateAnything-3B', trust_remote_code=True)
model = AutoModel.from_pretrained(
    'nvidia/LocateAnything-3B',
    torch_dtype=torch.bfloat16,
    device_map='cpu',
    trust_remote_code=True
)
model.eval()
print(f'Model loaded in {round(time.time()-t0, 1)}s')

# Load Bershka image
with open(r'C:\Users\saipr\Downloads\Bershka.jpg', 'rb') as f:
    img_bytes = f.read()
image = Image.open(BytesIO(img_bytes)).convert('RGB')
w, h = image.size
print(f'Image size: {w}x{h}')

def locate(prompt_text):
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': prompt_text}
        ]
    }]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors='pt')
    inputs = inputs.to(model.device)
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    
    t = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128,
            do_sample=False, use_cache=True,
            tokenizer=processor.tokenizer
        )
    latency = round(time.time()-t, 2)
    
    text = out
    
    # Parse bbox coords
    bboxes = []
    for match in re.finditer(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>', text):
        x1,y1,x2,y2 = [int(g)/1000 for g in match.groups()]
        bboxes.append({
            'x1_px': int(x1*w), 'y1_px': int(y1*h),
            'x2_px': int(x2*w), 'y2_px': int(y2*h),
            'normalized': [x1,y1,x2,y2]
        })
    bbox = bboxes if len(bboxes) > 1 else (bboxes[0] if bboxes else None)
    
    clean = re.sub(r'<[^>]+>', '', text).strip()
    return {'answer': clean, 'bbox': bbox, 'latency_s': latency}

print()
print('=== TEST 1: Detect all clothing items ===')
r = locate('Detect all clothing and fashion items in this image.')
print('Answer:', r['answer'])
print('Latency:', r['latency_s'], 's')

print()
print('=== TEST 2: Locate the cream sweater ===')
r = locate('Locate the cream or ivory knit sweater in this image.')
print('Answer:', r['answer'])
print('BBox:', r['bbox'])
print('Latency:', r['latency_s'], 's')

print()
print('=== TEST 3: Locate the striped sweater ===')
r = locate('Locate the black and white striped sweater.')
print('Answer:', r['answer'])
print('BBox:', r['bbox'])
print('Latency:', r['latency_s'], 's')

print()
print('=== TEST 4: Locate the denim ===')
r = locate('Locate the light wash denim jeans or skirt.')
print('Answer:', r['answer'])
print('BBox:', r['bbox'])
print('Latency:', r['latency_s'], 's')

print()
print('=== TEST 5: Locate the boots ===')
r = locate('Locate the black leather boots.')
print('Answer:', r['answer'])
print('BBox:', r['bbox'])
print('Latency:', r['latency_s'], 's')

print()
print('=== TEST 6: Point query — what is at top-left? ===')
r = locate(
    f'What clothing item is at pixel position '
    f'({int(w*0.1)}, {int(h*0.1)}) in this image?'
)
print('Answer:', r['answer'])
print('BBox:', r['bbox'])
print('Latency:', r['latency_s'], 's')

print()
print('=== TEST 7: Point query — what is in foreground center? ===')
r = locate(
    f'What clothing item is at pixel position '
    f'({int(w*0.5)}, {int(h*0.4)}) in this image?'
)
print('Answer:', r['answer'])
print('BBox:', r['bbox'])
print('Latency:', r['latency_s'], 's')

import subprocess
vram = subprocess.run(
    ['nvidia-smi', '--query-gpu=memory.used,memory.total',
     '--format=csv,noheader'],
    capture_output=True, text=True
)
print()
print('VRAM:', vram.stdout.strip())
