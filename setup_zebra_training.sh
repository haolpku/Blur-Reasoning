#!/bin/bash

SAMPLE_LIMIT=${1:-0}

cd /mnt/DataFlow/fyl

# 🔥 关键：不用单引号，让变量直接替换进去
python3 << EOF
import pandas as pd
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import re
import os

SAMPLE_LIMIT = ${SAMPLE_LIMIT}  # 直接用bash变量

print(f"🎯 限制样本数: {SAMPLE_LIMIT if SAMPLE_LIMIT > 0 else '全部'}\n")

ZEBRA_BASE = "/mnt/DataFlow/fyl/lyy/Zebra-CoT"
OUTPUT_DIR = "/mnt/DataFlow/fyl/lyy/LLaMA-Factory/data"
IMAGE_DIR = f"{OUTPUT_DIR}/zebra_images"
JSON_FILE = f"{OUTPUT_DIR}/zebra_all.json"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)

base_path = Path(ZEBRA_BASE)
all_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')])

all_dfs = []
total_rows = 0

for subset_dir in all_dirs:
    if SAMPLE_LIMIT > 0 and total_rows >= SAMPLE_LIMIT:
        break
    
    parquet_files = list(subset_dir.glob("*.parquet"))
    if not parquet_files:
        continue
    
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            df['subset'] = subset_dir.name
            
            # 立即截断
            if SAMPLE_LIMIT > 0:
                remaining = SAMPLE_LIMIT - total_rows
                if remaining <= 0:
                    break
                df = df.head(remaining)
            
            all_dfs.append(df)
            total_rows += len(df)
            print(f"已读取: {total_rows} 个样本", end='\r')
            
            if SAMPLE_LIMIT > 0 and total_rows >= SAMPLE_LIMIT:
                break
        except:
            pass

print(f"\n✅ 读取完成: {total_rows} 个样本\n")

df = pd.concat(all_dfs, ignore_index=True)

def extract_image(img_data, save_path):
    if os.path.exists(save_path):
        return str(save_path)
    if pd.isna(img_data):
        return None
    if isinstance(img_data, dict) and 'bytes' in img_data:
        try:
            img = Image.open(BytesIO(img_data['bytes']))
            img.convert('RGB').save(save_path, 'JPEG', quality=90)
            return str(save_path)
        except:
            return None
    return None

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'<image_start>\[.*?\]<image_end>', '', text)
    text = re.sub(r'<image_start>|<image_end>', '', text)
    return text.strip()

converted = []
for idx, row in df.iterrows():
    try:
        images = []
        subset_dir = Path(IMAGE_DIR) / row['subset'].replace(' ', '_')
        subset_dir.mkdir(parents=True, exist_ok=True)
        
        for col in row.index:
            if 'image' in col.lower() and col != 'subset':
                img_path = subset_dir / f"{col}_{idx}.jpg"
                result = extract_image(row[col], img_path)
                if result:
                    images.append(result)
        
        question = clean_text(row.get('Question', ''))
        reasoning = clean_text(row.get('Text Reasoning Trace', ''))
        answer = clean_text(row.get('Final Answer', ''))
        
        if images:
            image_tokens = '\n'.join(['<image>'] * len(images))
            user_content = f"{image_tokens}\n{question}"
        else:
            user_content = question
        
        assistant_content = f"{reasoning}\n\n最终答案: {answer}".strip() if reasoning else f"最终答案: {answer}"
        
        converted.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            "images": images
        })
    except:
        pass

with open(JSON_FILE, 'w', encoding='utf-8') as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

print(f"\n{'='*50}")
print(f"✅ 完成: {len(converted)} 样本")
print(f"   输出: {JSON_FILE}")
print(f"{'='*50}")

EOF