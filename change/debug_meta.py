import json

meta_file = 'data/amazon/meta_Kindle_Store.jsonl'
print(f"查看文件: {meta_file}")

# 查看前3行，了解数据结构
with open(meta_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        item = json.loads(line.strip())
        print(f"\n--- 第 {i+1} 条 ---")
        print(f"keys: {list(item.keys())}")
        print(f"asin: {item.get('asin', 'N/A')}")
        print(f"title: {item.get('title', 'N/A')[:50]}")