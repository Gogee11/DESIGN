import os

from huggingface_hub import snapshot_download

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
print("开始下载模型...")
print("目标目录: D:/models/all-MiniLM-L6-v2")

model_path = snapshot_download(
    repo_id='sentence-transformers/all-MiniLM-L6-v2',
    local_dir='D:/models/all-MiniLM-L6-v2',
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4,
    ignore_patterns=["*.safetensors"]  # 只下载需要的文件
)

print(f' 模型下载完成！保存在: {model_path}')
print('\n文件列表:')
for file in os.listdir(model_path):
    file_path = os.path.join(model_path, file)
    if os.path.isfile(file_path):
        size = os.path.getsize(file_path) / (1024*1024)
        print(f'  - {file}: {size:.2f} MB')