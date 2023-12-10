import os
import zipfile

# 指定源文件夹和目标文件夹
source_dir = "./TH"
target_dir = "./BulkHands"

# 遍历源文件夹下的所有文件
for filename in os.listdir(source_dir):
    # 如果是zip文件，就解压缩到目标文件夹
    if filename.endswith(".zip"):
        # 拼接完整的文件路径
        filepath = os.path.join(source_dir, filename)
        # 打开zip文件
        with zipfile.ZipFile(filepath, "r") as zip_file:
            # 解压缩到目标文件夹
            zip_file.extractall(target_dir)