import os

# 输入音频目录
audio_dir = r"/所有音频集合"
# 输出 txt 标签目录
labels_dir = r"../所有音频集合_label"

# 确保输出目录存在
os.makedirs(labels_dir, exist_ok=True)

# 映射关系
prefix_to_label = {
    "SL": "湿罗音",
    "ZC": "正常",
}

def generate_txt_labels(audio_dir, labels_dir):
    for fn in os.listdir(audio_dir):
        if fn.lower().endswith((".wav", ".mp3", ".flac")):  # 根据你的音频格式改
            name, _ = os.path.splitext(fn)
            # 前缀判断
            for prefix, label in prefix_to_label.items():
                if name.startswith(prefix):
                    txt_path = os.path.join(labels_dir, name + ".txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(label + "\n")
                    print(f"生成标签文件: {txt_path} -> {label}")
                    break

if __name__ == "__main__":
    generate_txt_labels(audio_dir, labels_dir)
