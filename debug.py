# import os
#
# labels_dir = r"E:\cat的方法\generated_by_class20k\labels"
#
# for fn in os.listdir(labels_dir):
#     if fn.endswith("_label.txt"):
#         old_path = os.path.join(labels_dir, fn)
#         new_fn = fn.replace("_label.txt", ".txt")
#         new_path = os.path.join(labels_dir, new_fn)
#
#         # 如果新名字已经存在，避免覆盖
#         if os.path.exists(new_path):
#             print(f"⚠️ 跳过 {fn}，目标文件已存在")
#             continue
#
#         os.rename(old_path, new_path)
#         print(f"✅ 重命名: {fn} -> {new_fn}")
#
# print("全部处理完成！")

import os

labels_dir = r"E:\cat的方法\generated_by_class20k_copy\labels"

for fn in os.listdir(labels_dir):
    if fn.endswith(".txt"):
        old_path = os.path.join(labels_dir, fn)
        new_fn = fn.replace(".txt", "_mel.txt")
        new_path = os.path.join(labels_dir, new_fn)

        # 如果新名字已经存在，避免覆盖
        if os.path.exists(new_path):
            print(f"⚠️ 跳过 {fn}，目标文件已存在")
            continue

        os.rename(old_path, new_path)
        print(f"✅ 重命名: {fn} -> {new_fn}")

print("全部处理完成！")
