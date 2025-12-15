# -*- coding: utf-8 -*-
"""
============================================================
Traditional ML - Real vs Synthetic (Ratio + Weights + Filter)
============================================================
- 分开读取真实/合成训练集（.npy + .txt）
- 真实样本全量使用；合成样本按每类比例或绝对上限加入
- Teacher 质量过滤（预测=标注且置信度≥阈值）
- 真实/合成样本权重
- 多模型评估与保存
"""

# ====================== 0) 全局配置（可直接改） ======================
import os, sys, csv, json, time, platform, random
from pathlib import Path
from typing import Tuple, List, Dict, Sequence

import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# —— 类别映射（与项目一致）——
LABEL_MAP = {"正常": 0, "湿罗音": 1}
IDX2LABEL = {v: k for k, v in LABEL_MAP.items()}

# —— 随机性 ——
RNG_SEED = 42
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

# —— 输出根目录 ——
OUT_ROOT_DEFAULT = "./ml_eval_outputs_v2"

# —— 默认数据目录（可直接改成你机器上的路径；也可运行时回车使用）——
DEFAULT_PATHS = {
    "train_real_audio": r"E:\cat的方法\T192_mel163\所有音频集合_dataset\train_audio_norm",
    "train_real_label": r"E:\cat的方法\T192_mel163\所有音频集合_dataset\train_label",
    "train_syn_audio":  r"E:\cat的方法\generated_by_class\mels",   # 若没有可留空/不存在
    "train_syn_label":  r"E:\cat的方法\generated_by_class\labels",
    "val_audio":        r"E:\cat的方法\T192_mel163\所有音频集合_dataset\val_audio_norm",
    "val_label":        r"E:\cat的方法\T192_mel163\所有音频集合_dataset\val_label",
}

# —— 合成样本加入策略（两种只需设其一；两者都设则同时生效，取更小的限制）——
SYN_RATIO_PER_CLASS = 0.5  # 每类合成样本 ≤ 真实样本数 * 0.5；设 None 代表不用比例限制
SYN_MAX_PER_CLASS   = None  # 每类合成样本绝对上限（如 50）；设 None 代表不用绝对上限

# —— Teacher 质量过滤 ——
TEACHER_PROB_TH = 0.0      # 置信度阈值（0.5~0.8常见）
USE_TEACHER_FILTER = True   # 是否启用

# —— 样本权重 ——
REAL_WEIGHT = 1.0
SYN_WEIGHT  = 0.5          # 合成样本权重（0.3~0.6 常用）
USE_SAMPLE_WEIGHTS = True

# ====================== 1) 小工具：IO / 可视化 ======================
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def now_str() -> str: return time.strftime("%Y%m%d_%H%M%S")
def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2, ensure_ascii=False)
def prompt_or_default(msg: str, default: str) -> str:
    try:
        s = input(f"{msg} [默认: {default}]：").strip()
        return s if s else default
    except Exception:
        return default

def plot_and_save_cm(cm: np.ndarray, classes: List[str], title: str, out_path: Path):
    plt.figure(figsize=(4.8, 4.8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)
    thr = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thr else "black")
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

# ====================== 2) 特征工程：从 [T,F] → 4F ======================
def extract_features(mel: np.ndarray) -> np.ndarray:
    """[T,F] → 4F (mean, std, Δmean, Δstd) 沿时间轴统计"""
    assert mel.ndim == 2, f"Expected [T,F], got {mel.shape}"
    mu = mel.mean(axis=0); sd = mel.std(axis=0)
    if mel.shape[0] > 1:
        d = np.diff(mel, axis=0); d_mu = d.mean(axis=0); d_sd = d.std(axis=0)
    else:
        F = mel.shape[1]; d_mu = np.zeros(F, np.float32); d_sd = np.zeros(F, np.float32)
    return np.concatenate([mu, sd, d_mu, d_sd], axis=0).astype(np.float32)

# ====================== 3) 读数据（兼容标签两种命名） ======================
def label_path_for(npy_name: str, label_dir: Path) -> Path:
    """
    支持：
      - xxx.npy       -> xxx.txt
      - xxx_mel.npy   -> xxx_label.txt (优先) 或 xxx_mel.txt (备选)
    """
    base = npy_name[:-4]  # 去掉 .npy
    # 情况A：标准规则
    p_std = label_dir / f"{base}.txt"
    if p_std.exists(): return p_std
    # 情况B：兼容 _mel.npy / _label.txt
    if base.endswith("_mel"):
        p_label = label_dir / f"{base[:-4]}_label.txt"
        if p_label.exists(): return p_label
        p_mel_txt = label_dir / f"{base}.txt"
        if p_mel_txt.exists(): return p_mel_txt
    # 默认返回标准路径（即便不存在，外层再判断）
    return p_std

def load_split(npy_dir: str, label_dir: str, tag_hint: str = ""):
    """
    读取一个 split，返回：
      X: [N, 4F]，y: [N]，paths: List[str]（每条对应的 .npy 路径），
      labels_str: List[str]（原始中文标签文本），bad: List[str]（异常项）
    """
    npy_dir, label_dir = Path(npy_dir), Path(label_dir)
    if not npy_dir.exists(): raise FileNotFoundError(f"{tag_hint} npy_dir 不存在: {npy_dir}")
    if not label_dir.exists(): raise FileNotFoundError(f"{tag_hint} label_dir 不存在: {label_dir}")

    X, y, paths, labels_str, bad = [], [], [], [], []
    for fn in sorted(p.name for p in npy_dir.glob("*.npy")):
        fpath = npy_dir / fn
        lpath = label_path_for(fn, label_dir)
        if not lpath.exists():
            bad.append(f"[无标签] {fn}")
            continue
        try:
            mel = np.load(fpath)
        except Exception as e:
            bad.append(f"[读取失败] {fn}: {e}")
            continue

        with open(lpath, "r", encoding="utf-8") as f:
            t = f.read().strip()
        if t not in LABEL_MAP:
            bad.append(f"[未知标签] {fn}: {t}")
            continue

        X.append(extract_features(mel))
        y.append(LABEL_MAP[t])
        paths.append(str(fpath))
        labels_str.append(t)

    if len(X) == 0:
        raise RuntimeError(f"{tag_hint} 目录 {npy_dir} 下未成功读取到样本。")
    return np.vstack(X), np.array(y, np.int64), paths, labels_str, bad

# ====================== 4) Teacher 过滤与采样策略 ======================
def train_teacher(X_real: np.ndarray, y_real: np.ndarray):
    """用真实数据训练 Teacher（LogReg+Scaler，可 predict_proba）"""
    teacher = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, random_state=RNG_SEED, class_weight="balanced"))
    ])
    teacher.fit(X_real, y_real)
    return teacher

def filter_synthetic_by_teacher(
    X_syn: np.ndarray, y_syn: np.ndarray, paths_syn: Sequence[str], teacher, prob_th: float
):
    """保留 预测=标注 且 max_proba≥阈值 的合成样本"""
    proba = teacher.predict_proba(X_syn)
    pred  = proba.argmax(axis=1)
    ok = (pred == y_syn) & (proba.max(axis=1) >= prob_th)
    Xf, yf = X_syn[ok], y_syn[ok]
    pf = [p for keep, p in zip(ok, paths_syn) if keep]
    return Xf, yf, pf

def per_class_indices(y: np.ndarray, n_classes: int) -> Dict[int, np.ndarray]:
    idx = {c: np.where(y == c)[0] for c in range(n_classes)}
    return idx

def pick_synthetic_by_quota(
    X_syn_f, y_syn_f, paths_syn_f, real_counts: Dict[int, int],
    ratio: float = None, abs_max: int = None, n_classes: int = 3
):
    """
    按每类配额选择合成样本：
      - 每类上限 = min( floor(real_cnt * ratio), abs_max )
      - ratio 或 abs_max 可为 None；若均 None 则全部使用
    """
    if ratio is None and abs_max is None:
        return X_syn_f, y_syn_f, paths_syn_f

    sel_mask = np.zeros(len(y_syn_f), dtype=bool)
    per_class = per_class_indices(y_syn_f, n_classes)
    for c, idxs in per_class.items():
        if len(idxs) == 0:
            continue
        quota_list = []
        if ratio is not None:
            quota_list.append(int(np.floor(real_counts.get(c, 0) * ratio)))
        if abs_max is not None:
            quota_list.append(int(abs_max))
        quota = min([q for q in quota_list if q is not None], default=len(idxs))
        quota = max(0, min(quota, len(idxs)))
        chosen = np.random.choice(idxs, size=quota, replace=False) if quota > 0 else np.array([], dtype=int)
        sel_mask[chosen] = True

    return X_syn_f[sel_mask], y_syn_f[sel_mask], [p for k,p in zip(sel_mask, paths_syn_f) if k]

def compute_sample_weights(paths: Sequence[str], w_real=1.0, w_syn=0.4):
    flags = np.array([("generated" in p.lower()) or ("gen" in p.lower()) or ("_mel.npy" in p.lower())
                      or ("syn" in p.lower()) for p in paths], dtype=bool)
    w = np.where(flags, w_syn, w_real).astype(np.float32)
    return w

# ====================== 5) 模型集合（支持 sample_weight） ======================
def build_models():
    models = {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, random_state=RNG_SEED, class_weight="balanced"))
        ]),
        "SVM-RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RNG_SEED, class_weight="balanced"))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=None, random_state=RNG_SEED, n_jobs=-1, class_weight="balanced_subsample"
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
    }
    try:
        from xgboost import XGBClassifier  # noqa
        models["XGBoost"] = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RNG_SEED, tree_method="hist", n_jobs=-1
        )
    except Exception:
        pass
    return models

def fit_with_optional_weights(model_name: str, model, X, y, sample_weight=None):
    """
    对 Pipeline 需用 stepname__param 传入权重；对原生估计器可直接传。
    - LogReg、SVM 在 Pipeline 中，末步名为 'clf'
    - RandomForest 是原生估计器
    - KNN 不支持 sample_weight
    """
    if sample_weight is None:
        return model.fit(X, y)

    if model_name in ["KNN"]:
        # KNN 不支持 sample_weight
        return model.fit(X, y)

    if isinstance(model, Pipeline):
        return model.fit(X, y, **{"clf__sample_weight": sample_weight})

    # 原生估计器（RF / XGB）
    try:
        return model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        return model.fit(X, y)

# ====================== 6) 训练与评估（两组：真实-only；真实+合成） ======================
def train_and_eval_group(Xtr, ytr, Xte, yte, tag, save_root: Path, results_csv: Path, use_weights=False, weights=None):
    models = build_models()
    ensure_dir(save_root / "saved_models")
    ensure_dir(save_root / "confusion_matrices")

    header = ["tag", "model", "acc", "f1_macro"]
    file_exists = results_csv.exists()
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists: w.writerow(header)

        metrics = {}
        print(f"\n===== [{tag}] =====")
        for name, clf in models.items():
            fit_with_optional_weights(name, clf, Xtr, ytr, sample_weight=(weights if use_weights else None))
            pred = clf.predict(Xte)
            acc = accuracy_score(yte, pred)
            f1m = f1_score(yte, pred, average="macro")
            cm  = confusion_matrix(yte, pred)

            print(f"\n-- {name} --")
            print(f"Accuracy : {acc:.4f}")
            print(f"F1-macro: {f1m:.4f}")
            print("\nConfusion Matrix:\n", cm)
            print("\nClassification Report:\n", classification_report(
                yte, pred, digits=4, target_names=[IDX2LABEL[i] for i in range(len(IDX2LABEL))]
            ))

            w.writerow([tag, name, f"{acc:.6f}", f"{f1m:.6f}"])
            joblib.dump(clf, save_root / "saved_models" / f"{tag}_{name}.pkl")
            plot_and_save_cm(cm, [IDX2LABEL[i] for i in range(len(IDX2LABEL))], f"{tag} - {name}",
                             save_root / "confusion_matrices" / f"{tag}_{name}_cm.png")
            metrics[name] = {"acc": float(acc), "f1_macro": float(f1m)}
    return metrics

# ====================== 7) 主流程 ======================
def main():
    print("\n=== 真实 vs 合成：比例+权重+过滤（V2）===")
    # 路径（可直接在 DEFAULT_PATHS 改；也可运行时输入）
    train_real_audio = prompt_or_default("真实训练 .npy 目录", DEFAULT_PATHS["train_real_audio"])
    train_real_label = prompt_or_default("真实训练 .txt 目录", DEFAULT_PATHS["train_real_label"])
    train_syn_audio  = prompt_or_default("合成训练 .npy 目录（可空）", DEFAULT_PATHS["train_syn_audio"])
    train_syn_label  = prompt_or_default("合成训练 .txt 目录（可空）", DEFAULT_PATHS["train_syn_label"])
    val_audio        = prompt_or_default("验证 .npy 目录", DEFAULT_PATHS["val_audio"])
    val_label        = prompt_or_default("验证 .txt 目录", DEFAULT_PATHS["val_label"])
    out_root         = prompt_or_default("输出根目录", OUT_ROOT_DEFAULT)

    run_dir = Path(out_root) / now_str()
    ensure_dir(run_dir)
    results_csv = run_dir / "results.csv"

    # 读真实训练
    X_real, y_real, paths_real, labels_real, badR = load_split(train_real_audio, train_real_label, "真实训练")
    if badR:
        print("\n[警告] 真实训练集中异常：")
        for s in badR: print("  -", s)

    # 读验证集
    X_val, y_val, paths_val, labels_val, badV = load_split(val_audio, val_label, "验证集")
    if badV:
        print("\n[警告] 验证集中异常：")
        for s in badV: print("  -", s)

    # 真实-only 基线（不使用权重）
    metrics_real = train_and_eval_group(
        X_real, y_real, X_val, y_val,
        tag="real_only", save_root=run_dir, results_csv=results_csv,
        use_weights=False, weights=None
    )

    summary = {"real_only": metrics_real}

    # 合成路径有效才做第二组
    if train_syn_audio and train_syn_label and Path(train_syn_audio).exists() and Path(train_syn_label).exists():
        # 读合成训练
        X_syn, y_syn, paths_syn, labels_syn, badS = load_split(train_syn_audio, train_syn_label, "合成训练")
        if badS:
            print("\n[警告] 合成训练集中异常：")
            for s in badS: print("  -", s)

        # Teacher 质量过滤（可选）
        if USE_TEACHER_FILTER:
            teacher = train_teacher(X_real, y_real)
            X_syn_f, y_syn_f, paths_syn_f = filter_synthetic_by_teacher(
                X_syn, y_syn, paths_syn, teacher, prob_th=TEACHER_PROB_TH
            )
        else:
            X_syn_f, y_syn_f, paths_syn_f = X_syn, y_syn, paths_syn

        # 统计真实每类数量
        real_counts = {c: int((y_real == c).sum()) for c in range(len(IDX2LABEL))}
        # 按配额挑选合成
        X_syn_pick, y_syn_pick, paths_syn_pick = pick_synthetic_by_quota(
            X_syn_f, y_syn_f, paths_syn_f,
            real_counts=real_counts,
            ratio=SYN_RATIO_PER_CLASS,
            abs_max=SYN_MAX_PER_CLASS,
            n_classes=len(IDX2LABEL)
        )

        # 合并训练集（真实全部 + 挑选后的合成）
        X_all = np.vstack([X_real, X_syn_pick])
        y_all = np.concatenate([y_real, y_syn_pick])
        paths_all = paths_real + paths_syn_pick

        # 样本权重（可选）
        weights = compute_sample_weights(paths_all, w_real=REAL_WEIGHT, w_syn=SYN_WEIGHT) \
                  if USE_SAMPLE_WEIGHTS else None

        # 第二组训练
        print("\n[信息] 合并训练集规模：真实={}，合成(挑选后)={}".format(len(y_real), len(y_syn_pick)))
        print("[信息] 每类真实数量：", real_counts)
        if USE_TEACHER_FILTER:
            print("[信息] Teacher过滤阈值 =", TEACHER_PROB_TH)
        if SYN_RATIO_PER_CLASS is not None or SYN_MAX_PER_CLASS is not None:
            print("[信息] 合成配额：ratio =", SYN_RATIO_PER_CLASS, ", abs_max =", SYN_MAX_PER_CLASS)
        if USE_SAMPLE_WEIGHTS:
            print("[信息] 样本权重：real =", REAL_WEIGHT, ", syn =", SYN_WEIGHT)

        metrics_all = train_and_eval_group(
            X_all, y_all, X_val, y_val,
            tag="real_plus_synth", save_root=run_dir, results_csv=results_csv,
            use_weights=USE_SAMPLE_WEIGHTS, weights=weights
        )
        summary["real_plus_synth"] = metrics_all
    else:
        print("\n[提示] 未提供有效的合成训练目录，跳过“真实+合成”。")

    # 保存本次配置与汇总
    run_cfg = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": {"python": sys.version, "os": platform.platform(), "numpy": np.__version__,
                     "sklearn": __import__("sklearn").__version__},
        "paths": {
            "train_real_audio": train_real_audio, "train_real_label": train_real_label,
            "train_syn_audio": train_syn_audio, "train_syn_label": train_syn_label,
            "val_audio": val_audio, "val_label": val_label
        },
        "controls": {
            "USE_TEACHER_FILTER": USE_TEACHER_FILTER, "TEACHER_PROB_TH": TEACHER_PROB_TH,
            "SYN_RATIO_PER_CLASS": SYN_RATIO_PER_CLASS, "SYN_MAX_PER_CLASS": SYN_MAX_PER_CLASS,
            "USE_SAMPLE_WEIGHTS": USE_SAMPLE_WEIGHTS, "REAL_WEIGHT": REAL_WEIGHT, "SYN_WEIGHT": SYN_WEIGHT,
            "RNG_SEED": RNG_SEED
        }
    }
    save_json(run_cfg, run_dir / "run_config.json")
    save_json(summary, run_dir / "run_summary.json")

    print(f"\n[完成] 输出位于：{run_dir.resolve()}\n"
          f"- results.csv / run_config.json / run_summary.json\n"
          f"- saved_models/*.pkl / confusion_matrices/*.png\n")

# ====================== 8) 入口 ======================
if __name__ == "__main__":
    main()
