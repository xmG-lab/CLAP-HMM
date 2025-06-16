import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_labels(pred_file, true_file):
    """
    加载预测值与真实标签
    要求格式为csv或tsv，包含两列：id, label
    """
    pred_df = pd.read_csv(pred_file)
    true_df = pd.read_csv(true_file)

    if 'label' not in pred_df.columns or 'label' not in true_df.columns:
        raise ValueError("CSV文件中必须包含 'label' 列")

    # 若存在id列，按 id 对齐
    if 'id' in pred_df.columns and 'id' in true_df.columns:
        merged = pd.merge(true_df, pred_df, on='id', suffixes=('_true', '_pred'))
        y_true = merged['label_true'].values
        y_pred = merged['label_pred'].values
    else:
        y_true = true_df['label'].values
        y_pred = pred_df['label'].values

    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print("=== Evaluation Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"MCC      : {mcc:.4f}")

    return acc, prec, rec, f1, mcc


def plot_confusion_matrix(y_true, y_pred, out_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Non-R", "R"], yticklabels=["Non-R", "R"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Confusion matrix saved to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction results")
    parser.add_argument('--pred', type=str, required=True, help="预测标签文件 (csv with id,label)")
    parser.add_argument('--true', type=str, required=True, help="真实标签文件 (csv with id,label)")
    parser.add_argument('--plot', action='store_true', help="是否保存混淆矩阵图像")
    parser.add_argument('--out', type=str, default="confusion_matrix.png", help="混淆矩阵输出路径")

    args = parser.parse_args()

    y_true, y_pred = load_labels(args.pred, args.true)
    compute_metrics(y_true, y_pred)

    if args.plot:
        plot_confusion_matrix(y_true, y_pred, args.out)


if __name__ == '__main__':
    main()
