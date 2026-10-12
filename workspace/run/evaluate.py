"""评测入口（M2）：读取对局日志，计算推理指标并产出可读报告。

用法：
    python run/evaluate.py --log-dir logs --out analysis

产出（analysis/）：
    - metrics.csv        每局指标明细
    - report.txt         文字结论（规则 5：可读）
    - restoration.png    各模型均值还原度柱状图（无 matplotlib 时跳过）

不依赖 API key，纯离线复算。
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import metrics


def main():
    parser = argparse.ArgumentParser(description="AI-Arena 评测报告")
    parser.add_argument("--log-dir", default="logs", help="对局 JSON 目录")
    parser.add_argument("--out", default="analysis", help="报告输出目录")
    args = parser.parse_args()

    logs = metrics.load_logs(args.log_dir)
    if not logs:
        raise SystemExit(f"❌ 在 {args.log_dir} 未找到任何对局日志。请先跑 run/single_game.py 生成。")
    rows = metrics.summarize(logs)
    agg = metrics.aggregate_by_model(rows)

    os.makedirs(args.out, exist_ok=True)

    # 1) 明细 CSV
    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 2) 文字结论
    report = metrics.text_report(agg)
    txt_path = os.path.join(args.out, "report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # 3) 图表
    png_path = os.path.join(args.out, "restoration.png")
    chart = metrics.make_chart(agg, png_path)
    if chart is None:
        png_path = "（未安装 matplotlib，已跳过图表）"

    print("✅ 评测完成")
    print(report)
    print(f"\n明细   : {csv_path}")
    print(f"结论   : {txt_path}")
    print(f"图表   : {png_path}")


if __name__ == "__main__":
    main()
