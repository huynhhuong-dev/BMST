# -*- coding: utf-8 -*-
"""TH01: đánh giá hiệu năng hệ thống sinh trắc học từ tệp điểm so khớp.

Chương trình này gọi các hàm mà sinh viên tự viết trong bio_metrics.py, rồi đối chiếu
kết quả EER với thư viện pyeer. Nếu hai giá trị lệch nhau quá 0,5 điểm phần trăm,
chương trình báo lỗi để sinh viên quay lại kiểm tra lại phần cài đặt.

Cách chạy (trong thư mục TH01_danh-gia-hieu-nang, môi trường ảo của học phần đã kích hoạt):
    python th01_main.py --ma-sv <mã sinh viên>
Kết quả được ghi vào thư mục ket-qua/ với tên đúng quy ước nộp bài, ví dụ
TH01_<mã sinh viên>_bang-ket-qua.csv và TH01_<mã sinh viên>_det.png, nên sinh viên chép thẳng
vào thư mục nộp mà không phải đổi tên.
"""
# %%
import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import bio_metrics as bm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "scores_th01.csv"
OUT = HERE / "ket-qua"
TOLERANCE_PP = 0.5  # sai lệch tối đa cho phép giữa EER tự tính và pyeer, tính bằng điểm phần trăm

# Hệ thống C dùng điểm khoảng cách: càng nhỏ càng giống nhau.
HIGHER_IS_BETTER = {"A": True, "B": True, "C": False}

# pyeer 0.5.6 nhập pkg_resources, nên setuptools 81 in cảnh báo "pkg_resources is deprecated".
# Cảnh báo này đã biết trước và vô hại với môi trường ghim phiên bản của học phần; ta lọc nó để
# sinh viên không nhầm là môi trường bị lỗi.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)


# %%
def load_scores(path):
    """Đọc tệp CSV thành dict: system -> {"genuine": array, "impostor": array}."""
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = data.setdefault(row["system"], {"genuine": [], "impostor": []})
            d[row["pair_type"]].append(float(row["score"]))
    return {k: {t: np.asarray(v) for t, v in d.items()} for k, d in data.items()}


def pyeer_eer(genuine, impostor, higher_is_better):
    """EER theo pyeer; ds_scores=True nghĩa là điểm khoảng cách."""
    from pyeer.eer_info import get_eer_stats

    stats = get_eer_stats(list(genuine), list(impostor), ds_scores=not higher_is_better)
    return float(stats.eer)


# %%
def parse_args():
    ap = argparse.ArgumentParser(description="TH01: đánh giá hiệu năng từ tệp điểm so khớp")
    ap.add_argument("--ma-sv", default=None,
                    help="mã sinh viên, dùng để đặt tên tệp kết quả theo quy ước nộp bài")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.ma_sv:
        prefix = f"TH01_{args.ma_sv.strip()}"
    else:
        prefix = "TH01_ma-sinh-vien"
        print("[CẢNH BÁO] Chưa có --ma-sv: tệp kết quả mang tên tạm TH01_ma-sinh-vien_...; "
              "chạy lại với --ma-sv <mã sinh viên> trước khi nộp.\n")
    OUT.mkdir(exist_ok=True)
    scores = load_scores(DATA)
    rows, det_curves, roc_curves, eer_points, dists = [], {}, {}, {}, {}

    for name in sorted(scores):
        gen, imp = scores[name]["genuine"], scores[name]["impostor"]
        hib = HIGHER_IS_BETTER[name]

        t0 = time.perf_counter()
        thr, fmr, fnmr = bm.error_rates(gen, imp, higher_is_better=hib)
        own = bm.eer(gen, imp, higher_is_better=hib)
        t_own = time.perf_counter() - t0

        t0 = time.perf_counter()
        ref = pyeer_eer(gen, imp, hib)
        t_ref = time.perf_counter() - t0

        gap_pp = abs(own["eer"] - ref) * 100
        fnmr_1, thr_1, _ = bm.fnmr_at_fmr(gen, imp, 0.01, higher_is_better=hib)
        fnmr_01, thr_01, _ = bm.fnmr_at_fmr(gen, imp, 0.001, higher_is_better=hib)
        rows.append({
            "he_thong": name,
            "so_cap_cung_nguoi": len(gen),
            "so_cap_khac_nguoi": len(imp),
            "eer_tu_tinh_%": round(own["eer"] * 100, 3),
            "eer_noi_suy_%": round(own["eer_interp"] * 100, 3),
            "eer_pyeer_%": round(ref * 100, 3),
            "chenh_lech_pp": round(gap_pp, 3),
            "nguong_eer": round(own["threshold"], 4),
            "fnmr_tai_fmr_1%_%": round(fnmr_1 * 100, 2),
            "fnmr_tai_fmr_0.1%_%": round(fnmr_01 * 100, 2),
            "d_prime": round(bm.decidability(gen, imp), 3),
            "auc": round(bm.roc_auc(fmr, fnmr), 4),
            "thoi_gian_tu_tinh_s": round(t_own, 3),
            "thoi_gian_pyeer_s": round(t_ref, 3),
        })
        label = f"{name} (EER {own['eer'] * 100:.2f}%)".replace(".", ",")
        det_curves[label] = (fmr, fnmr)
        roc_curves[label] = (fmr, fnmr)
        eer_points[label] = own["eer"]
        dists[f"Hệ thống {name}" + ("" if hib else " (khoảng cách)")] = (gen, imp)

        status = "ĐẠT" if gap_pp <= TOLERANCE_PP else "CHƯA ĐẠT"
        print(f"[{name}] EER tự tính = {own['eer'] * 100:.3f}%, pyeer = {ref * 100:.3f}%, "
              f"lệch {gap_pp:.3f} điểm phần trăm: {status}")

    with open(OUT / f"{prefix}_bang-ket-qua.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bm.plot_distributions(dists, OUT / f"{prefix}_phan-bo-diem.png")
    bm.plot_det(det_curves, OUT / f"{prefix}_det.png", title="Đường cong DET của ba hệ thống",
                eer_points=eer_points)
    bm.plot_roc(roc_curves, OUT / f"{prefix}_roc.png", title="Đường cong ROC của ba hệ thống")

    # %%
    # Từ xác minh 1:1 sang định danh 1:N (giả thiết các phép so khớp độc lập).
    gen, imp = scores["A"]["genuine"], scores["A"]["impostor"]
    _, thr_01, fmr_real = bm.fnmr_at_fmr(gen, imp, 0.001)
    print(f"\nHệ thống A tại ngưỡng {thr_01:.4f}: FMR đo được = {fmr_real * 100:.3f}%")
    for n in [1_000, 1_000_000, 100_000_000]:
        print(f"  N = {n:>11,}: số so khớp sai kỳ vọng N*FMR = {n * fmr_real:,.2f}; "
              f"FPIR = 1-(1-FMR)^N = {bm.fpir_from_fmr(fmr_real, n):.4f}")

    # TODO (câu hỏi phân tích 3): với 10.000 cặp khác người và 0 lần so khớp sai,
    # hãy tính cận trên 95% của FMR theo "quy tắc số 3" và in ra tại đây.
    # Tệp th01_main.py nằm trong danh sách nộp bài để trợ giảng xem được phần này.
    n_rule3 = 10_000
    p_upper_95 = 3.0 / n_rule3
    print(f"\nQuy tắc số 3 (Rule of Three) với N = {n_rule3:,} và 0 lần so khớp sai:")
    print(f"  Cận trên tin cậy 95% của FMR là p <= 3/N = {p_upper_95:.6f} ({p_upper_95 * 100:.3f}%).")
    print(f"  Nhận xét: '0 lỗi' trên mẫu thử không đồng nghĩa với FMR = 0, mà chỉ chứng minh FMR <= 0,03% ở mức tin cậy 95%.")

    failed = [r["he_thong"] for r in rows if r["chenh_lech_pp"] > TOLERANCE_PP]
    if failed:
        print(f"\nCác hệ thống chưa khớp pyeer: {failed}. Hãy kiểm tra lại bio_metrics.py.")
        sys.exit(1)
    print(f"\nĐã ghi bảng và đồ thị vào {OUT} (tiền tố {prefix})")


if __name__ == "__main__":
    main()
