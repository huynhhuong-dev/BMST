# -*- coding: utf-8 -*-
"""Kiểm thử nhỏ cho bio_metrics.py với ví dụ tính được bằng tay.

Cách chạy: python test_bio_metrics.py
Vì sao cần: nếu hàm sai ở ví dụ 9 điểm, nó chắc chắn sai ở 11.000 điểm; lỗi trên ví dụ
nhỏ dễ tìm hơn nhiều so với lỗi trên đồ thị.

Tổng số phép kiểm thử luôn cố định (in ở dòng cuối). Hàm chưa viết được tính là chưa đạt ở
mọi phép kiểm thử của nhóm đó, để mẫu số không đổi trong suốt quá trình làm bài.
"""
import sys

import numpy as np

import bio_metrics as bm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

GEN = [0.9, 0.8, 0.7, 0.4]            # 4 cặp cùng người
IMP = [0.1, 0.2, 0.3, 0.6, 0.75]      # 5 cặp khác người


def check(name, got, expected, tol=1e-9, hint=""):
    ok = bool(np.allclose(got, expected, atol=tol))
    print(f"{'ĐẠT     ' if ok else 'CHƯA ĐẠT'} {name}: nhận {got}, mong đợi {expected}")
    if not ok and hint:
        print(f"         gợi ý: {hint}")
    return ok


def t_rates():
    # Tại t = 0,5: khác người được chấp nhận là 0,6 và 0,75 nên FMR = 2/5;
    # cùng người bị từ chối là 0,4 nên FNMR = 1/4.
    thr, fmr, fnmr = bm.error_rates(GEN, IMP, thresholds=[0.5])
    return [check("FMR(0,5)", fmr[0], 0.4), check("FNMR(0,5)", fnmr[0], 0.25)]


def t_rates_distance():
    # Cùng dữ liệu ở dạng khoảng cách d = 1 - s, ngưỡng 0,5: kết quả phải giữ nguyên.
    thr, fmr, fnmr = bm.error_rates(1 - np.array(GEN), 1 - np.array(IMP), thresholds=[0.5],
                                    higher_is_better=False)
    return [check("FMR khoảng cách", fmr[0], 0.4), check("FNMR khoảng cách", fnmr[0], 0.25)]


def t_rates_tie():
    # Điểm bằng đúng ngưỡng phải được CHẤP NHẬN (s >= t).
    # t = 0,7 trùng một điểm cùng người: chỉ 0,4 bị từ chối, FNMR = 1/4 (dùng ">" sẽ ra 2/4).
    # t = 0,75 trùng một điểm khác người: 0,75 được chấp nhận, FMR = 1/5 (dùng ">" sẽ ra 0).
    thr, fmr, fnmr = bm.error_rates(GEN, IMP, thresholds=[0.7, 0.75])
    hint = "điểm bằng ngưỡng phải được chấp nhận; xem lại tham số side của np.searchsorted"
    return [check("FMR(0,7), điểm bằng ngưỡng", fmr[0], 0.2, hint=hint),
            check("FNMR(0,7), điểm bằng ngưỡng", fnmr[0], 0.25, hint=hint),
            check("FMR(0,75), điểm bằng ngưỡng", fmr[1], 0.2, hint=hint),
            check("FNMR(0,75), điểm bằng ngưỡng", fnmr[1], 0.5, hint=hint)]


def t_rates_tie_distance():
    # Khoảng cách: 1 - 0,75 = 0,25 biểu diễn chính xác trong dấu phẩy động, nên ngưỡng 0,25 trùng
    # đúng một điểm khác người. Chấp nhận khi d <= t: FMR = 1/5; cùng người có d = 0,1 và 0,2
    # được chấp nhận, d = 0,3 và 0,6 bị từ chối: FNMR = 2/4.
    thr, fmr, fnmr = bm.error_rates(1 - np.array(GEN), 1 - np.array(IMP), thresholds=[0.25],
                                    higher_is_better=False)
    hint = "với điểm khoảng cách, điểm bằng ngưỡng cũng phải được chấp nhận (d <= t)"
    return [check("FMR khoảng cách, điểm bằng ngưỡng", fmr[0], 0.2, hint=hint),
            check("FNMR khoảng cách, điểm bằng ngưỡng", fnmr[0], 0.5, hint=hint)]


def t_rates_default_thresholds():
    # Ngưỡng mặc định: đầu đường cong chấp nhận mọi cặp, cuối đường cong có điểm FMR = 0.
    thr, fmr, fnmr = bm.error_rates(GEN, IMP)
    return [check("Đầu đường cong: (FMR, FNMR) = (1, 0)", [fmr[0], fnmr[0]], [1.0, 0.0]),
            check("Cuối đường cong: FMR = 0, FNMR = 1", [fmr[-1], fnmr[-1]], [0.0, 1.0])]


def t_eer():
    # Ngưỡng t = 0,7: FMR = 1/5 (chỉ 0,75), FNMR = 1/4 (chỉ 0,4): EER = (0,2 + 0,25)/2.
    return [check("EER", bm.eer(GEN, IMP)["eer"], 0.225)]


def t_eer_interp():
    # Ngưỡng liền trước 0,7 là 0,6: FMR = 0,4, FNMR = 0,25, diff = 0,15; tại 0,7: diff = -0,05.
    # w = 0,15 / 0,2 = 0,75; eer_interp = 0,4 + 0,75 x (0,2 - 0,4) = 0,25.
    return [check("EER nội suy", bm.eer(GEN, IMP)["eer_interp"], 0.25,
                  hint="w = diff[i-1] / (diff[i-1] - diff[i]); nội suy FMR giữa i-1 và i")]


def t_fnmr_at_fmr():
    # FMR <= 0: ngưỡng phải vượt 0,75, tức t = 0,8: FNMR = 2/4.
    f, t, fm = bm.fnmr_at_fmr(GEN, IMP, 0.0)
    return [check("FNMR tại FMR = 0", f, 0.5)]


def t_decidability():
    d = bm.decidability([1.0, 3.0], [-1.0, -3.0])   # mu 2 và -2, sigma tổng thể = 1: d' = 4
    return [check("d'", d, 4.0,
                  hint="nhận 2,83 nghĩa là đang dùng phương sai mẫu (ddof=1); cần ddof=0")]


def t_fpir():
    return [check("FPIR với FMR = 0,01, N = 2", bm.fpir_from_fmr(0.01, 2), 0.0199)]


def t_roc_auc():
    # Ví dụ 9 điểm: AUC hình thang trên ROC thực nghiệm bằng tỷ lệ cặp (cùng người, khác người)
    # mà điểm cùng người lớn hơn: 0,9 và 0,8 thắng cả 5; 0,7 thắng 4; 0,4 thắng 3; (5+5+4+3)/20.
    thr, fmr, fnmr = bm.error_rates(GEN, IMP)
    auc = bm.roc_auc(fmr, fnmr)
    # Hai phân bố tách rời hoàn toàn: AUC = 1, bất kể thứ tự phần tử đầu vào.
    fmr2 = np.array([0.0, 1.0, 0.0, 0.0])
    fnmr2 = np.array([1.0, 0.0, 0.0, 0.5])
    hint = "sắp theo FMR tăng dần, khi bằng nhau theo TPR tăng dần (np.lexsort), rồi np.trapezoid"
    return [check("AUC ví dụ 9 điểm", auc, 0.85, hint=hint),
            check("AUC khi đầu vào chưa sắp xếp", bm.roc_auc(fmr2, fnmr2), 1.0, hint=hint)]


def t_probit():
    got = bm._probit(np.array([0.5, 0.975, 0.0]))
    return [check("probit(0,5) và probit(0,975)", got[:2], [0.0, 1.959963985], tol=1e-6),
            check("probit(0) được kẹp, hữu hạn", got[2], -4.264890794, tol=1e-6,
                  hint="kẹp p vào [lo, 1 - lo] trước khi gọi norm.ppf")]


# (tên nhóm, hàm kiểm thử, số phép kiểm thử trong nhóm)
GROUPS = [
    ("error_rates", t_rates, 2),
    ("error_rates (khoảng cách)", t_rates_distance, 2),
    ("error_rates (điểm bằng ngưỡng)", t_rates_tie, 4),
    ("error_rates (khoảng cách, điểm bằng ngưỡng)", t_rates_tie_distance, 2),
    ("error_rates (ngưỡng mặc định)", t_rates_default_thresholds, 2),
    ("eer", t_eer, 1),
    ("eer (nội suy)", t_eer_interp, 1),
    ("fnmr_at_fmr", t_fnmr_at_fmr, 1),
    ("decidability", t_decidability, 1),
    ("fpir_from_fmr", t_fpir, 1),
    ("roc_auc", t_roc_auc, 2),
    ("_probit", t_probit, 2),
]
TOTAL = sum(n for _, _, n in GROUPS)


def run(name, fn, n_checks):
    """Chạy một nhóm; nếu hàm chưa viết thì ghi nhận chưa đạt cho cả nhóm thay vì dừng hẳn."""
    try:
        res = fn()
    except NotImplementedError as exc:
        print(f"CHƯA LÀM {name}: {exc} ({n_checks} phép kiểm thử)")
        return [False] * n_checks
    except Exception as exc:  # lỗi khi chạy (ví dụ gọi np.trapz đã bị gỡ khỏi numpy 2.x)
        print(f"LỖI     {name}: {type(exc).__name__}: {exc} ({n_checks} phép kiểm thử)")
        return [False] * n_checks
    assert len(res) == n_checks, f"nhóm {name} khai báo {n_checks} phép kiểm thử"
    return res


def main():
    results = []
    for name, fn, n_checks in GROUPS:
        results += run(name, fn, n_checks)
    print(f"\n{sum(results)}/{TOTAL} phép kiểm thử đạt.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
