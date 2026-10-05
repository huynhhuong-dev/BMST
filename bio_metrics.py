# -*- coding: utf-8 -*-
"""bio_metrics.py: thư viện đo hiệu năng do sinh viên tự viết trong TH01.

Tệp này được dùng lại ở TH03, TH04, TH05, TH06, TH07, TH09 và TH10, vì vậy các bạn nên
viết cẩn thận, có chú thích, và kiểm thử kỹ ngay từ buổi đầu.

Quy ước dùng trong toàn bộ bộ bài thực hành:
- Điểm tương đồng (similarity, higher_is_better=True): chấp nhận khi score >= t.
- Điểm khoảng cách (distance, higher_is_better=False): chấp nhận khi score <= t.
- FMR(t)  = số cặp khác người được chấp nhận / tổng số cặp khác người.
- FNMR(t) = số cặp cùng người bị từ chối / tổng số cặp cùng người.

Mỗi chỗ "TODO" là một việc sinh viên cần làm. Sinh viên không dùng pyeer hay sklearn.metrics
trong tệp này, vì mục tiêu của bài là tự hiểu cách các con số được tạo ra.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.stats import norm
except ImportError:  # scipy chỉ cần cho đồ thị DET
    norm = None


def _as_array(x):
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("Mảng điểm rỗng: kiểm tra lại bước đọc dữ liệu.")
    return arr


def error_rates(genuine, impostor, thresholds=None, higher_is_better=True):
    """Tính FMR và FNMR tại từng ngưỡng.

    Trả về (thresholds, fmr, fnmr), mỗi phần tử là mảng numpy cùng độ dài,
    ngưỡng sắp theo chiều khắt khe dần: FMR giảm dần, FNMR tăng dần.
    Khi thresholds là None, phần tử đầu có FMR = 1, FNMR = 0 và phần tử cuối có FMR = 0.

    Phần chuẩn bị đã viết sẵn bên dưới:
    - với điểm khoảng cách, toàn bộ điểm và ngưỡng được đổi dấu để quy về "chấp nhận khi s >= t";
    - khi thresholds là None, ngưỡng là mọi giá trị điểm phân biệt, thêm một ngưỡng lớn hơn điểm
      lớn nhất để đường cong có điểm FMR = 0.

    Việc của TODO 1:
    1. Sắp xếp điểm cùng người và điểm khác người.
    2. Dùng np.searchsorted để đếm, với mỗi ngưỡng, số cặp cùng người có điểm < t (bị từ chối)
       và số cặp khác người có điểm >= t (được chấp nhận). Chọn tham số side sao cho điểm bằng
       đúng ngưỡng được CHẤP NHẬN; kiểm thử "điểm bằng ngưỡng" sẽ báo nếu chọn sai.
    3. Chia cho số cặp để được FNMR và FMR.
    4. Với điểm khoảng cách, đổi dấu ngưỡng trở lại trước khi trả về.
    Lý do không dùng vòng lặp lồng nhau: với 11.000 điểm và khoảng 11.000 ngưỡng, vòng lặp cần
    khoảng 1,2 x 10^8 phép so sánh, còn cách sắp xếp rồi tìm kiếm nhị phân chạy dưới một giây.
    """
    gen = _as_array(genuine)
    imp = _as_array(impostor)
    if not higher_is_better:
        # đổi dấu để quy mọi trường hợp về "chấp nhận khi s >= t"
        gen, imp = -gen, -imp
        if thresholds is not None:
            thresholds = -np.asarray(thresholds, dtype=np.float64)
    if thresholds is None:
        all_scores = np.unique(np.concatenate([gen, imp]))
        # thêm một ngưỡng lớn hơn mọi điểm để có điểm FMR = 0
        thresholds = np.append(all_scores, all_scores[-1] + 1e-9)
    else:
        thresholds = np.sort(np.asarray(thresholds, dtype=np.float64))

    # TODO 1: đếm bằng np.searchsorted, tính fnmr và fmr, đổi dấu ngưỡng trở lại nếu cần,
    # rồi trả về (thresholds, fmr, fnmr).
    gen_sorted = np.sort(gen)
    imp_sorted = np.sort(imp)

    # side='left': số phần tử < t
    # gen < t là các cặp cùng người bị từ chối -> FNMR
    fnmr = np.searchsorted(gen_sorted, thresholds, side="left") / float(len(gen))

    # imp >= t là các cặp khác người được chấp nhận -> FMR
    # len(imp) - np.searchsorted(imp_sorted, thresholds, side='left')
    fmr = (len(imp) - np.searchsorted(imp_sorted, thresholds, side="left")) / float(len(imp))

    if not higher_is_better:
        thresholds = -thresholds

    return thresholds, fmr, fnmr


def eer(genuine, impostor, higher_is_better=True):
    """Tính tỷ lệ lỗi cân bằng (EER).

    Gọi (thr, fmr, fnmr) là kết quả của error_rates với ngưỡng mặc định, diff = fmr - fnmr
    (giảm dần), i là chỉ số đầu tiên có diff[i] <= 0. Trả về dict gồm:
    - "eer": (fmr[k] + fnmr[k]) / 2 tại k thuộc {i - 1, i} có |diff[k]| nhỏ hơn
      (cùng cách làm của FVC và thư viện pyeer); nếu i = 0 thì k = 0;
    - "eer_interp": nội suy tuyến tính FMR giữa hai ngưỡng kẹp giao điểm:
      w = diff[i-1] / (diff[i-1] - diff[i]);  eer_interp = fmr[i-1] + w * (fmr[i] - fmr[i-1]).
      Nếu i = 0 hoặc diff[i-1] = diff[i] thì eer_interp = eer;
    - "threshold", "fmr", "fnmr": giá trị tại chỉ số k.

    Gợi ý: int(np.argmax(diff <= 0)) cho chỉ số đầu tiên thoả điều kiện.
    """
    # TODO 2: cài đặt hàm này (dùng error_rates ở trên).
    thr, fmr, fnmr = error_rates(genuine, impostor, higher_is_better=higher_is_better)
    diff = fmr - fnmr
    i = int(np.argmax(diff <= 0))
    if i == 0:
        k = 0
    else:
        if abs(diff[i - 1]) < abs(diff[i]):
            k = i - 1
        else:
            k = i

    eer_val = (fmr[k] + fnmr[k]) / 2.0

    if i == 0 or diff[i - 1] == diff[i]:
        eer_interp = eer_val
    else:
        w = diff[i - 1] / (diff[i - 1] - diff[i])
        eer_interp = fmr[i - 1] + w * (fmr[i] - fmr[i - 1])

    return {
        "eer": float(eer_val),
        "eer_interp": float(eer_interp),
        "threshold": float(thr[k]),
        "fmr": float(fmr[k]),
        "fnmr": float(fnmr[k]),
    }


def fnmr_at_fmr(genuine, impostor, target_fmr, higher_is_better=True):
    """FNMR tại ngưỡng dễ dãi nhất vẫn bảo đảm FMR <= target_fmr.

    Trả về (fnmr, threshold, fmr_dat_duoc). Đây là cách ngân hàng hoặc tiêu chuẩn đặt
    yêu cầu: cố định FMR trước (ví dụ 0,01%), rồi hỏi hệ thống từ chối nhầm bao nhiêu.
    """
    # TODO 3: cài đặt hàm này.
    thr, fmr, fnmr = error_rates(genuine, impostor, higher_is_better=higher_is_better)
    idx = int(np.argmax(fmr <= target_fmr))
    return float(fnmr[idx]), float(thr[idx]), float(fmr[idx])


def decidability(genuine, impostor):
    """Chỉ số phân tách d' = |mu_G - mu_I| / sqrt((sigma_G^2 + sigma_I^2) / 2).

    sigma^2 là phương sai tổng thể (population variance, chia cho n): np.var(x) với ddof=0,
    không phải phương sai mẫu ddof=1 (mặc định của pandas .var() và .std()).
    """
    # TODO 4: cài đặt hàm này.
    gen = _as_array(genuine)
    imp = _as_array(impostor)
    mu_g = np.mean(gen)
    mu_i = np.mean(imp)
    var_g = np.var(gen, ddof=0)
    var_i = np.var(imp, ddof=0)
    d_prime = abs(mu_g - mu_i) / np.sqrt((var_g + var_i) / 2.0)
    return float(d_prime)


def roc_auc(fmr, fnmr):
    """Diện tích dưới đường ROC (trục hoành FMR, trục tung TPR = 1 - FNMR).

    Đầu vào là fmr, fnmr do error_rates trả về với ngưỡng mặc định, nên đường cong đã có đủ
    hai đầu mút (FMR, TPR) = (0, 0) và (1, 1). Sắp các điểm theo FMR tăng dần, khi FMR bằng nhau
    thì theo TPR tăng dần (np.lexsort((tpr, fmr))), rồi tích phân TPR theo FMR bằng quy tắc hình
    thang: np.trapezoid(tpr, fmr) (numpy 2.x; np.trapz đã bị gỡ khỏi numpy).
    Vì sao cần sắp cả TPR: nhiều ngưỡng liền nhau có cùng FMR; nếu điểm có TPR thấp đứng cuối
    nhóm, hình thang kế tiếp dùng sai chiều cao và AUC bị tính thiếu.
    """
    # TODO 5: cài đặt hàm này.
    fmr_arr = _as_array(fmr)
    fnmr_arr = _as_array(fnmr)
    tpr = 1.0 - fnmr_arr
    order = np.lexsort((tpr, fmr_arr))
    fmr_sorted = fmr_arr[order]
    tpr_sorted = tpr[order]
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz_fn(tpr_sorted, fmr_sorted))


def fpir_from_fmr(fmr, n_gallery):
    """Xấp xỉ FPIR của tìm kiếm 1:N khi các phép so khớp độc lập: 1 - (1 - FMR)^N."""
    # TODO 6: một dòng lệnh.
    return float(1.0 - (1.0 - float(fmr)) ** n_gallery)


DET_TICKS = [0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
DET_TICK_LABELS = ["0,01", "0,1", "1", "2", "5", "10", "20", "40"]
EER_MARKERS = ["o", "s", "^", "D", "v"]


def _probit(p, lo=1e-5):
    """Biến đổi xác suất sang độ lệch chuẩn hoá (normal deviate) cho trục DET.

    Gợi ý: kẹp p vào [lo, 1 - lo] bằng np.clip để tránh vô cực, rồi dùng norm.ppf.
    Vì sao cần trục này: nếu điểm cùng người và khác người đều gần phân phối chuẩn,
    đường DET gần như thẳng, nên hai hệ thống so sánh với nhau rất rõ ở vùng lỗi nhỏ.
    """
    p_clipped = np.clip(p, lo, 1.0 - lo)
    if norm is not None:
        return norm.ppf(p_clipped)
    # Thuật toán xấp xỉ hữu tỉ Acklam cho hàm phân vị chuẩn nghịch đảo khi không có scipy
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    arr = np.asarray(p_clipped, dtype=np.float64)
    res = np.zeros_like(arr)
    p_low = 0.02425
    p_high = 1.0 - p_low
    mask_low = arr < p_low
    if np.any(mask_low):
        q = np.sqrt(-2.0 * np.log(arr[mask_low]))
        res[mask_low] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    mask_mid = (arr >= p_low) & (arr <= p_high)
    if np.any(mask_mid):
        q = arr[mask_mid] - 0.5
        r = q * q
        res[mask_mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                        (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    mask_high = arr > p_high
    if np.any(mask_high):
        q = np.sqrt(-2.0 * np.log(1.0 - arr[mask_high]))
        res[mask_high] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                          ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    return res.item() if arr.ndim == 0 else res


def plot_det(curves, path, title="Đường cong DET", eer_points=None):
    """Vẽ đường cong DET trên trục chuẩn hoá (normal deviate).

    curves: dict tên -> (fmr, fnmr). eer_points: dict tên -> giá trị EER (tuỳ chọn).
    Mỗi hệ thống dùng một kiểu dấu EER khác nhau để các điểm gần nhau vẫn phân biệt được.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for i, (name, (fmr, fnmr)) in enumerate(curves.items()):
        line, = ax.plot(_probit(fmr), _probit(fnmr), lw=2, label=name)
        if eer_points and name in eer_points:
            e = eer_points[name]
            ax.plot(_probit(e), _probit(e), EER_MARKERS[i % len(EER_MARKERS)],
                    color=line.get_color(), ms=7, mfc="none", mew=1.8)
    ticks = _probit(np.array(DET_TICKS))
    ax.set_xticks(ticks, DET_TICK_LABELS)
    ax.set_yticks(ticks, DET_TICK_LABELS)
    lim = (_probit(0.00005), _probit(0.5))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.plot(lim, lim, ls=":", color="grey", lw=1)
    ax.set_xlabel("FMR (%)")
    ax.set_ylabel("FNMR (%)")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(curves, path, title="Đường cong ROC"):
    """Vẽ ROC với trục hoành FMR theo thang log để nhìn rõ vùng FMR nhỏ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for name, (fmr, fnmr) in curves.items():
        ax.plot(np.clip(fmr, 1e-5, 1), 1 - np.asarray(fnmr), lw=2, label=name)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("FMR (thang log)")
    ax.set_ylabel("1 - FNMR")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distributions(systems, path, title="Phân bố điểm so khớp"):
    """systems: dict tên -> (genuine, impostor). Mỗi hệ thống một ô con.

    Trục tung dùng thang log: một cụm nhỏ cặp khác người điểm cao (vài phần trăm số cặp) gần
    như biến mất trên thang tuyến tính, nhưng chính cụm đó quyết định FNMR ở vùng FMR nhỏ.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(systems)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False)
    for ax, (name, (gen, imp)) in zip(axes[0], systems.items()):
        bins = np.linspace(min(np.min(gen), np.min(imp)), max(np.max(gen), np.max(imp)), 60)
        ax.hist(imp, bins=bins, density=True, alpha=0.55, label="khác người")
        ax.hist(gen, bins=bins, density=True, alpha=0.55, label="cùng người")
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("điểm")
        ax.set_ylabel("mật độ (thang log)")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)