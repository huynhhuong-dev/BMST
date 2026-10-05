# LAB02: Đánh giá Hiệu năng Hệ thống Sinh trắc (FMR, FNMR, EER, DET)

- **Học phần:** 04211 - Bảo mật sinh trắc (Biometric Security)
- **Sinh viên:** Nguyễn Đoàn Huỳnh Hương
- **MSSV:** 2305CT0721

## 1. Nội dung
Tự xây dựng các hàm đo hiệu năng hệ thống sinh trắc học không phụ thuộc thư viện tính sẵn:
- `error_rates(genuine, impostor, thresholds, higher_is_better)`: Tính FMR và FNMR theo ngưỡng bằng thuật toán sắp xếp và tìm kiếm nhị phân `np.searchsorted`.
- `eer(genuine, impostor, higher_is_better)`: Tính tỷ lệ lỗi cân bằng EER và EER nội suy tuyến tính `eer_interp`.
- `fnmr_at_fmr(genuine, impostor, target_fmr, higher_is_better)`: Tính FNMR tại mức bảo mật FMR mục tiêu (ví dụ 1%, 0.1%, 0.01%).
- `decidability(genuine, impostor)`: Tính chỉ số phân tách d' (d-prime) theo phương sai tổng thể.
- `roc_auc(fmr, fnmr)`: Tính diện tích dưới đường cong ROC (AUC) bằng tích phân hình thang `np.trapezoid`.
- `fpir_from_fmr(fmr, n_gallery)`: Xấp xỉ FPIR cho tìm kiếm định danh 1:N.
- `_probit(p)`: Biến đổi xác suất sang độ lệch chuẩn hóa cho trục DET.

## 2. Kết quả kiểm thử
- `python test_bio_metrics.py`: Đạt 21/21 phép kiểm thử đơn vị.
- `python th01_main.py --ma-sv 2305CT0721`: Cả 3 hệ thống A, B, C đều ĐẠT chuẩn với độ lệch EER so với `pyeer` bằng 0.000 điểm phần trăm (sai số cho phép <= 0.5 pp).

## 3. Tệp kết quả
- Bảng số liệu: `ket-qua/TH01_2305CT0721_bang-ket-qua.csv`
- Đồ thị DET: `ket-qua/TH01_2305CT0721_det.png`
- Đồ thị phân bố điểm: `ket-qua/TH01_2305CT0721_phan-bo-diem.png`
- Đồ thị ROC: `ket-qua/TH01_2305CT0721_roc.png`
- Báo cáo phân tích: `bao-cao.md`