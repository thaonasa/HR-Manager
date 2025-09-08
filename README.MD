🔎 Quy trình ra đề xuất

Dữ liệu gốc (từ file CSV bạn đưa) gồm các cột:

Thông tin cá nhân/công việc: employee_id, employee_name, role, appointed_date, tenure_months

Điểm đánh giá: hr_score, bod_score, overall_score, compliance_score

Kết quả KPI: kpi_2024, kpi_2025

Vi phạm: violations_2022, violations_2023, violations_2024, violations_total

Thông tin cửa hàng: store_id, store_name, store_kpi24_mean, store_violations_mean

Tiền xử lý (Preprocessing)

Chuẩn hóa dữ liệu (điền thiếu bằng median, mã hóa role nếu cần).

Tính thêm đặc trưng cho ứng viên như: mức KPI trung bình cửa hàng, số vi phạm trung bình, thời gian công tác, v.v.

Mô hình Random Forest (rf_bal_model.joblib)

Được train để dự đoán xác suất thành công khi nhân sự chuyển tới cửa hàng mục tiêu.

Input: vector đặc trưng của nhân sự + đặc trưng cửa hàng đích.

Output: proba = rf.predict_proba(X)[:, 1] → xác suất phù hợp.

Xếp hạng & chọn Top N

Với mỗi ứng viên, mô hình cho điểm success_proba.

Hệ thống sắp xếp giảm dần và chọn Top N nhân sự có xác suất cao nhất.

📊 Ví dụ (trích từ log)

Khi bạn gõ:

gợi ý cho cửa hàng 302


Backend tìm dòng dữ liệu của store_id=302 (Bkr 130 Tiểu La).

Lấy toàn bộ nhân sự khác cửa hàng 302 làm ứng viên.

Tạo vector đặc trưng cho từng ứng viên (KPI, vi phạm, điểm HR/BOD, thâm niên...).

Cho mô hình Random Forest dự đoán.

Trả về Top 3 nhân sự có success_proba cao nhất.

✅ Tóm lại

Kết quả đề xuất dựa vào:

Điểm KPI (hiện tại + trung bình cửa hàng)

Điểm HR/BOD/Overall

Số lần vi phạm

Thời gian làm việc (tenure)

Role (chức vụ)

Mức độ tuân thủ (compliance_score)

Đặc điểm cửa hàng đích

→ Mô hình ML tính toán và chọn ra những nhân sự có khả năng phù hợp/cống hiến tốt nhất cho cửa hàng đích.