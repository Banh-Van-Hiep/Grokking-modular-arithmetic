# Grokking Modular Arithmetic

Nghiên cứu hiện tượng **Grokking** trong bài toán **Modular Arithmetic** sử dụng mô hình Transformer.

## 1. Mục tiêu

- Xây dựng mô hình Transformer làm baseline cho bài toán Modular Arithmetic.
- Khảo sát hiện tượng Grokking trong quá trình huấn luyện.
- Nghiên cứu ảnh hưởng của các hyperparameter và cấu hình mô hình đến thời điểm Grokking.
- Làm cơ sở cho việc phân tích representation và Fourier representation.

## 2. Mô hình

Project sử dụng kiến trúc **GPT-2/Transformer** làm baseline.

Các tham số được thay đổi trong thực nghiệm gồm:

- Số layer
- Chiều ẩn (hidden dimension)
- Số attention head
- Learning rate
- Weight decay

Mô hình được huấn luyện bằng **AdamW** và theo dõi training loss, training accuracy và validation accuracy.

## 3. Dữ liệu

Bài toán sử dụng Modular Arithmetic với các số nguyên trong modulo `p`.

Mô hình nhận đầu vào dạng:

    a * b =

và học cách dự đoán kết quả:

    (a * b) mod p

Dữ liệu được sinh từ các cặp số trong miền modulo và được chia thành tập train và validation.

Các phép toán được hỗ trợ trong code:

- `x + y mod p`
- `x - y mod p`
- `x * y mod p`
- `x² + y² mod p`

Thực nghiệm baseline hiện tại tập trung vào phép nhân modulo.

## 4. Thực nghiệm

Các thực nghiệm được thực hiện bằng cách thay đổi cấu hình mô hình và hyperparameter, sau đó theo dõi **Grokking Step** – thời điểm validation accuracy đạt mức yêu cầu.

### Thực nghiệm với p = 197

Train fraction: `0.3`

Một số cấu hình:

| Cấu hình | Layer | Hidden | Weight Decay | Head | Learning Rate | Grokking Step |
|----------|------:|-------:|-------------:|-----:|--------------:|--------------:|
| Base | 2 | 128 | 1 | 4 | 0.01 | 2600 |
| Scale | 4 | 128 | 1 | 4 | 0.01 | 2200 |
| Scale | 2 | 128 | 1 | 8 | 0.01 | 2600 |
| Scale | 2 | 128 | 0.5 | 4 | 0.01 | 3200 |
| Scale | 2 | 128 | 1.5 | 4 | 0.01 | 1300 |
| Scale | 2 | 128 | 0.1 | 4 | 0.01 | Không Grokking |
| Scale | 2 | 128 | 2 | 4 | 0.01 | 1200 |
| Scale | 2 | 128 | 5 | 4 | 0.01 | 1600 |
| Scale | 2 | 128 | 10 | 4 | 0.01 | Chưa Grokking |

Các cấu hình tương tự cũng được chạy với learning rate `0.001`.

### Thực nghiệm với p = 383

Train fraction: `0.15`

Baseline:

- Layer: `2`
- Hidden: `128`
- Weight decay: `1`
- Head: `4`
- Learning rate: `0.001`
- Grokking Step: `34300`

## 5. Kết quả ban đầu

Các thực nghiệm cho thấy thời điểm Grokking thay đổi đáng kể khi thay đổi cấu hình mô hình và hyperparameter.

Đặc biệt, **weight decay có ảnh hưởng rõ rệt đến thời điểm Grokking**. Với một số giá trị weight decay, mô hình Grokking nhanh hơn; trong khi một số cấu hình không đạt Grokking trong thời gian thực nghiệm.

Ví dụ với `p = 197`, Grokking Step thay đổi từ khoảng `1200` đến `3200` tùy cấu hình, và một số cấu hình chưa xảy ra Grokking. 

Kết quả với `p = 383` cho thấy baseline đạt Grokking ở step `34300`.

Các kết quả này là cơ sở để tiếp tục nghiên cứu cơ chế Grokking và sự thay đổi representation của mô hình.

## 6. Cấu trúc project

    Grokking-modular-arithmetic/
    ├── configs/
    │   └── config.py
    ├── src/
    │   ├── data.py
    │   ├── model.py
    │   ├── train.py
    │   └── evaluate.py
    ├── experiments/
    │   └── multiplication.py
    ├── notebooks/
    ├── results/
    ├── README.md
    ├── TODO.md
    └── requirements.txt

## 7. Cài đặt

    pip install -r requirements.txt

## 8. Chạy thực nghiệm

Từ thư mục gốc của project:

    python -m experiments.multiplication

Các tham số thực nghiệm có thể được điều chỉnh trong:

    configs/config.py

## 9. Hướng phát triển

- Phân tích representation trước và sau Grokking.
- Nghiên cứu Fourier representation.
- Phân tích sâu hơn ảnh hưởng của hyperparameter.
- Mở rộng số lượng và phạm vi thực nghiệm.