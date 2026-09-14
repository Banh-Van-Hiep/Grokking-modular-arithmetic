# Grokking Modular Arithmetic

Nghiên cứu hiện tượng Grokking trong bài toán Modular Arithmetic sử dụng mô hình Transformer.

## 1. Mục tiêu

- Xây dựng mô hình Transformer làm baseline cho bài toán Modular Arithmetic.
- Khảo sát hiện tượng Grokking trong quá trình huấn luyện.
- Nghiên cứu ảnh hưởng của các hyperparameter và cấu hình mô hình đến thời điểm Grokking.
- Làm cơ sở cho việc phân tích representation và Fourier representation.

## 2. Mô hình

Project sử dụng mô hình GPT-2/Transformer làm baseline để nghiên cứu hiện tượng Grokking trên bài toán Modular Arithmetic.

Kiến trúc baseline hiện tại:

- Số layer: `2`
- Hidden dimension: `128`
- Attention heads: `4`
- Feed-forward dimension: `512`
- Context length: `32`
- Dropout: `0`
- Optimizer: `AdamW`
- Learning rate mặc định: `0.001`
- Weight decay: `1.0`
- Betas: `(0.9, 0.98)`
- Gradient clipping: `1.0`
- Warmup steps: `1000`

Các tham số như số layer, số attention head, learning rate và weight decay được thay đổi trong các thực nghiệm để khảo sát ảnh hưởng của chúng đến thời điểm Grokking.

Trong các thực nghiệm, learning rate được khảo sát ở các mức khác nhau, trong đó có `0.001` và `0.01`.

Mô hình được huấn luyện theo dạng language modeling nhưng loss được mask, chỉ tính loss tại vị trí kết quả ngay sau dấu `=`.

Ví dụ:

```text
<SOS> N_12 * N_34 = N_14 <EOS>
```

Với `p = 197`:

```text
12 * 34 mod 197 = 14
```

Mô hình cần dự đoán token `N_14` từ biểu thức phía trước.

## 3. Dữ liệu

### 3.1. Nguồn dữ liệu

Dữ liệu không lấy từ một dataset có sẵn mà được sinh trực tiếp bằng chương trình dựa trên phép toán Modular Arithmetic.

Với phép nhân modulo:

```text
y = (a * b) mod p
```

trong đó:

- `a, b ∈ {0, 1, ..., p-1}`
- `p` là modulo được sử dụng trong thực nghiệm.
- `y` là kết quả cần dự đoán.

Ví dụ với `p = 197`:

```text
12 * 34 mod 197 = 14
```

Kết quả được chuyển thành dạng token:

```text
<SOS> N_12 * N_34 = N_14 <EOS>
```

### 3.2. Các phép toán được hỗ trợ

Code hiện tại hỗ trợ:

- `x * y mod p`
- `x + y mod p`
- `x - y mod p`
- `x² + y² mod p`

Các thực nghiệm hiện tại tập trung vào:

```text
x * y mod p
```

### 3.3. Cách sinh dữ liệu

Với phép toán đối xứng như phép nhân, chương trình trước tiên tạo các cặp:

```text
(a, b), với 0 <= a <= b < p
```

Sau đó các cặp được xáo trộn ngẫu nhiên bằng seed cố định.

Tập dữ liệu được chia thành:

- Train: `TRAIN_FRACTION`
- Validation: phần còn lại

Ví dụ:

```text
p = 197
train_fraction = 0.3
```

Trong trường hợp này, khoảng 30% các cặp được sử dụng để train và phần còn lại được sử dụng để validation.

Đối với phép toán đối xứng, nếu `a != b`, dữ liệu còn tạo thêm biểu thức đảo:

```text
a * b = y
b * a = y
```

nhằm cung cấp cả hai thứ tự biểu diễn cho cùng một phép toán.

### 3.4. Đặc điểm và phạm vi dữ liệu

Dữ liệu có tính chất synthetic và exhaustive theo miền modulo được chọn.

Với một giá trị `p` cố định, các cặp số được xét trên toàn bộ miền:

```text
{0, 1, ..., p-1}
```

Đối với phép nhân, code sử dụng các cặp duy nhất:

```text
(a, b), 0 <= a <= b < p
```

sau đó chia các cặp này thành train và validation.

Do đó, dữ liệu không phải dữ liệu thực tế mà là dữ liệu toán học được sinh theo một quy luật xác định.

### 3.5. Chống Data Leakage

Việc chia dữ liệu được thực hiện ở mức cặp toán học `(a, b)`, trước khi tạo các sample.

Các cặp train và validation là hai tập riêng biệt:

```text
train_pairs ∩ val_pairs = ∅
```

Điều này giúp tránh trường hợp cùng một cặp `(a, b)` xuất hiện đồng thời trong train và validation.

Tuy nhiên, do bài toán Modular Arithmetic có cấu trúc toán học chung, các giá trị `a` và `b` riêng lẻ vẫn có thể xuất hiện ở cả train và validation.

### 3.6. Train, Validation và Test

Phiên bản hiện tại của project mới có:

- Training set: dùng để tối ưu trọng số mô hình.
- Validation set: dùng để theo dõi accuracy và xác định thời điểm Grokking.

Hiện tại chưa xây dựng một test set độc lập hoàn toàn trong pipeline.

## 4. Quy trình thực nghiệm

Quy trình thực nghiệm gồm các bước:

1. Chọn modulo `p`.
2. Sinh toàn bộ các cặp toán học trong miền modulo.
3. Chia các cặp thành training và validation.
4. Chuyển biểu thức toán học thành chuỗi token.
5. Khởi tạo mô hình Transformer.
6. Huấn luyện bằng AdamW.
7. Theo dõi training loss, training accuracy và validation accuracy.
8. Xác định Grokking Step khi validation accuracy đạt ngưỡng yêu cầu.
9. Thay đổi các hyperparameter và lặp lại thực nghiệm.
10. So sánh Grokking Step giữa các cấu hình.

## 5. Kết quả thực nghiệm

Các thực nghiệm được thực hiện bằng cách thay đổi cấu hình mô hình và hyperparameter, sau đó theo dõi Grokking Step – thời điểm validation accuracy đạt mức yêu cầu.

### 5.1. Thực nghiệm với p = 197

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

### 5.2. Thực nghiệm với p = 383

Train fraction: `0.15`

Baseline:

- Layer: `2`
- Hidden: `128`
- Weight decay: `1`
- Head: `4`
- Learning rate: `0.001`
- Grokking Step: `34300`

### 5.3. Đánh giá ban đầu

Các thực nghiệm cho thấy thời điểm Grokking thay đổi đáng kể khi thay đổi cấu hình mô hình và hyperparameter.

Đặc biệt, weight decay có ảnh hưởng rõ rệt đến thời điểm Grokking. Với một số giá trị weight decay, mô hình Grokking nhanh hơn; trong khi một số cấu hình không đạt Grokking trong thời gian thực nghiệm.

Ví dụ với `p = 197`, Grokking Step thay đổi từ khoảng `1200` đến `3200` tùy cấu hình, và một số cấu hình chưa xảy ra Grokking.

Kết quả với `p = 383` cho thấy baseline đạt Grokking ở step `34300`.

Các kết quả này là cơ sở để tiếp tục nghiên cứu cơ chế Grokking và sự thay đổi representation của mô hình.

## 6. Những điểm còn thiếu

Phiên bản hiện tại vẫn còn một số điểm cần tiếp tục hoàn thiện:

- Xây dựng test set độc lập với validation set.
- Thực hiện nhiều seed để kiểm tra độ ổn định của kết quả.
- Mở rộng phạm vi các giá trị `p`.
- Khảo sát hệ thống hơn ảnh hưởng của train fraction.
- Khảo sát đầy đủ hơn ảnh hưởng của model size.
- Phân tích representation của model trước và sau Grokking.
- Phân tích Fourier representation.
- So sánh kết quả giữa các phép toán Modular Arithmetic khác nhau.
- Lưu checkpoint và kết quả của từng lần chạy theo cấu hình.

## 7. Cấu trúc project

```text
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
├── .gitignore
├── README.md
├── TODO.md
└── requirements.txt
```

## 8. Cài đặt

Cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install -r requirements.txt
```

## 9. Chạy thực nghiệm

Từ thư mục gốc của project:

```bash
python -m experiments.multiplication
```

Các tham số thực nghiệm có thể được điều chỉnh trong:

```text
configs/config.py
```

## 10. Hướng phát triển

- Phân tích representation trước và sau Grokking.
- Nghiên cứu Fourier representation.
- Phân tích sâu hơn ảnh hưởng của hyperparameter.
- Mở rộng số lượng và phạm vi thực nghiệm.