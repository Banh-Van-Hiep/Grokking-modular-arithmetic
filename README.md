# Grokking Modular Arithmetic

Nghiên cứu hiện tượng Grokking trong bài toán Modular Arithmetic sử dụng mô hình Transformer.

## PHẦN I — GROKKING MODULAR ARITHMETIC

### 1. Mục tiêu

- Xây dựng mô hình Transformer làm baseline cho bài toán Modular Arithmetic.
- Khảo sát hiện tượng Grokking trong quá trình huấn luyện.
- Nghiên cứu ảnh hưởng của các hyperparameter và cấu hình mô hình đến thời điểm Grokking.
- Làm cơ sở cho việc phân tích representation và Fourier representation.

### 2. Mô hình

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

Trong các thực nghiệm, learning rate được khảo sát ở các mức `0.001`.

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

### 3. Dữ liệu

#### 3.1. Nguồn dữ liệu

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

#### 3.2. Các phép toán được hỗ trợ

Code hiện tại hỗ trợ:

- `x * y mod p`
- `x + y mod p`
- `x - y mod p`
- `x² + y² mod p`

Các thực nghiệm hiện tại tập trung vào:

```text
x * y mod p
```

#### 3.3. Cách sinh dữ liệu

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

#### 3.4. Đặc điểm và phạm vi dữ liệu

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

#### 3.5. Chống Data Leakage

Việc chia dữ liệu được thực hiện ở mức cặp toán học `(a, b)`, trước khi tạo các sample.

Các cặp train và validation là hai tập riêng biệt:

```text
train_pairs ∩ val_pairs = ∅
```

Điều này giúp tránh trường hợp cùng một cặp `(a, b)` xuất hiện đồng thời trong train và validation.

Tuy nhiên, do bài toán Modular Arithmetic có cấu trúc toán học chung, các giá trị `a` và `b` riêng lẻ vẫn có thể xuất hiện ở cả train và validation.

#### 3.6. Train, Validation và Test

Phiên bản hiện tại của project mới có:

- Training set: dùng để tối ưu trọng số mô hình.
- Validation set: dùng để theo dõi accuracy và xác định thời điểm Grokking.

Hiện tại chưa xây dựng một test set độc lập hoàn toàn trong pipeline.

### 4. Quy trình thực nghiệm

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

### 5. Kết quả thực nghiệm

Các thực nghiệm được thực hiện bằng cách thay đổi cấu hình mô hình và hyperparameter, sau đó theo dõi Grokking Step – thời điểm validation accuracy đạt mức yêu cầu. Các hình ảnh kết quả đầy đủ được lưu tại thư mục [results/](results/).

#### 5.1. Thực nghiệm với p = 197

Train fraction: `0.3`

Một số cấu hình:

| Cấu hình | Layer | Hidden | Weight Decay | Head | Learning Rate | Grokking Step |
|----------|------:|-------:|-------------:|-----:|--------------:|--------------:|
| Base | 2 | 128 | 1 | 4 | 0.001 | 2600 |
| Scale | 4 | 128 | 1 | 4 | 0.001 | 2200 |
| Scale | 2 | 128 | 1 | 8 | 0.001 | 2600 |
| Scale | 2 | 128 | 0.5 | 4 | 0.001 | 3200 |
| Scale | 2 | 128 | 1.5 | 4 | 0.001 | 1300 |
| Scale | 2 | 128 | 0.1 | 4 | 0.001 | Không Grokking |
| Scale | 2 | 128 | 2 | 4 | 0.001 | 1200 |
| Scale | 2 | 128 | 5 | 4 | 0.001 | 1600 |
| Scale | 2 | 128 | 10 | 4 | 0.001 | Chưa Grokking |

Các cấu hình tương tự cũng được chạy với learning rate `0.001`.

#### 5.2. Thực nghiệm với p = 383

Train fraction: `0.15`

Baseline:

- Layer: `2`
- Hidden: `128`
- Weight decay: `1`
- Head: `4`
- Learning rate: `0.001`
- Grokking Step: `34300`

#### 5.3. Đánh giá ban đầu

Các thực nghiệm cho thấy thời điểm Grokking thay đổi đáng kể khi thay đổi cấu hình mô hình và hyperparameter.

Các kết quả ban đầu cho thấy thời điểm Grokking thay đổi khi thay đổi weight decay trong các cấu hình được khảo sát.

Ví dụ với `p = 197`, Grokking Step thay đổi từ khoảng `1200` đến `3200` tùy cấu hình, và một số cấu hình chưa xảy ra Grokking.

Kết quả với `p = 383` cho thấy baseline đạt Grokking ở step `34300`.

Các kết quả này là cơ sở để tiếp tục nghiên cứu cơ chế Grokking và sự thay đổi representation của mô hình.

### 6. Những điểm còn thiếu

Phiên bản hiện tại vẫn còn một số điểm cần tiếp tục hoàn thiện:

- Xây dựng test set độc lập với validation set.
- Thực hiện nhiều seed để kiểm tra độ ổn định của kết quả.
- Mở rộng phạm vi các giá trị `p`, và các tham số model.
- Phân tích representation của model trước và sau Grokking.
- So sánh kết quả giữa các phép toán Modular Arithmetic khác nhau.

### 7. Cấu trúc project

```text
Grokking-modular-arithmetic/

├── configs/
│   └── config.py
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── experiments/
│   └── multiplication.py
│
├── notebooks/
├── results/
├── .gitignore
├── README.md
├── TODO.md
└── requirements.txt
```

### 8. Cài đặt

Cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install -r requirements.txt
```

### 9. Chạy thực nghiệm

Từ thư mục gốc của project:

```bash
python -m experiments.multiplication
```

Các tham số thực nghiệm có thể được điều chỉnh trong:

```text
configs/config.py
```

### 10. Hướng phát triển

- Phân tích representation trước và sau Grokking.
- Nghiên cứu Fourier representation.
- Phân tích sâu hơn ảnh hưởng của hyperparameter.
- Mở rộng số lượng và phạm vi thực nghiệm.

---

# PHẦN II — IN-CONTEXT LEARNING (ICL)

Nghiên cứu khả năng In-Context Learning (ICL) của Transformer trên bài toán Modular Arithmetic.

## 11. Mục tiêu

- Kiểm tra khả năng In-Context Learning (ICL) của Transformer trên bài toán Modular Arithmetic.
- Khảo sát sự thay đổi accuracy khi số lượng examples trong context tăng.
- Nghiên cứu ảnh hưởng của các hyperparameter và cấu hình mô hình đến khả năng ICL.
- Làm cơ sở cho việc nghiên cứu mối quan hệ giữa Grokking và In-Context Learning.

## 12. Mô hình

Phần ICL sử dụng mô hình Transformer với Rotary Positional Embedding (RoPE) để thực hiện bài toán Modular Arithmetic có cấu trúc task.

Kiến trúc baseline hiện tại:

- Số layer: `2`
- Hidden dimension: `128`
- Attention heads: `4`
- Feed-forward dimension: `512`
- Context length: `32`
- Optimizer: `AdamW`
- Learning rate: `1.5e-4`
- Weight decay: `2.0`
- Betas: `(0.9, 0.98)`
- Gradient clipping: `1.0`
- Effective batch size: `1024`
- Training steps: `80000`
- Seed: `42`

Các tham số như số layer, hidden dimension, learning rate, weight decay và modulo `P` được thay đổi trong các thực nghiệm để khảo sát ảnh hưởng của chúng đến khả năng ICL.

Mô hình được huấn luyện để dự đoán output `z` của input `(x, y)` theo rule `(a, b)`:

```text
z = (a * x + b * y) mod P
```

Loss chỉ được tính tại các vị trí output `z`.

## 13. Dữ liệu

### 13.1. Bài toán

Mỗi task được xác định bởi một cặp hệ số:

```text
(a, b)
```

Với input:

```text
(x, y)
```

mô hình cần dự đoán:

```text
z = (a * x + b * y) mod P
```

Ví dụ với `P = 17`:

```text
a = 3
b = 5
x = 4
y = 2

z = (3 * 4 + 5 * 2) mod 17
  = 5
```

### 13.2. Chia dữ liệu

Với một giá trị `P` cố định:

```text
P² rules
P² input pairs
```

Rules được chia thành:

```text
Train rules = rule_frac × P²
Test rules  = phần còn lại
```

Input pairs được chia thành:

```text
Train inputs = input_frac × P²
Test inputs  = phần còn lại
```

Cấu hình chính:

```text
P = 17
rule_frac = 0.8
input_frac = 0.8
seed = 42
```

Với `P = 17`:

```text
Train rules = 231
Test rules = 58

Train inputs = 231
Test inputs = 58
```

Các rules và input pairs được xáo trộn bằng seed cố định để đảm bảo khả năng tái lập thực nghiệm.

### 13.3. Structured Task Sampling

Training sử dụng các nhóm task có cấu trúc dạng rectangle.

Một rectangle gồm 4 task:

```text
(a, b)
(a + Δa, b)
(a, b + Δb)
(a + Δa, b + Δb)
```

Các giá trị được tính theo modulo `P`.

Bốn task trong cùng một rectangle sử dụng cùng sequence input `(x, y)` nhưng có output `z` khác nhau theo rule tương ứng.

Với cấu hình:

```text
P = 17
rule_frac = 0.8
```

số rectangle được tạo là:

```text
7369
```

### 13.4. Đặc điểm dữ liệu

Dữ liệu có tính chất synthetic và được sinh trực tiếp từ quy luật toán học.

Các rules và input pairs được xáo trộn bằng seed cố định để đảm bảo khả năng tái lập thực nghiệm.

Trong quá trình huấn luyện, các task được lấy theo các rectangle để tạo ra cấu trúc liên hệ giữa các task.

## 14. Điều kiện đánh giá

Mô hình được đánh giá trên 4 điều kiện:

| Condition | Rules | Inputs |
|----------|-------|--------|
| `S_train_id` | Train | Train |
| `S_test_id` | Train | Test |
| `S_train_ood` | Test | Train |
| `S_test_ood` | Test | Test |

Trong đó:

- `S_train_id`: rule và input đều thuộc tập training.
- `S_test_id`: rule thuộc training nhưng input thuộc test.
- `S_train_ood`: rule thuộc test nhưng input thuộc training.
- `S_test_ood`: cả rule và input đều thuộc test.

Các điều kiện này được sử dụng để theo dõi khả năng của mô hình khi rule hoặc input chưa xuất hiện trong quá trình huấn luyện.

## 15. Đánh giá theo số lượng shots

Khả năng ICL được đánh giá bằng cách thay đổi số lượng examples trong context.

Các mức shots được sử dụng:

```text
0, 1, 2, 4, 8, 16, 31
```

Accuracy được tính theo từng vị trí trong sequence.

Ngoài accuracy tại từng số lượng shots, code sử dụng `late accuracy`, là accuracy trung bình trên nửa sau của sequence.

Với `context length = 32`:

```text
late accuracy = accuracy trung bình từ shot 16 đến shot 31
```

Chance level được tính:

```text
1 / P
```

Với:

```text
P = 17
```

chance level:

```text
5.9%
```

Với:

```text
P = 29
```

chance level:

```text
3.4%
```

Việc tăng accuracy khi số lượng examples trong context tăng được sử dụng như một tín hiệu để quan sát khả năng sử dụng thông tin trong context.

## 16. Quy trình thực nghiệm

Quy trình thực nghiệm gồm các bước:

1. Chọn modulo `P`.
2. Sinh toàn bộ rules và input pairs.
3. Chia rules và input pairs thành training và test.
4. Xây dựng các rectangle từ training rules.
5. Sinh sequence cho các task.
6. Khởi tạo mô hình Transformer.
7. Huấn luyện bằng AdamW.
8. Theo dõi accuracy trên các điều kiện đánh giá.
9. Đánh giá accuracy theo số lượng shots.
10. Thay đổi từng hyperparameter và lặp lại thực nghiệm.
11. So sánh kết quả giữa các cấu hình.

## 17. Kết quả thực nghiệm

Các thực nghiệm được thực hiện bằng cách thay đổi từng hyperparameter trong khi giữ các tham số còn lại theo baseline.

Các hình ảnh và file kết quả chi tiết được lưu tại:

[results/icl/](results/icl/)

### 17.1. Baseline

Cấu hình baseline:

| Tham số | Giá trị |
|---------|---------|
| `P` | `17` |
| `rule_frac` | `0.8` |
| `input_frac` | `0.8` |
| Layers | `2` |
| Embedding | `128` |
| Heads | `4` |
| Context | `32` |
| Learning rate | `1.5e-4` |
| Weight decay | `2.0` |
| Training steps | `80000` |
| Seed | `42` |

Kết quả baseline:

[Baseline](results/icl/baseline.png)

Các metric chính được theo dõi:

- `S_train_id` late accuracy
- `S_train_ood` late accuracy
- `S_test_ood` late accuracy
- `S_test_ood` accuracy tại `0-shot`
- `S_test_ood` accuracy tại `31-shot`
- Best step

`Best step` là training step tại đó `S_test_ood` late accuracy đạt giá trị cao nhất trong quá trình đánh giá.

### 17.2. Thực nghiệm với Weight Decay

Baseline:

```text
wd = 2.0
```

Các cấu hình được khảo sát:

```text
wd = 1.0
wd = 3.0
```

Kết quả:

| Cấu hình | Weight Decay | Best Step | `S_train_id` Late | `S_train_ood` Late | `S_test_ood` Late |
|----------|-------------:|----------:|------------------:|-------------------:|------------------:|
| WD 1 | `1.0` | `54000` | `45.35%` | `29.28%` | `11.57%` |
| WD 3 | `3.0` | `70000` | `25.16%` | `17.04%` | `14.31%` |

Kết quả chi tiết:

- [Weight Decay = 1](results/icl/wd1.png)
- [Weight Decay = 3](results/icl/wd3.png)

### 17.3. Thực nghiệm với Learning Rate

Baseline:

```text
lr = 1.5e-4
```

Các cấu hình được khảo sát:

```text
lr = 1e-4
lr = 3e-4
```

Kết quả:

| Cấu hình | Learning Rate | Best Step | `S_train_id` Late | `S_train_ood` Late | `S_test_ood` Late |
|----------|--------------:|----------:|------------------:|-------------------:|------------------:|
| LR 1e-4 | `0.0001` | `27000` | `29.31%` | `19.17%` | `14.22%` |
| LR 3e-4 | `0.0003` | `22500` | `61.94%` | `55.93%` | `20.17%` |

Kết quả chi tiết:

- [Learning Rate = 1e-4](results/icl/lr1e-4.png)
- [Learning Rate = 3e-4](results/icl/lr3e-4.png)

### 17.4. Thực nghiệm với Embedding Dimension

Baseline:

```text
embd = 128
```

Các cấu hình được khảo sát:

```text
embd = 64
embd = 256
```

Kết quả:

| Cấu hình | Embedding | Best Step | `S_train_id` Late | `S_train_ood` Late | `S_test_ood` Late |
|----------|----------:|----------:|------------------:|-------------------:|------------------:|
| Embd 64 | `64` | `54500` | `20.95%` | `15.52%` | `10.71%` |
| Embd 256 | `256` | `10000` | `59.57%` | `53.50%` | `27.39%` |

Kết quả chi tiết:

- [Embedding = 64](results/icl/embd64.png)
- [Embedding = 256](results/icl/embd256.png)

### 17.5. Thực nghiệm với số Layer

Baseline:

```text
layers = 2
```

Các cấu hình được khảo sát:

```text
layers = 4
layers = 6
```

Kết quả:

| Cấu hình | Layers | Best Step | `S_train_id` Late | `S_train_ood` Late | `S_test_ood` Late |
|----------|-------:|----------:|------------------:|-------------------:|------------------:|
| Layers 4 | `4` | `63000` | `97.39%` | `91.64%` | `88.90%` |
| Layers 6 | `6` | `61500` | `97.78%` | `94.25%` | `89.75%` |

Kết quả chi tiết:

- [Layers = 4](results/icl/layers4.png)
- [Layers = 6](results/icl/layers6.png)

### 17.6. Thực nghiệm với modulo P

Baseline:

```text
P = 17
```

Thực nghiệm mở rộng:

```text
P = 29
```

Kết quả:

| Tham số | Giá trị |
|---------|--------:|
| `P` | `29` |
| Layers | `2` |
| Embedding | `128` |
| Context | `32` |
| Rule fraction | `0.8` |
| Input fraction | `0.8` |
| Learning rate | `1.5e-4` |
| Weight decay | `2.0` |
| Best Step | `34500` |
| `S_train_id` Late | `4.16%` |
| `S_train_ood` Late | `2.05%` |
| `S_test_ood` Late | `3.56%` |
| Chance | `3.4%` |
| Steps done | `53500` |
| Status | `stuck_stop` |

Kết quả chi tiết:

[P = 29](results/icl/P29.png)

## 18. Đánh giá ban đầu

Các thực nghiệm cho thấy accuracy thay đổi theo số lượng examples trong context ở nhiều cấu hình.

Đối với `P = 17`, một số cấu hình cho thấy accuracy trên `S_test_ood` tăng khi số lượng shots tăng.

Với `embd = 256`, accuracy `S_test_ood` tăng từ:

```text
5.27% ở 0-shot
```

lên:

```text
28.12% ở 31-shot
```

Với `layers = 4`, accuracy `S_test_ood` tăng từ:

```text
5.86% ở 0-shot
```

lên:

```text
92.58% ở 31-shot
```

Với `layers = 6`, accuracy `S_test_ood` tăng từ:

```text
7.03% ở 0-shot
```

lên:

```text
93.36% ở 31-shot
```

Đối với `P = 29`, accuracy `S_test_ood` vẫn gần mức chance `3.4%` ở các mức shots được đánh giá.

Các kết quả cho thấy khả năng sử dụng thông tin trong context thay đổi theo cấu hình mô hình, hyperparameter và modulo `P`.

## 19. Cấu trúc project

```text
Grokking-modular-arithmetic/

├── configs/
│   ├── config.py
│   └── icl_config.py
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── icl_data.py
│   ├── icl_model.py
│   ├── icl_train.py
│   ├── icl_evaluate.py
│   └── icl_utils.py
│
├── experiments/
│   ├── multiplication.py
│   └── icl.py
│
├── notebooks/
├── results/
│   ├── ...
│   └── icl/
│       ├── baseline.png
│       ├── wd1.png
│       ├── wd3.png
│       ├── lr1e-4.png
│       ├── lr3e-4.png
│       ├── embd64.png
│       ├── embd256.png
│       ├── layers4.png
│       ├── layers6.png
│       └── P29.png
│
├── .gitignore
├── README.md
├── TODO.md
└── requirements.txt
```

## 20. Cài đặt

Cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install -r requirements.txt
```

## 21. Chạy thực nghiệm

Từ thư mục gốc của project:

```bash
python -m experiments.icl
```