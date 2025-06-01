import torch
from kan import KAN
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler as SparkStandardScaler
from pyspark.ml.feature import VectorAssembler
import time
import os

# --- 1. KHỞI TẠO SPARK SESSION ---
# Cấu hình Spark: Tăng bộ nhớ executor và driver, và cho phép Arrow để tối ưu chuyển đổi dữ liệu.
# Số lượng executor có thể được điều chỉnh tùy theo tài nguyên cluster.
spark = SparkSession.builder \
    .appName("Mini_KAN_Distributed_Example") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

sc = spark.sparkContext
print(f"Spark Version: {spark.version}")
print(f"PySpark Session Active: {spark.sparkContext.appName}")

# --- CẤU HÌNH CHUNG CHO KAN (PyTorch) ---
torch.set_default_dtype(torch.float32)  # float32 thường tốt hơn cho DL và GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for KAN operations: {device}")

# --- 2. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU BẰNG PYSPARK ---
# (Sử dụng dữ liệu mẫu hoặc file CSV của bạn)
# Nếu file CSV quá lớn, Spark sẽ tự động xử lý phân tán.
try:
    df_spark = spark.read.csv('data/processed_country_data.csv', header=True, inferSchema=True)
except Exception as e:
    print(f"Lỗi khi đọc file CSV: {e}. Tạo dữ liệu giả định để minh họa.")
    # Tạo DataFrame giả định nếu file không tồn tại, để code có thể chạy.
    num_records = 10000  # Số lượng bản ghi lớn để thấy hiệu quả phân tán
    data_dummy = {
        'new_cases_smoothed_lag_5': np.random.rand(num_records) * 100,
        'new_cases_smoothed_lag_3': np.random.rand(num_records) * 100,
        'new_cases_smoothed_lag_1': np.random.rand(num_records) * 100,
        'new_cases_smoothed_lag_4': np.random.rand(num_records) * 100,
        'new_cases_smoothed_lag_2': np.random.rand(num_records) * 100,
        'new_deaths_smoothed_lag_5': np.random.rand(num_records) * 10,
        'new_deaths_smoothed_lag_3': np.random.rand(num_records) * 10,
        'new_deaths_smoothed_lag_1': np.random.rand(num_records) * 10,
        'total_deaths_lag_5': np.random.rand(num_records) * 1000,
        'total_deaths_lag_3': np.random.rand(num_records) * 1000,
        'total_deaths_lag_1': np.random.rand(num_records) * 1000,
        'population': np.random.rand(num_records) * 1e7,
        'new_deaths_smoothed_lag_4': np.random.rand(num_records) * 10,
        'new_deaths_smoothed_lag_2': np.random.rand(num_records) * 10,
        'new_cases_next_day': np.random.rand(num_records) * 50,
        'continent': ['Asia'] * num_records,
        'location': np.random.choice(['Vietnam', 'Thailand', 'China', 'Indonesia', 'Malaysia'], num_records).tolist()
    }
    df_spark = spark.createDataFrame(pd.DataFrame(data_dummy))

selected_features = [
    'new_cases_smoothed_lag_5', 'new_cases_smoothed_lag_3', 'new_cases_smoothed_lag_1',
    'new_cases_smoothed_lag_4', 'new_cases_smoothed_lag_2', 'new_deaths_smoothed_lag_5',
    'new_deaths_smoothed_lag_3', 'new_deaths_smoothed_lag_1', 'total_deaths_lag_5',
    'total_deaths_lag_3', 'total_deaths_lag_1', 'population',
    'new_deaths_smoothed_lag_4', 'new_deaths_smoothed_lag_2'
]
target_feature = 'new_cases_next_day'
countries_near_vietnam = [
    'Vietnam', 'Thailand', 'Laos', 'Cambodia', 'China', 'Malaysia',
    'Singapore', 'Indonesia', 'Philippines', 'Brunei', 'South Korea', 'Japan'
]

# Lọc dữ liệu bằng PySpark DataFrame API (Phân tán)
df_filtered = df_spark.filter(
    (df_spark['continent'] == 'Asia') &
    (df_spark['location'].isin(countries_near_vietnam))
)

# Chia dữ liệu huấn luyện và kiểm tra bằng PySpark (Phân tán)
train_df, test_df = df_filtered.randomSplit([0.8, 0.2], seed=42)

# Chuẩn hóa dữ liệu bằng Spark MLlib StandardScaler (Phân tán)
# Gom các cột features và target thành Vector
assembler_X = VectorAssembler(inputCols=selected_features, outputCol="features_vector")
train_features_df = assembler_X.transform(train_df)
test_features_df = assembler_X.transform(test_df)

scaler_X_spark = SparkStandardScaler(inputCol="features_vector", outputCol="scaled_features", withStd=True,
                                     withMean=True)
scaler_X_model = scaler_X_spark.fit(train_features_df)  # Fit phân tán

train_scaled_X_df = scaler_X_model.transform(train_features_df)  # Transform phân tán
test_scaled_X_df = scaler_X_model.transform(test_features_df)  # Transform phân tán

assembler_y = VectorAssembler(inputCols=[target_feature], outputCol="label_vector")
train_label_df = assembler_y.transform(train_df)
test_label_df = assembler_y.transform(test_df)

scaler_y_spark = SparkStandardScaler(inputCol="label_vector", outputCol="scaled_label", withStd=True, withMean=True)
scaler_y_model = scaler_y_spark.fit(train_label_df)  # Fit phân tán

train_scaled_y_df = scaler_y_model.transform(train_label_df)  # Transform phân tán
test_scaled_y_df = scaler_y_model.transform(test_label_df)  # Transform phân tán

# Kết hợp các cột đã chuẩn hóa
train_final_df = train_scaled_X_df.withColumnRenamed("scaled_features", "input_features") \
    .withColumnRenamed(target_feature, "original_label_X") \
    .join(
    train_scaled_y_df.select(target_feature, "scaled_label").withColumnRenamed(target_feature, "original_label_Y"),
    on=(train_scaled_X_df[target_feature] == train_scaled_y_df[target_feature]), how="inner") \
    .select("input_features", "scaled_label")

test_final_df = test_scaled_X_df.withColumnRenamed("scaled_features", "input_features") \
    .withColumnRenamed(target_feature, "original_label_X") \
    .join(test_scaled_y_df.select(target_feature, "scaled_label").withColumnRenamed(target_feature, "original_label_Y"),
          on=(test_scaled_X_df[target_feature] == test_scaled_y_df[target_feature]), how="inner") \
    .select("input_features", "scaled_label")

# Cache DataFrames để tối ưu hiệu suất cho các tác vụ lặp (như huấn luyện)
train_final_df.cache()
test_final_df.cache()

print(f"\nSố bản ghi dữ liệu Train sau tiền xử lý: {train_final_df.count()}")
print(f"Số bản ghi dữ liệu Test sau tiền xử lý: {test_final_df.count()}")


# --- 3. HUẤN LUYỆN KAN CỤC BỘ TRÊN MỖI WORKER (VÀ THU THẬP METRICS) ---

def train_kan_on_partition(partition_id, iterator):
    """
    Hàm này sẽ chạy trên mỗi phân vùng của dữ liệu.
    Mỗi worker sẽ huấn luyện một mô hình KAN cục bộ.
    """
    # Khởi tạo lại thiết bị trong mỗi tiến trình worker
    current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Worker {partition_id}: Sử dụng thiết bị {current_device}")

    rows = list(iterator)
    if not rows:
        return []  # Trả về rỗng nếu phân vùng trống

    # Chuyển đổi dữ liệu từ Spark Row sang PyTorch Tensors
    local_X = torch.tensor([row.input_features.toArray() for row in rows], dtype=torch.float32).to(current_device)
    local_y = torch.tensor([row.scaled_label.toArray() for row in rows], dtype=torch.float32).to(current_device)

    # Chuẩn bị dataset cục bộ cho KAN
    local_dataset = {
        'train_input': local_X,
        'train_label': local_y,
        # Đối với mục đích đánh giá cục bộ, có thể dùng lại dữ liệu train cho test_loss
        # Trong thực tế, bạn sẽ cần một chiến lược để truyền test_data nhỏ tới worker
        'test_input': local_X,
        'test_label': local_y
    }

    # Định nghĩa mô hình KAN (phải được định nghĩa lại trên mỗi worker)
    input_dim = len(selected_features)
    output_dim = 1

    # Khởi tạo mô hình KAN
    kan_model = KAN(width=[input_dim, 10, output_dim], mult_arity=3, base_fun='identity', grid=3, device=current_device)

    # Huấn luyện mô hình KAN cục bộ
    local_steps = 50  # Số bước huấn luyện trên mỗi partition
    local_lamb = 0.01

    print(f"Worker {partition_id}: Bắt đầu huấn luyện KAN với {len(rows)} mẫu...")
    local_results = kan_model.fit(local_dataset, steps=local_steps, opt='LBFGS', lamb=local_lamb, lamb_coef=1.0,
                                  log=0)  # log=0 để không in log quá nhiều từ mỗi worker

    final_train_rmse = local_results['train_loss'][-1]
    final_test_rmse = local_results['test_loss'][-1]  # Đây là test_loss trên dữ liệu cục bộ của worker

    # Trả về các chỉ số từ worker này.
    # Để đơn giản, không trả về model.state_dict vì việc tổng hợp phức tạp hơn.
    return [(partition_id, final_train_rmse, final_test_rmse)]


start_training_time = time.time()
# Phân vùng lại dữ liệu huấn luyện để kiểm soát số lượng worker.
# Mỗi worker sẽ chạy hàm `train_kan_on_partition` trên một phân vùng.
num_partitions = 4  # Điều chỉnh số lượng phân vùng/worker tùy theo tài nguyên cluster
train_final_df_repartitioned = train_final_df.repartition(num_partitions)

# Chạy huấn luyện trên các phân vùng và thu thập kết quả
training_metrics_rdd = train_final_df_repartitioned.rdd.mapPartitionsWithIndex(train_kan_on_partition)
all_training_metrics = training_metrics_rdd.collect()  # Thu thập metrics về driver
end_training_time = time.time()

print("\n--- Kết quả Huấn luyện KAN từ các Partition (Cục bộ) ---")
for metrics in all_training_metrics:
    print(f"Partition {metrics[0]} - Train RMSE: {metrics[1]:.4f}, Test RMSE (Local): {metrics[2]:.4f}")

# Tính toán RMSE trung bình trên các partition
avg_train_rmse = np.mean([m[1] for m in all_training_metrics])
avg_test_rmse_local = np.mean([m[2] for m in all_training_metrics])
print(f"\nRMSE Huấn luyện trung bình (trên các partition): {avg_train_rmse:.4f}")
print(f"RMSE Kiểm thử trung bình (trên dữ liệu cục bộ của partition): {avg_test_rmse_local:.4f}")
print(f"Tổng thời gian huấn luyện (phân tán cục bộ): {end_training_time - start_training_time:.2f} giây")

# --- 4. DỰ ĐOÁN (INFERENCE) PHÂN TÁN ---
# Để dự đoán phân tán, bạn cần một mô hình KAN đã huấn luyện.
# Ở đây, chúng ta sẽ giả định một mô hình KAN đã được huấn luyện tốt (ví dụ: đã lưu)
# và phát sóng (broadcast) state_dict của nó tới các worker để dự đoán.
# Trong thực tế, bạn sẽ có một mô hình đã huấn luyện từ một quy trình riêng.

# Tạo một mô hình KAN "giả" để minh họa broadcast và inference.
# TRONG THỰC TẾ, HÃY TẢI MÔ HÌNH KAN ĐÃ HUẤN LUYỆN CỦA BẠN TẠI ĐÂY!
# Ví dụ: model = KAN(width=[input_dim, 10, output_dim], ...); model.load_state_dict(torch.load("your_model.pt"))
input_dim_inference = len(selected_features)
output_dim_inference = 1
# Mô hình KAN giả định này sẽ được broadcast.
inference_kan_model = KAN(width=[input_dim_inference, 10, output_dim_inference], mult_arity=3, base_fun='identity',
                          grid=3, device='cpu')
# Giả sử bạn đã huấn luyện nó và đây là state_dict của nó
# Bạn có thể tải state_dict từ một mô hình đã lưu
# inference_kan_model.load_state_dict(torch.load("path/to/your_trained_kan_model.pth"))
inference_kan_model.eval()  # Đặt mô hình ở chế độ đánh giá

# Phát sóng state_dict của mô hình và các scaler parameters tới các worker
broadcasted_kan_model_state_dict = sc.broadcast(inference_kan_model.state_dict())
broadcasted_scaler_y_mean = sc.broadcast(scaler_y_model.mean)
broadcasted_scaler_y_std = sc.broadcast(scaler_y_model.std)


def predict_on_spark_partition(iterator):
    """
    Hàm này sẽ chạy trên mỗi phân vùng của dữ liệu kiểm thử để thực hiện dự đoán.
    """
    # Lấy state_dict và tải lại mô hình trên worker
    model_state = broadcasted_kan_model_state_dict.value
    input_dim_worker = len(selected_features)
    output_dim_worker = 1
    # Sử dụng CPU trên worker cho inference nếu không có GPU
    model = KAN(width=[input_dim_worker, 10, output_dim_worker], mult_arity=3, base_fun='identity', grid=3,
                device='cpu')
    model.load_state_dict(model_state)
    model.eval()  # Đặt mô hình ở chế độ đánh giá

    # Tải tham số scaler_y để inverse transform
    scaler_y_mean = broadcasted_scaler_y_mean.value
    scaler_y_std = broadcasted_scaler_y_std.value
    scaler_y_recreated_worker = StandardScaler()
    scaler_y_recreated_worker.mean_ = np.array([scaler_y_mean[0]])
    scaler_y_recreated_worker.scale_ = np.array([scaler_y_std[0]])

    results = []
    with torch.no_grad():  # Không tính gradient trong quá trình dự đoán
        for row in iterator:
            scaled_features = torch.tensor(row.input_features.toArray(), dtype=torch.float32).unsqueeze(
                0)  # Thêm dimension batch
            actual_scaled_label = torch.tensor(row.scaled_label.toArray(), dtype=torch.float32)

            # Dự đoán
            predicted_scaled = model(scaled_features)

            # Inverse transform về thang đo gốc
            predicted_original = scaler_y_recreated_worker.inverse_transform(predicted_scaled.cpu().numpy()).flatten()
            actual_original = scaler_y_recreated_worker.inverse_transform(actual_scaled_label.cpu().numpy()).flatten()

            results.append((actual_original[0], predicted_original[0]))  # (Actual, Predicted)
    return results


start_inference_time = time.time()
# Chạy dự đoán trên các phân vùng của tập kiểm thử
predictions_rdd = test_final_df.rdd.mapPartitions(predict_on_spark_partition)

# Tính toán RMSE phân tán trên RDD (KHÔNG dùng collect() tất cả dự đoán)
# Các hàm sum() và count() của RDD được tối ưu để chạy phân tán.
total_squared_error = predictions_rdd.map(lambda x: (x[0] - x[1]) ** 2).sum()
total_samples = predictions_rdd.count()
rmse_distributed_inference = np.sqrt(total_squared_error / total_samples) if total_samples > 0 else 0

end_inference_time = time.time()

print(f"\n--- Kết quả Dự đoán Phân tán ---")
print(f"RMSE trên thang đo gốc (từ dự đoán phân tán): {rmse_distributed_inference:.4f}")
print(f"Tổng thời gian Dự đoán Phân tán: {end_inference_time - start_inference_time:.2f} giây")

# --- 5. DỪNG SPARK SESSION ---
spark.stop()