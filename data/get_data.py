import kagglehub
import os
import shutil

# Download dataset
dataset_path = kagglehub.dataset_download(
    "dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024"
)

print("Downloaded to:", dataset_path)
print("Files:", os.listdir(dataset_path))

# Identify CSV file
csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
assert len(csv_files) == 1, "Expected exactly one CSV file"

src_csv = os.path.join(dataset_path, csv_files[0])
dst_csv = os.path.join("data", "sp500_news.csv")

shutil.copyfile(src_csv, dst_csv)

print("Saved dataset to:", dst_csv)
