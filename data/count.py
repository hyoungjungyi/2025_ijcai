import pandas as pd

# CSV 파일 로드
file_path = '/hdd1/hyunjung/2025_ijcai/data/ftse/ftse_general_data.csv'
data = pd.read_csv(file_path)
unique_tickers_count = data['tic'].nunique()
print(f"Unique tickers count: {unique_tickers_count}")


kospi_file_path = '/hdd1/hyunjung/2025_ijcai/data/kospi/kospi_general_data.csv'
kospi_data = pd.read_csv(kospi_file_path)
kospi_unique_tickers_count = kospi_data['tic'].nunique()
print(f"Kospi Unique tickers count: {kospi_unique_tickers_count}")


nasdaq_file_path = '/hdd1/hyunjung/2025_ijcai/data/nasdaq/nasdaq_general_data.csv'
nasdaq_data = pd.read_csv(nasdaq_file_path)
nasdaq_unique_tickers_count = nasdaq_data['tic'].nunique()
print(f"nasdaq Unique tickers count: {nasdaq_unique_tickers_count}")


dj30_file_path = '/hdd1/hyunjung/2025_ijcai/data/dj30/dj30_general_data.csv'
dj30_data = pd.read_csv(dj30_file_path)
dj30_unique_tickers_count = dj30_data['tic'].nunique()
print(f"dj30 Unique tickers count: {dj30_unique_tickers_count}")


csi300_file_path = '/hdd1/hyunjung/2025_ijcai/data/csi300/csi300_general_data.csv'
csi300_data = pd.read_csv(csi300_file_path)
csi300_unique_tickers_count = csi300_data['tic'].nunique()
print(f"csi300 Unique tickers count: {csi300_unique_tickers_count}")