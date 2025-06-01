import torch
from kan import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df=pd.read_csv('data/processed_country_data.csv')
df = pd.DataFrame(df)
countries_near_vietnam = [
    'Vietnam',
    'Thailand',
    'Laos',
    'Cambodia',
    'China',
    'Malaysia',
    'Singapore',
    'Indonesia',
    'Philippines',
    'Brunei',
    'South Korea',
    'Japan'
]
df = df[df['continent'] == 'Asia']
df = df[df['location'].isin(countries_near_vietnam)]
selected_features = [
    'new_cases_smoothed_lag_5',
    'new_cases_smoothed_lag_3',
    'new_cases_smoothed_lag_1',
    'new_cases_smoothed_lag_4',
    'new_cases_smoothed_lag_2',
    'new_deaths_smoothed_lag_5',
    'new_deaths_smoothed_lag_3',
    'new_deaths_smoothed_lag_1',
    'total_deaths_lag_5',
    'total_deaths_lag_3',
    'total_deaths_lag_1',
    'population',
    'new_deaths_smoothed_lag_4',
    'new_deaths_smoothed_lag_2'
]
X = df[selected_features]
y = df[['new_cases_next_day']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 1. Chuẩn hóa Input (X_train, X_test) ---

# Khởi tạo StandardScaler cho input
scaler_X = StandardScaler()

# CHỈ fit scaler trên X_train để học các tham số chuẩn hóa (mean, std) từ tập huấn luyện
X_train_scaled = scaler_X.fit_transform(X_train)

# Transform X_test sử dụng scaler đã fit từ X_train
X_test_scaled = scaler_X.transform(X_test)
# --- 2. Chuẩn hóa Label (y_train, y_test) ---

# Khởi tạo StandardScaler cho label
scaler_y = StandardScaler()

# CHỈ fit scaler trên y_train
# y_train cần được reshape nếu nó là Series (dạng (N,)) để trở thành 2D (dạng (N, 1))
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

# Transform y_test sử dụng scaler đã fit từ y_train
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float64).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float64).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float64).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float64).to(device)

dataset={}
dataset['train_input'] = X_train_tensor
dataset['test_input'] = X_test_tensor
dataset['train_label'] = y_train_tensor
dataset['test_label'] = y_test_tensor

steps = 50
grid=3

model = KAN(width=[dataset['train_input'].shape[1],[3,2],1], mult_arity=3,base_fun='identity',grid=grid,device=device)
model.get_act(dataset)
model.plot() 
model.fit(dataset, steps=steps, opt='LBFGS', lamb=0.01, lamb_coef=1.0);
model.prune()
model.plot()
train_losses = []
test_losses = []

results = model.fit(dataset, opt='LBFGS', steps=50)
train_losses += results['train_loss']
test_losses += results['test_loss']
plt.plot(train_losses)
plt.plot(test_losses)
plt.legend(['train', 'test'])
plt.ylabel('RMSE')
plt.xlabel('step')
plt.yscale('log')