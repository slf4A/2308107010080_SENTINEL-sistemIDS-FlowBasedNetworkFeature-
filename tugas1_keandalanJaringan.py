# =========================
# 1. IMPORT & LOAD DATA
# =========================
import pandas as pd
import numpy as np

df = pd.read_csv("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

print(df.shape)
print(df.dtypes)
print(df.head())

# =========================
# 2. PREPROCESSING
# =========================

# Bersihkan nama kolom (hapus spasi depan/belakang)
df.columns = df.columns.str.strip()

# Hapus kolom duplikat (kalau ada)
df = df.loc[:, ~df.columns.duplicated()]

# Replace infinite → NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# =========================
# DETECT KOLOM PACKET (AUTO)
# =========================
if 'Tot Fwd Pkts' in df.columns:
    fwd_col = 'Tot Fwd Pkts'
    bwd_col = 'Tot Bwd Pkts'
elif 'Total Fwd Packets' in df.columns:
    fwd_col = 'Total Fwd Packets'
    bwd_col = 'Total Backward Packets'
else:
    raise ValueError("Kolom packet tidak ditemukan")

df['TotalPackets'] = df[fwd_col] + df[bwd_col]

# =========================
# HANDLE MISSING VALUE
# =========================
missing_summary = df.isna().sum()
print("Missing Values:\n", missing_summary[missing_summary > 0])

# Isi NaN numerik dengan median (AMAN)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].apply(lambda col: col.fillna(col.median()))

# Hapus baris dengan label kosong
df = df.dropna(subset=['Label'])

# =========================
# 3. FEATURE ENGINEERING (AGGREGATION)
# =========================

# DETECT KOLOM IP
if 'Src IP' in df.columns:
    ip_col = 'Src IP'
elif 'Source IP' in df.columns:
    ip_col = 'Source IP'
else:
    raise ValueError("Kolom IP tidak ditemukan")

# Aggregation
conn_per_ip = df.groupby(ip_col).size().rename('TotalConnPerIP')
avg_pkt_per_ip = df.groupby(ip_col)['TotalPackets'].mean().rename('AvgPktPerFlowIP')

df = df.merge(conn_per_ip, on=ip_col, how='left')
df = df.merge(avg_pkt_per_ip, on=ip_col, how='left')

# =========================
# 4. ENCODE LABEL
# =========================
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

print("Mapping Label:", dict(zip(le.classes_, le.transform(le.classes_))))

# =========================
# 5. DROP KOLOM NON-NUMERIK
# =========================

# ✅ Save IP column BEFORE dropping
ip_series = df[ip_col].copy()

drop_cols = [
    'Flow ID',
    'Src IP', 'Dst IP',
    'Source IP', 'Destination IP',
    'Timestamp'
]

df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Drop semua kolom object (safety)
for col in df.columns:
    if df[col].dtype == 'object':
        df = df.drop(columns=[col])
# =========================
# 6. EDA (AUTO DETECT FLOW BYTES)
# =========================
import seaborn as sns
import matplotlib.pyplot as plt

# detect kolom Flow Bytes
if 'Flow Byts/s' in df.columns:
    flow_col = 'Flow Byts/s'
elif 'Flow Bytes/s' in df.columns:
    flow_col = 'Flow Bytes/s'
else:
    flow_col = None

if flow_col:
    plt.figure(figsize=(10,5))
    sns.boxplot(x='Label', y=flow_col, data=df)
    plt.title("Flow Bytes/s per Class (DoS Insight)")
    plt.show()

# Korelasi
corr = df.corr(numeric_only=True)

plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# =========================
# 7. SPLIT DATA
# =========================
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Label'])
y = df['Label']

print("Jumlah kelas:", y.nunique())

if y.nunique() < 2:
    raise ValueError("Dataset hanya punya 1 kelas!")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 8. MODEL LIGHTGBM
# =========================
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y)),
    n_estimators=100,
    learning_rate=0.1
)

model.fit(X_train, y_train)

# =========================
# 9. EVALUATION
# =========================
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# =========================
# 10. FEATURE IMPORTANCE
# =========================
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 15 Feature Importance:")
print(importance.head(15))

df_test = X_test.copy()
df_test['Actual'] = y_test
df_test['Predicted'] = y_pred

# ✅ Use saved ip_series
df_test[ip_col] = ip_series.reindex(df_test.index)

label_map = dict(zip(le.transform(le.classes_), le.classes_))

df_test['Actual_Label'] = df_test['Actual'].map(label_map)
df_test['Predicted_Label'] = df_test['Predicted'].map(label_map)

attack_df = df_test[df_test['Predicted_Label'] != 'BENIGN']
compare = df_test.groupby('Predicted_Label').agg({
    flow_col: 'mean',
    'TotalConnPerIP': 'mean',
    'AvgPktPerFlowIP': 'mean'
})

print(compare)

plt.figure(figsize=(10,5))
sns.boxplot(x='Predicted_Label', y=flow_col, data=df_test)
plt.title("Flow Bytes/s: BENIGN vs ATTACK")
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='Predicted_Label', y='TotalConnPerIP', data=df_test)
plt.title("Total Connection per IP")
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='Predicted_Label', y='AvgPktPerFlowIP', data=df_test)
plt.title("Avg Packet per Flow per IP")
plt.show()

top_attacker = df_test.groupby(ip_col)['Predicted'].count().sort_values(ascending=False).head(10)
print("---Top Attacker---")
print(top_attacker)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['BENIGN', 'PortScan'],
            yticklabels=['BENIGN', 'PortScan'])

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix Heatmap')
plt.show()