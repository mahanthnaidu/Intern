import pandas as pd

file_path = 'metrics.csv'  
df = pd.read_csv(file_path)

deviation_threshold = 0.36
normal_df = df.iloc[:7].copy()
abnormal_df = pd.DataFrame(columns=df.columns)

for i in range(7, len(df)):
    last_7_values = df['Monthly Sales'].iloc[i-7:i]
    last_7_avg = last_7_values.mean()
    current_value = df['Monthly Sales'].iloc[i]
    deviation = abs(current_value - last_7_avg) / last_7_avg
    if deviation >= deviation_threshold:
        abnormal_df = pd.concat([abnormal_df, df.iloc[[i]]])
    else:
        normal_df = pd.concat([normal_df, df.iloc[[i]]])

normal_df.to_csv('normal_data.csv', index=False)
abnormal_df.to_csv('abnormal_data.csv', index=False)

print("Normal data rows:", normal_df.shape[0])
print("Abnormal data rows:", abnormal_df.shape[0])
