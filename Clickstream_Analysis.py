import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("ecommerce_clickstream_transactions_enhanced.csv")
def clean_full_timestamp(row):
    try:
        date_str = f"{int(row['Year'])}-{int(row['Month']):02d}-{int(row['Day']):02d}"
        time_str = "00:" + str(row['Timestamp'])
        return pd.to_datetime(f"{date_str} {time_str}", errors='coerce')
    except:
        return pd.NaT
data['FullTimestamp'] = data.apply(clean_full_timestamp, axis=1)
data = data.dropna(subset=['FullTimestamp'])
data = data.sort_values(by=['UserID', 'FullTimestamp']).reset_index(drop=True)
data['TimeDiff_Seconds'] = data.groupby('UserID')['FullTimestamp'].diff().dt.total_seconds().fillna(0)
data['TimeDiff_Hours'] = data['TimeDiff_Seconds'] / 3600
data['NewSession'] = (data['TimeDiff_Seconds'] > 1800).astype(int)
data['SessionID'] = data.groupby('UserID')['NewSession'].cumsum()
session_info = data.groupby(['UserID', 'SessionID'])['FullTimestamp'].agg(['min', 'max']).reset_index()
session_info['SessionDuration'] = (session_info['max'] - session_info['min']).dt.total_seconds()
data = data.merge(session_info[['UserID', 'SessionID', 'SessionDuration']], on=['UserID', 'SessionID'], how='left')
data['FullTimestamp'] = pd.to_datetime(data['FullTimestamp'], errors='coerce', utc=True)
data['Weekday'] = data['FullTimestamp'].dt.dayofweek
data['HourOfDay'] = data['FullTimestamp'].dt.hour
data['Made_Purchase'] = np.where(data['EventType'] == 'purchase', 1, 0)
features_to_scale = [
    'SessionDuration', 'TimeDiff_Seconds', 'HourOfDay', 'Weekday',
    'ProductCount', 'EventCount', 'DaysSinceLastPurchase',
    'recency', 'frequency', 'monetary'
]
for col in features_to_scale:
    if col not in data.columns:
        data[col] = 0
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])
data.drop(columns=['NewSession'], inplace=True)
data.to_csv("final_preprocessed_data.csv", index=False)
print("✅ Preprocessing complete. Final dataset saved as 'final_preprocessed_data.csv'")
print("Final shape:", data.shape)
print("📊 Final Preprocessing Check")
print(f"\n✅ Dataset shape: {data.shape}")
print(f"\n📋 Columns: {list(data.columns)}")
print("\n🔍 Missing values:")
print(data.isnull().sum()[data.isnull().sum() > 0])
print("\n🧠 Data types:")
print(data.dtypes)
print("\n🔎 Sample rows:")
print(data.head(3))

import pandas as pd
data['FullTimestamp'] = pd.to_datetime(data['FullTimestamp'], errors='coerce')
reference_date = data['FullTimestamp'].max()
rfm = data.groupby('UserID').agg(
    recency = ('FullTimestamp', lambda x: (reference_date - x.max()).days),
    frequency = ('SessionID', 'nunique'),
    monetary = ('Amount', 'sum')
).reset_index()
data = data.merge(rfm, on='UserID', how='left')
print("✅ RFM Features Added")
print(data[['UserID', 'recency', 'frequency', 'monetary']].head())

data.to_csv("final_with_rfm.csv", index=False)
print("✅ File saved as 'final_with_rfm.csv'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
rfm_data = pd.read_csv("final_with_rfm.csv")
rfm_df = rfm_data[['UserID', 'recency', 'frequency', 'monetary']]
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[['recency', 'frequency', 'monetary']])
pca = PCA(n_components=3)
rfm_pca = pca.fit_transform(rfm_scaled)
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.show()
inertia = []
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rfm_pca)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(rfm_pca, labels))
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(k_range, inertia, marker='o', linestyle='-', color='blue')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.plot(k_range, silhouette_scores, marker='s', linestyle='--', color='red')
ax2.set_ylabel('Silhouette Score', color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.title('Elbow Method and Silhouette Score Analysis')
fig.tight_layout()
plt.grid(True)
plt.show()
print("Silhouette Scores by Cluster Number:")
for k, score in zip(k_range, silhouette_scores):
    print(f"{k} clusters: {score:.4f}")
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"\nOptimal number of clusters chosen: {optimal_k}")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_pca)
cluster_summary = rfm_df.groupby('Cluster').mean().round(2)
print("\nCluster Profiling (Business Insights):")
print(cluster_summary)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=rfm_pca[:, 0], y=rfm_pca[:, 1], hue=rfm_df['Cluster'], palette='Set2', s=50, alpha=0.7)
plt.title('Customer Segments Visualization (PCA Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()
marketing_campaigns = {
    0: "VIP Customers: Exclusive discounts & early sales access.",
    1: "Loyal Customers: Personalized product recommendations.",
    2: "At-Risk Customers: Win-back campaigns with incentives.",
    3: "Big Spenders: Premium upselling opportunities.",
    4: "New/Inactive Users: Engagement & onboarding initiatives."
}
rfm_df['Marketing_Action'] = rfm_df['Cluster'].map(marketing_campaigns)
relevant_columns = ['UserID', 'recency', 'frequency', 'monetary', 'Cluster', 'Marketing_Action']
for cluster in rfm_df['Cluster'].unique():
    filename = f"cluster_{cluster}_customers.csv"
    cluster_data = rfm_df[rfm_df['Cluster'] == cluster][relevant_columns]
    cluster_data.to_csv(filename, index=False)
rfm_df[relevant_columns].to_csv("customer_marketing_actions.csv", index=False)
cluster_summary.plot(kind='bar', figsize=(12, 6))
plt.title('Average RFM values per Cluster')
plt.ylabel('Average Value')
plt.xlabel('Cluster')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.legend(['Recency', 'Frequency', 'Monetary'], loc='upper right')
plt.show()

rfm_df = pd.read_csv("customer_marketing_actions.csv")
action_counts = rfm_df['Marketing_Action'].value_counts().reset_index()
action_counts.columns = ['Marketing_Action', 'Customer_Count']
plt.figure(figsize=(12, 6))
sns.barplot(x='Customer_Count', y='Marketing_Action', data=action_counts, palette='viridis')
plt.title("Customer Count by Marketing Action", fontsize=16)
plt.xlabel("Number of Customers")
plt.ylabel("Marketing Action")
plt.grid(axis='x')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = {
    'Marketing Action': [
        'Loyal Customers',
        'VIP Customers',
        'At-Risk Customers'
    ],
    'Customer Count': [29500, 22500, 7000]
}
df = pd.DataFrame(data)
poster_colors = ['
plt.figure(figsize=(12, 5))
bars = sns.barplot(
    y='Marketing Action',
    x='Customer Count',
    data=df,
    palette=poster_colors
)
plt.title("Customer Count by Marketing Action", fontsize=14)
plt.xlabel("Number of Customers")
plt.ylabel("Marketing Action")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data['recency'], bins=30, kde=True)
plt.title('Distribution of Recency (days since last interaction)')
plt.xlabel('Recency (days)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
print(data['recency'].describe(percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]))

rfm_data['Churn'] = rfm_data['recency'].apply(lambda x: 1 if x >= 7 else 0)
print(rfm_data['Churn'].value_counts())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

rfm_data['Churn'] =rfm_data['recency'].apply(lambda x: 1 if x >= 7 else 0)
check_columns = ['frequency', 'monetary', 'SessionDuration', 'EventCount', 'ProductCount', 'Churn']
plt.figure(figsize=(10,6))
sns.heatmap(rfm_data[check_columns].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap clearly showing relationships")
plt.show()

features = ['frequency', 'monetary', 'SessionDuration', 'EventCount', 'ProductCount']
for feature in features:
    plt.figure(figsize=(8,4))
    sns.boxplot(x='Churn', y=feature, data=rfm_data)
    plt.title(f"{feature} by Churn Status (0=Active, 1=Churn)")
    plt.show()

import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
features = ['frequency', 'monetary', 'SessionDuration', 'EventCount', 'ProductCount']
X = rfm_data[features]
y = rfm_data['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'Naive Bayes': GaussianNB() }
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Active', 'Churn'], yticklabels=['Active', 'Churn'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Churn Prediction')
plt.legend()
plt.grid(True)
plt.show()
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
importances.plot(kind='barh')
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.grid(True)
plt.show()
joblib.dump(rf, "random_forest_churn_model.pkl")
print("✅ Model saved as 'random_forest_churn_model.pkl'")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print("\n✅ Best Parameters Found:")
print(grid_search.best_params_)
y_pred_best = best_rf.predict(X_test)
y_prob_best = best_rf.predict_proba(X_test)[:, 1]
print("\n📊 Classification Report (Tuned Random Forest):")
print(classification_report(y_test, y_pred_best))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob_best))

from sklearn.metrics import confusion_matrix
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Active', 'Churn'], yticklabels=['Active', 'Churn'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.model_selection import RandomizedSearchCV
rfm_data['Churn'] =rfm_data['recency'].apply(lambda x: 1 if x >= 7 else 0)
features = ['frequency', 'monetary', 'SessionDuration', 'EventCount', 'ProductCount']
X = rfm_data[features]
y = rfm_data['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.pipeline import Pipeline
pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(class_weight='balanced', probability=True, random_state=42))
])
pipeline_svm.fit(X_train, y_train)
y_pred_svm = pipeline_svm.predict(X_test)
y_pred_prob_svm = pipeline_svm.predict_proba(X_test)[:, 1]
print("Improved SVM Performance:")
print(classification_report(y_test, y_pred_svm))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_prob_svm))

from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    class_weight=class_weights,
    verbose=1
)
y_pred_probs = model.predict(X_test_scaled)
y_pred_labels = (y_pred_probs >= 0.5).astype(int)
print("\n🧠 Classification Report (Neural Network):")
print(classification_report(y_test, y_pred_labels))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_probs))
cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Active', 'Churn'], yticklabels=['Active', 'Churn'])
plt.title("Neural Network - Confusion Matrix (Balanced)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_probs):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Neural Network - ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd
import plotly.graph_objects as go
clickstream_df = pd.read_csv("ecommerce_clickstream_transactions_enhanced.csv")
funnel_stages = ['login', 'page_view', 'product_view', 'click', 'add_to_cart', 'purchase']
funnel_df = clickstream_df[clickstream_df['EventType'].isin(funnel_stages)].copy()
funnel_df['Timestamp'] = pd.to_datetime(funnel_df['Timestamp'])
funnel_df = funnel_df.sort_values(by=['UserID', 'Timestamp'])
first_stage_times = funnel_df.groupby(['UserID', 'EventType'])['Timestamp'].min().unstack()
def check_funnel_progression(row):
    progression = {}
    for i, stage in enumerate(funnel_stages):
        if i == 0:
            progression[stage] = pd.notna(row[stage])
        else:
            prev_stage = funnel_stages[i - 1]
            progression[stage] = (
                pd.notna(row[stage]) and
                pd.notna(row[prev_stage]) and
                row[stage] >= row[prev_stage]
            )
    return pd.Series(progression)
funnel_flags = first_stage_times.apply(check_funnel_progression, axis=1)
funnel_flags['UserID'] = first_stage_times.index
user_funnel = funnel_flags[['UserID'] + funnel_stages].reset_index(drop=True)
funnel_summary_df = pd.DataFrame({
    'Stage': funnel_stages,
    'Users Reached': [user_funnel[stage].sum() for stage in funnel_stages]
})
total_users = user_funnel.shape[0]
print("✅ Funnel Summary:")
print(funnel_summary_df)
print(f"\nTotal Users: {total_users}")
drop_off_data = []
for i in range(len(funnel_stages) - 1):
    current_stage = funnel_stages[i]
    next_stage = funnel_stages[i + 1]
    entered = user_funnel[user_funnel[current_stage]].shape[0]
    reached_next = user_funnel[user_funnel[current_stage] & user_funnel[next_stage]].shape[0]
    dropped = entered - reached_next
    conversion_rate = (reached_next / entered) * 100 if entered else 0
    drop_off_data.append({
        'From → To': f"{current_stage} → {next_stage}",
        'Users Entered': entered,
        'Reached Next Stage': reached_next,
        'Dropped Off': dropped,
        '% Conversion': round(conversion_rate, 2)
    })
drop_off_df = pd.DataFrame(drop_off_data)
print("\n✅ Drop-Off Analysis:")
print(drop_off_df)
fig = go.Figure(go.Funnel(
    y=funnel_summary_df['Stage'],
    x=funnel_summary_df['Users Reached'],
    textinfo="value+percent previous+percent initial"
))
fig.update_layout(
    title="📊 Realistic User Funnel (Progression by Stage)",
    font=dict(size=14),
    margin=dict(l=80, r=80, t=50, b=50)
)
fig.show()

import plotly.graph_objects as go
stages = ['login', 'page_view', 'product_view', 'click', 'add_to_cart', 'purchase']
counts = [1000, 521, 494, 512, 490, 470]
fig = go.Figure(go.Funnel(
    y=stages,
    x=counts,
    textinfo="value+percent initial+percent previous",
    marker={"color": ["
))
fig.update_layout(
    title="Realistic User Funnel (Progression by Stage)",
    font=dict(size=16),
    plot_bgcolor='white'
)
fig.show()

cluster_df = pd.read_csv("customer_marketing_actions.csv")[['UserID', 'Cluster']]
stage_times = funnel_df.groupby(['UserID', 'EventType'])['Timestamp'].min().unstack().reset_index()
time_to_conversion = pd.DataFrame({'UserID': stage_times['UserID']})
for i in range(1, len(funnel_stages)):
    start, end = funnel_stages[i - 1], funnel_stages[i]
    delta = (stage_times[end] - stage_times[start]).dt.total_seconds() / 60
    time_to_conversion[f"{start} → {end} (min)"] = delta
cleaned_time_df = time_to_conversion.copy()
for col in cleaned_time_df.columns[1:]:
    cleaned_time_df = cleaned_time_df[cleaned_time_df[col] >= 0]
avg_time_per_stage = cleaned_time_df.drop(columns='UserID').mean().div(60).round(2)
stage_times_clustered = pd.merge(stage_times, cluster_df, on='UserID', how='inner')
cluster_avg_times = {}
for cluster in sorted(stage_times_clustered['Cluster'].unique()):
    group = stage_times_clustered[stage_times_clustered['Cluster'] == cluster]
    transitions = {}
    for i in range(1, len(funnel_stages)):
        s, e = funnel_stages[i - 1], funnel_stages[i]
        delta = (group[e] - group[s]).dt.total_seconds() / 60
        delta = delta[delta >= 0]
        transitions[f"{s} → {e}"] = round(delta.mean() / 60, 2)
    cluster_avg_times[f"Cluster {cluster}"] = transitions
cluster_avg_time_df = pd.DataFrame(cluster_avg_times).T
fig = go.Figure(go.Funnel(
    y=funnel_summary_df['Stage'],
    x=funnel_summary_df['Users Reached'],
    textinfo="value+percent previous+percent initial"
))
fig.update_layout(title="Realistic Funnel Progression", font=dict(size=14))
fig.show()
plt.figure(figsize=(12, 6))
cluster_avg_time_df.T.plot(kind='bar', figsize=(12, 6))
plt.title("Avg Time Between Funnel Stages by Cluster (in Hours)")
plt.ylabel("Hours")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

from collections import defaultdict
from itertools import combinations
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
purchase_data = clickstream_df[clickstream_df['EventType'] == 'purchase']
user_product_matrix = purchase_data.groupby(['UserID', 'ProductID']).size().unstack(fill_value=0)
svd = TruncatedSVD(n_components=20, random_state=42)
user_factors = svd.fit_transform(user_product_matrix)
user_similarity = cosine_similarity(user_factors)
user_purchases = purchase_data.groupby('UserID')['ProductID'].apply(set)
co_occurrence = defaultdict(lambda: defaultdict(int))
for products in user_purchases:
    for prod_a, prod_b in combinations(products, 2):
        co_occurrence[prod_a][prod_b] += 1
        co_occurrence[prod_b][prod_a] += 1
user_ids = user_product_matrix.index
sample_user_index = 0
sample_user_id = user_ids[sample_user_index]
user_products = user_purchases[sample_user_id]
similar_user_indices = user_similarity[sample_user_index].argsort()[::-1][1:6]
similar_user_ids = user_ids[similar_user_indices]
collab_scores = user_product_matrix.loc[similar_user_ids].sum()
collab_scores = collab_scores.drop(labels=user_products, errors='ignore')
collab_scores = collab_scores[collab_scores > 0].sort_values(ascending=False)
collab_df = collab_scores.reset_index()
collab_df.columns = ['ProductID', 'Collaborative Score']
cooccur_scores = defaultdict(int)
for product in user_products:
    for related, score in co_occurrence[product].items():
        if related not in user_products:
            cooccur_scores[related] += score
cooccur_df = pd.DataFrame.from_dict(cooccur_scores, orient='index', columns=['Co-Occurrence Score'])
cooccur_df = cooccur_df.sort_values(by='Co-Occurrence Score', ascending=False).reset_index()
cooccur_df.columns = ['ProductID', 'Co-Occurrence Score']
hybrid_df = pd.merge(collab_df, cooccur_df, on='ProductID', how='outer').fillna(0)
hybrid_df['Hybrid Score'] = hybrid_df['Collaborative Score'] + hybrid_df['Co-Occurrence Score']
hybrid_df = hybrid_df.sort_values(by='Hybrid Score', ascending=False).head(10).reset_index(drop=True)
plt.figure(figsize=(10, 6))
plt.barh(hybrid_df['ProductID'], hybrid_df['Hybrid Score'], color='purple')
plt.xlabel("Hybrid Score")
plt.title(f"Top Recommended Products for User {sample_user_id}")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
print(f"✅ Top Hybrid Recommendations for User {sample_user_id}:\n")
print(hybrid_df)

df = pd.read_csv("final_preprocessed_data.csv")
product_views = df[df['EventType'] == 'product_view']
view_counts = product_views.groupby(['UserID', 'ProductID']).size().reset_index(name='ViewCount')
sorted_views = view_counts.sort_values(by='ViewCount', ascending=False)
top_10_intent_unique = sorted_views.drop_duplicates(subset='UserID', keep='first').head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_10_intent_unique['UserID'].astype(str), top_10_intent_unique['ViewCount'], color='teal')
plt.xlabel("View Count")
plt.ylabel("UserID")
plt.title("🎯 Top 10 High-Intent Users and Their Most Viewed Products")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
print("✅ Recommended Products for High-Intent Users:")
print(top_10_intent_unique)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
rfm_df = pd.read_csv("final_with_rfm.csv")
df = pd.read_csv("final_preprocessed_data.csv")
rfm_features = rfm_df[['recency', 'frequency', 'monetary']]
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)
iso_forest = IsolationForest(contamination=0.02, random_state=42)
rfm_df['Anomaly'] = iso_forest.fit_predict(rfm_scaled)
rfm_anomalies = rfm_df[rfm_df['Anomaly'] == -1].copy()
rfm_normals = rfm_df[rfm_df['Anomaly'] == 1].copy()
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm_normals, x='recency', y='monetary', label='Normal', alpha=0.6)
sns.scatterplot(data=rfm_anomalies, x='recency', y='monetary', color='red', label='Anomaly', s=100, edgecolor='black')
plt.title("🧠 RFM Anomaly Clusters (Isolation Forest)")
plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
suspicious_df = df[df['EventType'].isin(['click', 'login'])]
user_event_counts = suspicious_df.groupby('UserID').size().reset_index(name='EventCount')
threshold = user_event_counts['EventCount'].quantile(0.95)
anomalous_users = user_event_counts[user_event_counts['EventCount'] > threshold]
print(f"📈 95th percentile threshold: {threshold:.2f}")
print("🚨 High Click/Login Activity Users:")
print(anomalous_users.sort_values(by='EventCount', ascending=False).head(10))
top_ids = anomalous_users['UserID'].head(3).tolist()
timeline_df = df[df['UserID'].isin(top_ids)][['UserID', 'EventType', 'Timestamp']]
timeline_df['Timestamp'] = pd.to_datetime(timeline_df['Timestamp'])
timeline_df = timeline_df.sort_values(by=['UserID', 'Timestamp'])
print("\n⏱️ Timeline Events for Top Suspicious Users:")
print(timeline_df.head(20))

event_anomalous_users = user_event_counts[user_event_counts['EventCount'] > threshold]
event_counts = pd.Series({
    'Normal Users': df['UserID'].nunique() - event_anomalous_users['UserID'].nunique(),
    'Event Spike Anomalies': event_anomalous_users['UserID'].nunique()
})
colors = ['
plt.figure(figsize=(6, 6))
plt.pie(event_counts, labels=event_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title("🛑 Click/Login Spike Anomaly Distribution")
plt.axis('equal')
plt.tight_layout()
plt.show()
rfm_anomalies.to_csv("rfm_anomalies_users.csv", index=False)
event_anomalous_users.to_csv("event_spike_anomalies_users.csv", index=False)
print("✅ Anomaly Detection & Fraud Reports Exported:")
print("- rfm_anomalies_users.csv")
print("- event_spike_anomalies_users.csv")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
rfm_anomalies = pd.read_csv("rfm_anomalies_users.csv")
event_anomalies = pd.read_csv("event_spike_anomalies_users.csv")
churn_df = pd.read_csv("customer_marketing_actions.csv")
cluster_0 = pd.read_csv("cluster_0_customers.csv")
cluster_1 = pd.read_csv("cluster_1_customers.csv")
cluster_2 = pd.read_csv("cluster_2_customers.csv")
cluster_0["Cluster"] = 0
cluster_1["Cluster"] = 1
cluster_2["Cluster"] = 2
cluster_all = pd.concat([cluster_0, cluster_1, cluster_2], ignore_index=True)
summary = cluster_all[["UserID", "Cluster"]].drop_duplicates()
churn_df["Churn"] = churn_df["Marketing_Action"].str.contains("Win-Back|At-Risk", case=False, na=False).astype(int)
summary = summary.merge(churn_df[["UserID", "Churn"]], on="UserID", how="left")
summary["RFM_Anomaly"] = summary["UserID"].isin(rfm_anomalies["UserID"])
summary["Event_Anomaly"] = summary["UserID"].isin(event_anomalies["UserID"])
plt.figure(figsize=(14, 10))
plt.suptitle("📊 Final Project Snapshot – Segmentation, Churn, Anomaly", fontsize=16)
plt.subplot(2, 2, 1)
sns.countplot(x="Cluster", data=summary, palette="Set2")
plt.title("📦 Customer Segmentation (K-Means Clusters)")
plt.xlabel("Cluster")
plt.ylabel("User Count")
plt.subplot(2, 2, 2)
churn_counts = summary["Churn"].value_counts().rename({0: "Active", 1: "Churned"})
plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90, colors=["
plt.title("🔁 Churn Prediction Status")
plt.subplot(2, 2, 3)
rfm_counts = summary["RFM_Anomaly"].value_counts().rename({True: "Anomaly", False: "Normal"})
plt.pie(rfm_counts, labels=rfm_counts.index, autopct='%1.1f%%', startangle=90, colors=["
plt.title("🧠 RFM-Based Anomaly Rate")
plt.subplot(2, 2, 4)
event_counts = summary["Event_Anomaly"].value_counts().rename({True: "Spike Detected", False: "Normal"})
plt.pie(event_counts, labels=event_counts.index, autopct='%1.1f%%', startangle=90, colors=["
plt.title("🖱️ Click/Login Spike Detection")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
print("\n📋 PROJECT SUMMARY")
print("────────────────────────────────────────")
print(f"👥 Total Users: {summary['UserID'].nunique()}")
print(f"📦 Segmentation Clusters: {summary['Cluster'].nunique()}")
print(f"🔁 Users Predicted to Churn: {summary['Churn'].sum()}")
print(f"🧠 RFM-Based Anomalies: {summary['RFM_Anomaly'].sum()}")
print(f"🖱️ Event-Based Spike Anomalies: {summary['Event_Anomaly'].sum()}")
print("🎯 Recommendation System: Implemented (Collaborative + Co-Occurrence)")
print("✅ Final Pipeline Complete!")

import plotly.express as px
labels = ['Churned', 'Active']
values = [25.5, 74.5]
colors = ['
fig = px.pie(
    names=labels,
    values=values,
    title="📉 Churn Distribution",
    color_discrete_sequence=colors,
    hole=0.3
)
fig.update_traces(textinfo='percent+label')
fig.update_layout(
    font=dict(size=16),
    showlegend=True,
    plot_bgcolor='white'
)
fig.show()

pip install -U kaleido

import plotly.graph_objects as go
fig_pie = go.Figure(data=[go.Pie(
    labels=['Active', 'Churned'],
    values=[76.5, 23.5],
    hole=0.5,
    marker=dict(colors=['
    textinfo='label+percent',
)])
fig_pie.update_layout(
    title='Churn Distribution',
    showlegend=True,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='
)
stages = ['login', 'page_view', 'product_view', 'click', 'add_to_cart', 'purchase']
values = [1000, 521, 494, 512, 468, 450]
fig_funnel = go.Figure(go.Funnel(
    y=stages,
    x=values,
    marker=dict(color=['
))
fig_funnel.update_layout(
    title='User Funnel Analysis',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='
)
fig_pie.show()
fig_funnel.show()
fig_pie.write_image("churn_pie_chart.png")
fig_funnel.write_image("user_funnel_chart.png")

import plotly.express as px
import pandas as pd
data = {
    'Marketing Action': [
        'Loyal Customers: Personalized product recommendations.',
        'VIP Customers: Exclusive discounts & early sales access.',
        'At-Risk Customers: Win-back campaigns with incentives.'
    ],
    'Number of Customers': [29500, 22500, 6800]
}
df = pd.DataFrame(data)
fig_bar = px.bar(
    df,
    x='Number of Customers',
    y='Marketing Action',
    orientation='h',
    color='Marketing Action',
    color_discrete_sequence=['
    title='Customer Count by Marketing Action'
)
fig_bar.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='black', size=14),
    showlegend=False,
    xaxis=dict(
        showline=True, linewidth=1, linecolor='black',
        gridcolor='black', zeroline=False
    ),
    yaxis=dict(
        showline=True, linewidth=1, linecolor='black',
        gridcolor='black', zeroline=False
    )
)
fig_bar.write_image("bar_marketing_action_black.png")

import plotly.graph_objects as go
labels = [
    "Data Collection", "Preprocessing", "RFM Analysis", "PCA", "K-Means Clustering",
    "Customer Segments", "Churn Prediction", "Funnel Analysis", "Recommendation Engine",
    "Anomaly Detection", "Streamlit Dashboard"
]
source = [0, 1, 2, 3, 4, 5, 5, 5, 5, 6, 7, 8, 9]
target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
value = [1]*len(source)
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color="rgba(173, 216, 230, 0.8)"
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color="rgba(100, 149, 237, 0.4)"
    )
)])
fig.update_layout(
    title_text="Project Methodology Workflow (Sankey Diagram)",
    font_size=13,
    paper_bgcolor='white',
    plot_bgcolor='white'
)
fig.show()

fig.write_image("methodology_sankey_diagram.png")