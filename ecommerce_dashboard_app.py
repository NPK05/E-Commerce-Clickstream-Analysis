import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(page_title='E-Commerce Dashboard', layout='wide')

@st.cache_data
def load_all():
    churn = pd.read_csv('customer_marketing_actions.csv')
    rfm_anom = pd.read_csv('rfm_anomalies_users.csv')
    event_anom = pd.read_csv('event_spike_anomalies_users.csv')
    funnel = pd.read_csv('user_funnel_table.csv')
    return (churn, rfm_anom, event_anom, funnel)
churn_df, rfm_anomalies, event_anomalies, funnel_df = load_all()
st.sidebar.title('📊 Dashboard Navigation')
tab = st.sidebar.radio('Go to Section:', ['Overview', 'Customer Segmentation', 'Churn Prediction', 'Model Performance', 'Anomaly Detection', 'Funnel Analysis', 'Product Recommendations'])
if tab == 'Overview':
    st.title('📊 Project Overview')
    col1, col2, col3 = st.columns(3)
    total_users = churn_df['UserID'].nunique()
    churned = churn_df['Churn'].sum()
    active = total_users - churned
    col1.metric('Total Users', f'{total_users:,}')
    col2.metric('Churned Users', f'{churned:,}')
    col3.metric('Active Users', f'{active:,}')
    fig = px.pie(values=[active, churned], names=['Active', 'Churned'], title='Churn Distribution')
    st.plotly_chart(fig, use_container_width=True)
elif tab == 'Customer Segmentation':
    st.title('👥 Customer Segmentation')
    cluster_counts = churn_df['Cluster'].value_counts().sort_index()
    fig = px.bar(x=cluster_counts.index, y=cluster_counts.values, labels={'x': 'Cluster', 'y': 'User Count'}, title='Cluster Distribution', color=cluster_counts.values)
    st.plotly_chart(fig, use_container_width=True)
elif tab == 'Churn Prediction':
    st.title('🔁 Churn Prediction Table')
    cluster = st.selectbox('Select Cluster:', churn_df['Cluster'].unique())
    filtered = churn_df[churn_df['Cluster'] == cluster]
    st.dataframe(filtered[['UserID', 'Cluster', 'Churn', 'Marketing_Action']])
    st.download_button('⬇️ Download CSV', data=filtered.to_csv(index=False), file_name='churn_predictions.csv')
elif tab == 'Model Performance':
    st.title('🤖 Model Evaluation')
    model_data = pd.DataFrame({'Model': ['Random Forest', 'XGBoost', 'Neural Network', 'Naive Bayes', 'SVM'], 'Accuracy': [0.99, 0.98, 0.94, 0.91, 0.92], 'Recall': [0.91, 0.88, 0.75, 0.01, 0.0], 'F1-Score': [0.95, 0.92, 0.82, 0.02, 0.0], 'ROC-AUC': [0.999, 0.998, 0.89, 0.55, 0.5]})
    fig = px.bar(model_data, x='Model', y=['Accuracy', 'Recall', 'F1-Score', 'ROC-AUC'], barmode='group')
    st.plotly_chart(fig, use_container_width=True)
elif tab == 'Anomaly Detection':
    st.title('🧠 Anomaly Detection')
    col1, col2 = st.columns(2)
    rfm_count = len(rfm_anomalies)
    event_count = len(event_anomalies)
    col1.metric('RFM Anomalies', rfm_count)
    col2.metric('Event Spike Users', event_count)
    pie_fig = px.pie(values=[rfm_count, event_count], names=['RFM Anomalies', 'Event Spikes'], title='Anomaly Types')
    st.plotly_chart(pie_fig, use_container_width=True)
elif tab == 'Funnel Analysis':
    st.title('🚦 Funnel Conversion by Cluster')
    st.dataframe(funnel_df)
    funnel_fig = px.bar(funnel_df, x=funnel_df.index, y=['login', 'page_view', 'product_view', 'add_to_cart', 'purchase'], title='Funnel Completion per Cluster')
    st.plotly_chart(funnel_fig, use_container_width=True)
elif tab == 'Product Recommendations':
    st.title('🛍️ Product Recommendations')
    st.markdown('\n    This section displays hybrid recommendations based on:\n    - Collaborative filtering\n    - Co-occurrence patterns\n    - User revisit intent (high-interest)\n    ')
    st.info('⚠️ This section is prototype only. Full model integration needed for dynamic recommendations.')
    sample = pd.DataFrame({'UserID': [1, 5, 9], 'Top Product Recommended': ['prod_6845', 'prod_2594', 'prod_8422']})
    st.dataframe(sample)
    st.success('✅ Recommendation logic developed in project. Integrated models pending.')