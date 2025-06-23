# E-Commerce-Clickstream-Analysis
Churn prediction, anomaly detection &amp; behavior analysis

🛍️ E-Commerce Clickstream Analysis
A full-stack data science project to analyze user behavior on an e-commerce platform. It tracks funnel stages, predicts churn, detects anomalies, segments customers, and recommends products—all tied together with a custom-built dashboard.
📂 Dataset
The dataset is sourced from [Kaggle's E-Commerce Dataset](https://www.kaggle.com/datasets/mervemenekse/ecommerce-dataset).  
It contains transactional clickstream logs like login, product view, cart addition, and purchase.
## 🧠 Strategy and Business Mapping
Business logic (e.g., churn tagging, loyalty classification, win-back targeting) was derived from the `Marketing_Action` Excel file provided. We:
- Mapped action labels like "Win-Back", "At-Risk", and "Loyal" to churn status
- Grouped customer behavior for segmentation and targeting
- Used this to drive both modeling and dashboard segmentation

📊 **Interactive Dashboard – Built from Scratch**
No AI builders or BI tools like Power BI/Tableau were used.
🛠️ Built with:
- Streamlit for frontend layout
- Plotly & Seaborn for visuals
- Pandas for data manipulation

📌 [Dashboard Codebase → `ecommerce_dashboard_app.py`](ecommerce_dashboard_app.py)
> The entire dashboard layout, interaction logic, data caching, and export functionality was authored manually in Python.
🧪 Core Features
- **Preprocessing**: Timestamp parsing, session allocation, and feature generation
- **Funnel Analysis**: Visual breakdown of multi-stage user flow (login → cart → purchase)
- **Recommendation System**: Product-level interaction-based suggestions using co-occurrence analysis
- **Churn Prediction**: Trained ML models (Random Forest, XGBoost, Neural Net) focused on recall optimization
- **Anomaly Detection**: Applied Isolation Forest on RFM and behavioral features
- **Dashboard View**: Dynamic dashboard for visual interaction, product discovery, and user cohort tracking

🔍 **Derived from Real-World References**
This project was guided by insights from the following sources:

📚 Books:
- Data Science for Business by Foster Provost & Tom Fawcett
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron
- Storytelling with Data by Cole Nussbaumer Knaflic

🧠Bogs / Code Inspiration:
- Top-rated Kaggle notebooks on churn, funnel analytics, and product recommendation
- Real-world e-commerce ML blog posts from Towards Data Science and Analytics Vidhya

🤖 Content Assistance & Structuring:
- GPT-4 was used to:
  - Polish markdown formatting
  - Generate meaningful symbol usage
  - Improve code readability and simplify variable naming
  - Suggest structure for documentation and funnel flow
  
💡 Future Enhancements
- Deploy dashboard to Heroku** or Streamlit Cloud
- Add A/B testing logic for retention strategies
- Integrate product metadata for hybrid content + behavior-based recommendation
- Automate model retraining using CI/CD pipelines (GitHub Actions)

📬 **Contact**
Pavan Kumar Nallabothula

📧 [nallabothulapavan05@gmail.com](mailto:nallabothulapavan05@gmail.com)  

🔗 [LinkedIn](https://www.linkedin.com/in/pavan-kumar-nallabothula)


> ⚠️ All work—data cleaning, modeling, analysis, dashboarding, and visualizations—was built manually from the ground up. No third-party visualization tools or low-code AI builders were used. External content was only used for reference and inspiration.
