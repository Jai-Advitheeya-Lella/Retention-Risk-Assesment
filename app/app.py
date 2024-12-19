import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Database initialization
def init_db():
    conn = sqlite3.connect('bank_customers.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS customers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  credit_score INTEGER,
                  age INTEGER,
                  tenure INTEGER,
                  balance REAL,
                  num_products INTEGER,
                  has_credit_card BOOLEAN,
                  is_active_member BOOLEAN,
                  estimated_salary REAL,
                  geography TEXT,
                  gender TEXT,
                  churn_probability REAL,
                  prediction_date TIMESTAMP)''')
    conn.commit()
    conn.close()

# Load model and scaler
model = joblib.load('rf_model.pkl')
scaler, feature_names = joblib.load('scaler.pkl')

# Print debugging information
print("Exact feature names and order from scaler:", feature_names)


def main():
    init_db()
    
    st.title('Bank Customer Churn Prediction')
    
    # Enhanced navigation with clear question-based structure
    page = st.sidebar.selectbox(
        'Choose a page', 
        ['Individual Churn Prediction',
         'Segment Analysis',
         'Feature Impact Analysis',
         'Retention Strategy',
         'Data Management',
         'Model Performance']
    )
    
    if page == 'Individual Churn Prediction':
        show_prediction_page()
    elif page == 'Segment Analysis':
        show_segment_analysis()
    elif page == 'Feature Impact Analysis':
        show_feature_impact()
    elif page == 'Retention Strategy':
        show_retention_strategy()
    elif page == 'Data Management':
        show_data_management()
    else:
        show_model_performance()

def make_prediction(credit_score, age, tenure, balance, num_products,
                    has_credit_card, is_active_member, estimated_salary,
                    geography, gender):
    try:
        # Create input dictionary
        input_dict = {
            'Tenure': float(tenure),
            'Age': float(age),
            'Balance': float(balance),
            'NumOfProducts': float(num_products),
            'HasCrCard': float(has_credit_card),
            'CreditScore': float(credit_score),
            'IsActiveMember': float(is_active_member),
            'EstimatedSalary': float(estimated_salary),
            'Gender': 1 if gender == 'Female' else 0,
            'Geography_Germany': 1 if geography == 'Germany' else 0,
            'Geography_Spain': 1 if geography == 'Spain' else 0
        }
        
        # Create DataFrame and enforce feature order
        input_data = pd.DataFrame([input_dict])
        input_data = input_data[feature_names]  # Use exact order from training
        
        # Debug prints
        print("\nFeature order check:")
        print("Expected features:", feature_names)
        print("Input data features:", input_data.columns.tolist())
        
        # Scale the data
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict_proba(scaled_data)[0][1]
        
        # Save to database
        conn = sqlite3.connect('bank_customers.db')
        c = conn.cursor()
        c.execute('''INSERT INTO customers 
                     (credit_score, age, tenure, balance, num_products, 
                      has_credit_card, is_active_member, estimated_salary,
                      geography, gender, churn_probability, prediction_date)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (credit_score, age, tenure, balance, num_products,
                   has_credit_card, is_active_member, estimated_salary,
                   geography, gender, float(prediction), datetime.now()))
        conn.commit()
        conn.close()

        return float(prediction)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        print(f"Detailed error: {e}")  
        return None
    
def show_data_management():
    st.header("Data Management")
    
    # Add delete functionality
    with st.expander("Delete Records"):
        conn = sqlite3.connect('bank_customers.db')
        df = pd.read_sql_query("SELECT * FROM customers", conn)
        if not df.empty:
            record_to_delete = st.selectbox("Select record to delete", df.index)
            if st.button("Delete Selected Record"):
                conn.execute(f"DELETE FROM customers WHERE id = ?", (df.iloc[record_to_delete]['id'],))
                conn.commit()
                st.success("Record deleted!")
        conn.close()

    # Add export functionality
    with st.expander("Export Data"):
        if st.button("Export to CSV"):
            conn = sqlite3.connect('bank_customers.db')
            df = pd.read_sql_query("SELECT * FROM customers", conn)
            conn.close()
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="customer_predictions.csv",
                    mime="text/csv"
                )

def show_model_performance():
    st.header('Neural Network Model Performance')
    st.write('Accuracy: 86.63%')
    st.write('ROC-AUC: 88.54%')
    
    st.subheader('Classification Report')
    st.text('''
              precision    recall  f1-score   support

           0       0.89      0.95      0.92     25185
           1       0.74      0.53      0.62      6493

    accuracy                           0.87     31678
    macro avg      0.82      0.74      0.77     31678
    weighted avg   0.86      0.87      0.86     31678
    ''')
    
    st.subheader('Confusion Matrix')
    conf_matrix = np.array([[23998, 1187], [3048, 3445]])  
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)
    
    st.subheader('Model Architecture')
    st.write('- Input Layer Size: 11')
    st.write('- Hidden Layers: (100, 50)')
    st.write('- Output Layer Size: 2')
    st.write('- Total Iterations: 26')
    st.write('- Early Stopping: Enabled')
    st.write('- Activation Function: ReLU')
    st.write('- Optimizer: Adam')

def show_segment_analysis():
    st.header("Customer Segment Analysis")
    st.subheader("Question: Which customer segments have the highest churn risk?")
    
    conn = sqlite3.connect('bank_customers.db')
    df = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    
    if not df.empty:
        # Create segment filters
        col1, col2 = st.columns(2)
        with col1:
            segment_by = st.multiselect(
                "Select Segmentation Criteria:",
                ["Age", "Geography", "Balance", "Credit Score"]
            )
        
        if segment_by:
            if "Age" in segment_by:
                df['Age_Group'] = pd.cut(df['age'], 
                                       bins=[0, 30, 40, 50, 60, 100],
                                       labels=['<30', '30-40', '40-50', '50-60', '60+'])
            
            if "Balance" in segment_by:
                df['Balance_Range'] = pd.qcut(df['balance'], 
                                            q=5, 
                                            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            if "Credit Score" in segment_by:
                df['Credit_Score_Range'] = pd.qcut(df['credit_score'], 
                                                  q=5,
                                                  labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            # Create dynamic grouping based on selected segments
            group_cols = []
            if "Age" in segment_by:
                group_cols.append('Age_Group')
            if "Geography" in segment_by:
                group_cols.append('geography')
            if "Balance" in segment_by:
                group_cols.append('Balance_Range')
            if "Credit Score" in segment_by:
                group_cols.append('Credit_Score_Range')
            
            # Calculate average churn probability by segments
            segment_analysis = df.groupby(group_cols)['churn_probability'].agg([
                'mean', 'count'
            ]).reset_index()
            
            # Visualization
            if len(group_cols) == 1:
                fig = px.bar(segment_analysis, 
                           x=group_cols[0], 
                           y='mean',
                           text=segment_analysis['count'].astype(str) + ' customers',
                           title=f"Average Churn Probability by {group_cols[0]}")
                st.plotly_chart(fig)
            elif len(group_cols) == 2:
                fig = px.density_heatmap(df, 
                                       x=group_cols[0], 
                                       y=group_cols[1],
                                       z='churn_probability',
                                       title=f"Churn Probability Heatmap")
                st.plotly_chart(fig)
            
            # Detailed statistics
            st.subheader("Detailed Segment Statistics")
            st.dataframe(segment_analysis)
    else:
        st.info("No data available for analysis!")

def show_feature_impact():
    st.header("Feature Impact Analysis")
    st.subheader("Question: What are the key factors influencing customer churn?")
    
    conn = sqlite3.connect('bank_customers.db')
    df = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    
    if not df.empty:
        # Feature correlation analysis
        numerical_cols = ['credit_score', 'age', 'tenure', 'balance', 
                         'num_products', 'estimated_salary', 'churn_probability']
        corr_matrix = df[numerical_cols].corr()
        
        # Correlation heatmap
        fig = px.imshow(corr_matrix,
                       labels=dict(color="Correlation"),
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig)
        
        # Interactive feature analysis
        st.subheader("Analyze Feature Impact")
        feature = st.selectbox(
            "Select feature to analyze:",
            ['credit_score', 'age', 'tenure', 'balance', 'num_products', 'estimated_salary']
        )
        
        fig = px.scatter(df, 
                        x=feature, 
                        y='churn_probability',
                        color='geography',
                        title=f"Impact of {feature} on Churn Probability")
        st.plotly_chart(fig)
        
        # What-if analysis
        st.subheader("What-If Analysis")
        st.write("Adjust feature values to see their impact on churn probability")
        
        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.slider('Credit Score', 300, 850, 600)
            age = st.slider('Age', 18, 100, 35)
            tenure = st.slider('Tenure', 0, 10, 5)
        
        with col2:
            balance = st.slider('Balance', 0, 250000, 50000)
            num_products = st.slider('Number of Products', 1, 4, 1)
            estimated_salary = st.slider('Estimated Salary', 0, 200000, 50000)
        
        if st.button("Calculate Impact"):
            # Create scenarios
            base_prediction = make_prediction(credit_score, age, tenure, balance, 
                                           num_products, True, True, estimated_salary,
                                           'France', 'Male')
            
            # Show impact
            st.write(f"Predicted Churn Probability: {base_prediction:.2%}")
            
            # Feature importance visualization based on your model
            feature_importance = {
                'Credit Score': 0.15,
                'Age': 0.12,
                'Tenure': 0.10,
                'Balance': 0.18,
                'Products': 0.08,
                'Salary': 0.14
            }
            
            fig = px.bar(x=list(feature_importance.keys()),
                        y=list(feature_importance.values()),
                        title="Feature Importance")
            st.plotly_chart(fig)
    else:
        st.info("No data available for analysis!")

def show_retention_strategy():
    st.header("Retention Strategy Analysis")
    st.subheader("Question: How can we identify and target high-risk customers for retention?")
    
    conn = sqlite3.connect('bank_customers.db')
    df = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    
    if not df.empty:
        # Risk categorization
        df['risk_category'] = pd.cut(df['churn_probability'],
                                   bins=[0, 0.3, 0.7, 1],
                                   labels=['Low Risk', 'Medium Risk', 'High Risk'])
        
        # Risk distribution
        fig = px.pie(df, 
                    names='risk_category',
                    title="Customer Risk Distribution")
        st.plotly_chart(fig)
        
        # High-risk customer profile
        st.subheader("High-Risk Customer Profile")
        high_risk = df[df['risk_category'] == 'High Risk']
        
        if not high_risk.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Age", f"{high_risk['age'].mean():.0f}")
            with col2:
                st.metric("Average Balance", f"${high_risk['balance'].mean():,.2f}")
            with col3:
                st.metric("Average Tenure", f"{high_risk['tenure'].mean():.1f} years")
            
            # Retention recommendations
            st.subheader("Retention Recommendations")
            
            # Example of conditional recommendations
            if high_risk['balance'].mean() < df['balance'].mean():
                st.write("💰 Consider offering premium account benefits")
            if high_risk['tenure'].mean() < 3:
                st.write("🎯 Focus on early engagement programs")
            if high_risk['num_products'].mean() < 2:
                st.write("📦 Opportunity for product cross-selling")
            
            # Customer similarity analysis
            st.subheader("Similar Customer Analysis")
            customer_id = st.number_input("Enter Customer ID for similarity analysis", min_value=1)
            
            if st.button("Analyze Similar Customers"):
                customer = df[df['id'] == customer_id]
                if not customer.empty:
                    # Find similar customers based on features
                    similar_customers = df[
                        (df['age'].between(customer['age'].iloc[0] - 5, customer['age'].iloc[0] + 5)) &
                        (df['credit_score'].between(customer['credit_score'].iloc[0] - 50, customer['credit_score'].iloc[0] + 50)) &
                        (df['id'] != customer_id)
                    ]
                    
                    st.write(f"Found {len(similar_customers)} similar customers")
                    st.write("Average churn probability of similar customers:", 
                            f"{similar_customers['churn_probability'].mean():.2%}")
                else:
                    st.warning("Customer not found!")
        else:
            st.info("No high-risk customers identified!")
    else:
        st.info("No data available for analysis!")

def show_prediction_page():
    st.header('Individual Customer Churn Prediction')
    st.subheader("Question: What is the probability that a specific customer will churn?")
    
    # Add tabs for different prediction methods
    tab1, tab2 = st.tabs(["New Prediction", "Customer Search"])
    
    with tab1:
        # Demographic Information
        st.subheader("Customer Information")
        credit_score = st.slider('Credit Score', 300, 850, 600)
        age = st.slider('Age', 18, 100, 35)
        tenure = st.slider('Tenure', 0, 10, 5)
        balance = st.number_input('Balance', 0.0, 250000.0, 50000.0)
        num_products = st.slider('Number of Products', 1, 4, 1)
        has_credit_card = st.checkbox('Has Credit Card')
        is_active_member = st.checkbox('Is Active Member')
        estimated_salary = st.number_input('Estimated Salary', 0.0, 200000.0, 50000.0)
        geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
        gender = st.selectbox('Gender', ['Male', 'Female'])

        if st.button('Predict Churn'):
            result = make_prediction(credit_score, age, tenure, balance, num_products,
                                   has_credit_card, is_active_member, estimated_salary,
                                   geography, gender)
            
            # Show prediction with color coding and risk factors
            print("result",result)
            if result:
                col1, col2 = st.columns(2)
                with col1:
                    if result > 0.75:
                        st.error(f'High Risk - Churn Probability: {result:.2%}')
                    elif result > 0.50:
                        st.warning(f'Medium Risk - Churn Probability: {result:.2%}')
                    else:
                        st.success(f'Low Risk - Churn Probability: {result:.2%}')
                
                with col2:
                    st.subheader("Risk Factors")
                    if credit_score < 600:
                        st.write("⚠️ Low credit score")
                    if balance < 30000:
                        st.write("⚠️ Low balance")
                    if tenure < 2:
                        st.write("⚠️ New customer")
                    if not is_active_member:
                        st.write("⚠️ Inactive member")

    with tab2:
        st.subheader("Search Existing Customer")
        customer_id = st.number_input("Enter Customer ID", min_value=1, step=1)
        
        if st.button('Search'):
            conn = sqlite3.connect('bank_customers.db')
            query = "SELECT * FROM customers WHERE id = ? ORDER BY prediction_date DESC LIMIT 1"
            df = pd.read_sql_query(query, conn, params=[customer_id])
            conn.close()
            
            if not df.empty:
                st.subheader("Customer Details")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"Credit Score: {df['credit_score'].iloc[0]}")
                    st.write(f"Age: {df['age'].iloc[0]}")
                    st.write(f"Tenure: {df['tenure'].iloc[0]} years")
                    st.write(f"Balance: ${df['balance'].iloc[0]:,.2f}")
                    st.write(f"Number of Products: {df['num_products'].iloc[0]}")
                
                with col2:
                    st.write(f"Has Credit Card: {'Yes' if df['has_credit_card'].iloc[0] else 'No'}")
                    st.write(f"Active Member: {'Yes' if df['is_active_member'].iloc[0] else 'No'}")
                    st.write(f"Estimated Salary: ${df['estimated_salary'].iloc[0]:,.2f}")
                    st.write(f"Geography: {df['geography'].iloc[0]}")
                    st.write(f"Gender: {df['gender'].iloc[0]}")
                
                # Show prediction with color coding
                churn_prob = df['churn_probability'].iloc[0]
                st.subheader("Churn Analysis")
                if churn_prob > 0.7:
                    st.error(f"High Risk - Churn Probability: {churn_prob:.2%}")
                elif churn_prob > 0.3:
                    st.warning(f"Medium Risk - Churn Probability: {churn_prob:.2%}")
                else:
                    st.success(f"Low Risk - Churn Probability: {churn_prob:.2%}")
                
                st.write(f"Last Prediction: {df['prediction_date'].iloc[0]}")
            else:
                st.warning(f"No customer found with ID: {customer_id}")

if __name__ == '__main__':
    main()