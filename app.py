import streamlit as st,pandas as pd,plotly.express as px
from sklearn.cluster import KMeans
df=pd.read_csv('data/customer_data.csv')
df['Segment']=KMeans(n_clusters=3,n_init=10,random_state=42).fit_predict(df[['AnnualIncome','SpendingScore']])
st.title('Customer Segmentation Dashboard')
st.dataframe(df)
st.plotly_chart(px.scatter(df,x='AnnualIncome',y='SpendingScore',color=df['Segment'].astype(str)))
