import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set the title of the app
st.title("Clustering Model Deployment with Streamlit")
# Load the pre-trained clustering model
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('model.pkl')
    return model
model = load_model()
# Sidebar for user input
st.sidebar.header("Input Features")
def user_input_features():
    # Replace these with the actual features your model expects
    ICD_DGNS_CD1=st.sidebar.text_input('ICD_DGNS_CD1',value=None)
    ICD_DGNS_CD2=st.sidebar.text_input('ICD_DGNS_CD2',value=None)
    ICD_DGNS_CD3=st.sidebar.text_input('ICD_DGNS_CD3',value=None)
    ICD_DGNS_CD4=st.sidebar.text_input('ICD_DGNS_CD4',value=None)
    ICD_DGNS_CD5=st.sidebar.text_input('ICD_DGNS_CD5',value=None)
    ICD_DGNS_CD6=st.sidebar.text_input('ICD_DGNS_CD6',value=None)
    ICD_DGNS_CD7=st.sidebar.text_input('ICD_DGNS_CD7',value=None)
    ICD_DGNS_CD8=st.sidebar.text_input('ICD_DGNS_CD8',value=None)
    ICD_DGNS_CD9=st.sidebar.text_input('ICD_DGNS_CD9',value=None)
    ICD_DGNS_CD10=st.sidebar.text_input('ICD_DGNS_CD10',value=None)
    ICD_DGNS_CD11=st.sidebar.text_input('ICD_DGNS_CD11',value=None)
    ICD_DGNS_CD12=st.sidebar.text_input('ICD_DGNS_CD12',value=None)
    ICD_DGNS_CD13=st.sidebar.text_input('ICD_DGNS_CD13',value=None)
    ICD_DGNS_CD14=st.sidebar.text_input('ICD_DGNS_CD14',value=None)
    ICD_DGNS_CD15=st.sidebar.text_input('ICD_DGNS_CD15',value=None)
    ICD_DGNS_CD16=st.sidebar.text_input('ICD_DGNS_CD16',value=None)
    ICD_DGNS_CD17=st.sidebar.text_input('ICD_DGNS_CD17',value=None)
    ICD_DGNS_CD18=st.sidebar.text_input('ICD_DGNS_CD18',value=None)
    ICD_DGNS_CD19=st.sidebar.text_input('ICD_DGNS_CD19',value=None)
    ICD_DGNS_CD20=st.sidebar.text_input('ICD_DGNS_CD20',value=None)
    ICD_DGNS_CD21=st.sidebar.text_input('ICD_DGNS_CD21',value=None)
    ICD_DGNS_CD22=st.sidebar.text_input('ICD_DGNS_CD22',value=None)
    ICD_DGNS_CD23=st.sidebar.text_input('ICD_DGNS_CD23',value=None)
    ICD_DGNS_CD24=st.sidebar.text_input('ICD_DGNS_CD24',value=None)
    ICD_DGNS_CD25=st.sidebar.text_input('ICD_DGNS_CD25',value=None)
    CLM_TOT_CHRG_AMT=st.sidebar.number_input('CLM_TOT_CHRG_AMT',value=None)

    data = {
        'ICD_DGNS_CD1':ICD_DGNS_CD1,
        'ICD_DGNS_CD2':ICD_DGNS_CD2,
        'ICD_DGNS_CD3':ICD_DGNS_CD3,
        'ICD_DGNS_CD4':ICD_DGNS_CD4,
        'ICD_DGNS_CD5':ICD_DGNS_CD5,
        'ICD_DGNS_CD6':ICD_DGNS_CD6,
        'ICD_DGNS_CD7':ICD_DGNS_CD7,
        'ICD_DGNS_CD8':ICD_DGNS_CD8,
        'ICD_DGNS_CD9':ICD_DGNS_CD9,
        'ICD_DGNS_CD10':ICD_DGNS_CD10,
        'ICD_DGNS_CD11':ICD_DGNS_CD11,
        'ICD_DGNS_CD12':ICD_DGNS_CD12,
        'ICD_DGNS_CD13':ICD_DGNS_CD13,
        'ICD_DGNS_CD14':ICD_DGNS_CD14,
        'ICD_DGNS_CD15':ICD_DGNS_CD15,
        'ICD_DGNS_CD16':ICD_DGNS_CD16,
        'ICD_DGNS_CD17':ICD_DGNS_CD17,
        'ICD_DGNS_CD18':ICD_DGNS_CD18,
        'ICD_DGNS_CD19':ICD_DGNS_CD19,
        'ICD_DGNS_CD20':ICD_DGNS_CD20,
        'ICD_DGNS_CD21':ICD_DGNS_CD21,
        'ICD_DGNS_CD22':ICD_DGNS_CD22,
        'ICD_DGNS_CD23':ICD_DGNS_CD23,
        'ICD_DGNS_CD24':ICD_DGNS_CD24,
        'ICD_DGNS_CD25':ICD_DGNS_CD25,
        'CLM_TOT_CHRG_AMT':CLM_TOT_CHRG_AMT
        # 'Feature1': feature1,
        # 'Feature2': feature2,
        # # Add more features as needed
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
try:
    # Display user input
    st.subheader('User Input Features')
    st.write(input_df)
    # Preprocess input if necessary
    unique_codes = pd.Series(pd.unique(input_df.filter(regex='ICD_DGNS_CD').values.ravel('K'))).dropna()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[unique_codes])
    encoded_dfs = []
    for column in input_df.filter(regex='ICD_DGNS_CD').columns:
        encoded_data = encoder.fit_transform(input_df[[column]])
        col_names = [name.split('_')[-1] for name in encoder.get_feature_names_out([column])]
        encoded_dfs.append(pd.DataFrame(encoded_data, columns=col_names))
    encoded_df = pd.concat(encoded_dfs, axis=1).groupby(level=0, axis=1).max()
    encoded_df = encoded_df.reset_index(drop=True)
    features_df = pd.concat([encoded_df.reset_index(drop=True), input_df[['CLM_TOT_CHRG_AMT']]], axis=1)
    master_feature_df=pd.read_csv('FeatureNames.csv')
    cols=features_df.columns
    mastercols=master_feature_df.FeatureName.values
    for col in mastercols:
        if col not in cols:
            features_df[col]=0
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(features_df)
    # Predict cluster

    cluster = model.predict(scaled_input)
    st.subheader('Cluster Prediction')
    st.write(f'The input data belongs to cluster **{cluster[0]}**.')
    # Function to save or append data to CSV
    def save_to_csv(data, filename='Flag.csv'):
        if os.path.exists(filename):
            data.to_csv(filename, mode='a', header=False, index=False)
        else:
            data.to_csv(filename, index=False)

    def display_csv(file_name):
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            st.subheader('Flag Content:')
            st.write(df)
        else:
            st.warning(f'{file_name} does not exist.')
    # Save to CSV button
    if st.button('Flags'):
        input_df['Cluster'] = cluster[0]
        save_to_csv(input_df)
        st.success('Data saved to Flag.csv.')
    

    # Example to display the clustered_data.csv
    if st.button('ShoeFlagData'):
        display_csv('Flag.csv')
    
except:
    st.subheader('Cluster Prediction')
    st.write(f'Please enter valid data')