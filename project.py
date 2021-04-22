# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 21:44:26 2021

@author: pooja
"""

import streamlit as st

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')

dataset_loc = "effort.csv"

@st.cache
def load_data(dataset_loc):
    missing_values = ["?","-","n/a"]
    df = pd.read_csv(dataset_loc,na_values = missing_values)
    return df

def load_sidebar(df):
    st.sidebar.subheader("Efforts calculator")
    st.sidebar.success(' To download the dataset :https://drive.google.com/file/d/17yZ1NSSsRrDF7qfGOJyGHRDDiBce7EQN/view')
    st.sidebar.info('Detailed Description about the dataset : http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html')
    st.sidebar.success('For EDA, Model Building and Web App code : https://github.com/pooja1207/CIPHERSCHOOLS__ML1 ')
    st.sidebar.warning('Made by Prayas :heart:')
    
def load_description(df):
    # Preview of the dataset
        st.header("Data Preview")
        preview = st.radio("Choose Head/Tail?", ("Top", "Bottom"))
        if(preview == "Top"):
            st.write(df.head())
        if(preview == "Bottom"):
            st.write(df.tail())

        # display the whole dataset
        if(st.checkbox("Show complete Dataset")):
            df.isnull().sum()
            st.write(df)
            
        if(st.checkbox("Show data description")):
            st.write(df.describe(include = "all"))

        # Show shape
        if(st.checkbox("Display the shape")):
            st.write(df.shape)
            dim = st.radio("Rows/Columns?", ("Rows", "Columns"))
            if(dim == "Rows"):
                st.write("Number of Rows", df.shape[0])
            if(dim == "Columns"):
                st.write("Number of Columns", df.shape[1])

        # show columns
        if(st.checkbox("Show the Columns")):
            st.write(df.columns)
   
def load_heatmap(df):
    st.image("heatmap.png", use_column_width = True)
    
def load_jointplot(df):
    sns.jointplot(x="Effort", y='Length', data=df, kind = 'kde')


def preprocessor(l):
    missing_values = ["?","-","n/a"]
    df = pd.read_csv(dataset_loc,na_values = missing_values)
    name=df.columns
    x1=df.drop(["Effort"],axis=1)
    y1=df["Effort"]
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.33, random_state=22)
    regressor = LinearRegression()
    regressor.fit(x1_train, y1_train)
    x1_test.loc[len(x1_test.index)] = l
    predict = regressor.predict(x1_test) 
    return predict[-1]
        

def load_predictor(df):
    l=[82,82]
    
    st.subheader('Man-Hours Predictor')
    st.info("Team Experience")
    
    team_exp = st.slider("",0,5,1)
    l.append(team_exp)
    
    st.info("Manager Experience")
    mang_exp = st.slider("",0,4,1)
    l.append(mang_exp)
    
    st.info("Year-End")
    year_end = st.number_input("Year-End")
    l.append(year_end)
    
    st.info("Time taken to complete project")
    time=st.number_input("Time taken to complete project")
    l.append(time)
    
    st.info("Language")
    language = st.radio("",("1","2","3"))
    l.append(language)
    
    st.info("Transactions")
    transactions=st.number_input("Transactions")
    l.append(transactions)
    
    st.info("Entities")
    entities=st.number_input("Entities")
    l.append(entities)
    
    st.info("Adjustment :- PointsNonAdjust, Adjustment, PointAdjust")
    pn=st.number_input("PointsNonAdjust")
    adjust=st.number_input("Adjustment")
    py=st.number_input("PointAdjust")
    l.append(pn)
    l.append(adjust)
    l.append(py)
    
    result=preprocessor(l)
    if st.button('Estimated Efforts'):
        st.write(result)
    
    
    
def main(): 
    df = load_data(dataset_loc)
    load_sidebar(df)
    st.title('Efforts Calculator')
    st.image('3.jpg',use_column_width=True)
    st.text('Estimating man-hours or man-days for designing a project')
    
    add_selectbox = st.selectbox('',('Choose an option', 'Home', 'Data Description', 'Heat Map', 'Joint Plot', 'Effort Calculator'))
    if(add_selectbox == 'Home'):
        pass
    elif(add_selectbox == 'Data Description'):
        load_description(df)
    elif(add_selectbox == 'Heat map'):
        load_heatmap(df)
    elif(add_selectbox == 'Joint Plot'):
        load_jointplot(df)
    elif(add_selectbox == 'Effort Calculator'):
        load_predictor(df)




if __name__=='__main__':
    main()