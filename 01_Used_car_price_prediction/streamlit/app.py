#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 23:38:56 2023

@author: sergeypervov
"""

import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
from sklearn.compose import TransformedTargetRegressor
from catboost import CatBoostRegressor

st.title('Предсказание цены б/у авто')
st.header('по вашим данным')

df = pd.read_csv('train_st.csv')
df = df.drop([
    'vin', 'saledate', 'sale_year', 'ct_make', 'ct_model', 'ct_trim', 'ct_state',
    'ct_seller', 'ct_color', 'ct_interior', 'ct_region', 'ct_country', 'vin_model',
    'vin_body', 'vin_year', 'age', 'region', 'country'
],
    axis=1)

year = np.unique(df.year)
brand = np.unique(df.make.astype('str'))
model = np.unique(df.model.astype('str'))
trim = np.unique(df.trim.astype('str'))
body = np.unique(df.body.astype('str'))
odo = list(range(0, 400001, 5000))
cond = list(range(1, 6, 1))

with open('model_st.pkl', 'rb') as f_in:
    clf = pickle.load(f_in)

st.sidebar.markdown('Загрузите данные, чтобы получить прогноз')

select_event_2 = st.sidebar.selectbox(
    'Какая марка вашего авто?', brand.tolist())
select_event_3 = st.sidebar.selectbox(
    'Какая модель вашего авто?', model.tolist())
select_event_4 = st.sidebar.selectbox(
    'Какая комплектация вашего авто?', trim.tolist())
select_event_5 = st.sidebar.selectbox(
    'Какой кузов у вашего авто?', body.tolist())
select_event_6 = st.sidebar.selectbox(
    'Какой год вашего авто?', year.tolist())
select_event_7 = st.sidebar.selectbox(
    'Какой пробег вашего авто?', odo)
select_event_8 = st.sidebar.selectbox(
    'Какое состояние вашего авто?', cond)

selected = {
    'year': [select_event_6], 'make': [select_event_2], 'model': [select_event_3],
    'trim': [select_event_4], 'body': [select_event_5], 'condition': [select_event_8],
    'odometer': [select_event_7] 
    } 
 
features_st = pd.DataFrame(selected)
    
st.header('Статистика по ' + str(select_event_2))
dfX = df[df.make == select_event_2]
dfX = dfX.dropna()
st.write(dfX.describe())

fig, ax = plt.subplots()
ax = sns.scatterplot(data = dfX, x='year', y='sellingprice')
st.header('График отношения цены к году')
st.pyplot(fig)

fig, ax = plt.subplots()
ax = sns.scatterplot(data = dfX, x='odometer', y='sellingprice')
st.header('График отношения цены к пробегу')
st.pyplot(fig)

fig, ax = plt.subplots()
ax = sns.scatterplot(data = dfX, x='condition', y='sellingprice')
st.header('График отношения цены к состоянию')
st.pyplot(fig)

fig = px.scatter(
    dfX,
    x='year',
    y='sellingprice',
    size='condition',
    color='model',
    hover_name='seller',
    log_x=True,
    size_max=60
    )

tab1,tab2 = st.tabs(['Streamlit theme (default)', 'Plotly native theme'])

with tab1:
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)
with tab2:
    st.plotly_chart(fig, theme=None, use_container_width=True)


with st.sidebar:
    if st.button('Оценить!'):
        predicted = clf.predict(features_st)
        st.write('Ориентировочная стоимость Вашего авто составит: ', predicted)
    else:
        st.write('Нажмите на кнопку для расчета')
        









