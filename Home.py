import streamlit as st

# Bibliotecas auxiliares
import geopandas as gpd
import numpy as np
import pandas as pd


from notebooks.src.config import DADOS_GEO_MEDIAN, DADOS_LIMPOS, MODELO_FINAL

from joblib import load


# funções que persintem com o cachê
@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)

@st.cache_data
def carregar_dados_geo():
    return gpd.read_parquet(DADOS_GEO_MEDIAN)

@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)


df = carregar_dados_limpos()
geo = carregar_dados_geo()
modelo = carregar_modelo()


st.title("Previsão de preços de imóveis")

# valores de entrada

condados = list(geo['name'].sort_values())
selecionar_condado = st.selectbox("Condado", condados)

longitude = geo.query("name == @selecionar_condado")["longitude"].values
latitude = geo.query("name == @selecionar_condado")["latitude"].values

housing_median_age = st.number_input('Idade do imóvel', value=10,
                                     min_value=0, max_value=50)

# total_rooms = st.number_input('Total de cômodos', value=800)
total_rooms = geo.query("name == @selecionar_condado")["total_rooms"].values

# total_bedrooms = st.number_input('Total de quartos', value=100)
total_bedrooms = geo.query("name == @selecionar_condado")["total_bedrooms"].values

# population = st.number_input('População', value=300)
population = geo.query("name == @selecionar_condado")["population"].values

# households = st.number_input('Domicílio', value=100)
households = geo.query("name == @selecionar_condado")["households"].values

median_income = st.slider("Renda média (em milhares de US$)", 
                          5, 100, 45, 5)
median_income_10 = median_income / 10

# ocean_proximity = st.selectbox("Proximidade do oceano", df['ocean_proximity'].unique())
ocean_proximity = geo.query("name == @selecionar_condado")['ocean_proximity'].values


# median_income_cat = st.number_input('Categoria de renda', value=4)
bins_income = [0, 1.5, 3, 4.5, 6, np.inf]
median_income_cat = np.digitize(median_income_10, 
                                bins=bins_income)


# rooms_per_household = st.number_input("Quartos por dominílio", value=7)
rooms_per_household = geo.query("name == @selecionar_condado")['rooms_per_household'].values

# bedrooms_per_room = st.number_input("Quartos por cômodo", value=0.2)
bedrooms_per_room = geo.query("name == @selecionar_condado")['bedrooms_per_room'].values

# population_per_household = st.number_input("Pessoas por dominílio", value=2)
population_per_household = geo.query("name == @selecionar_condado")['population_per_household'].values


entrada_modelo = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income_10,
    "ocean_proximity": ocean_proximity,
    "median_income_cat": median_income_cat,
    "rooms_per_household": rooms_per_household,
    "bedrooms_per_room": bedrooms_per_room,
    "population_per_household": population_per_household,
}

df_entrada_modelo = pd.DataFrame(entrada_modelo, 
                                 index=[0])

botao_previsao = st.button("Prever Preço")

if botao_previsao:
    preco = modelo.predict(df_entrada_modelo)
    st.write(f"Preço previsot: US$ {round(preco[0][0], -2):,.0f}")
