import streamlit as st

# Bibliotecas auxiliares
import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import shapely.geometry


from notebooks.src.config import DADOS_GEO_MEDIAN, DADOS_LIMPOS, MODELO_FINAL

from joblib import load


# funções que persintem com o cachê
@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)

@st.cache_data
def carregar_dados_geo():

    geo = gpd.read_parquet(DADOS_GEO_MEDIAN)

    # Explode MultiPolygons into individual polygons
    geo = geo.explode(ignore_index=True)

    # Function to check and fix invalid geometries
    def fix_and_orient_geometry(geometry):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)  # Fix invalid geometry
        # Orient the polygon to be counter-clockwise if it's a Polygon or MultiPolygon
        if isinstance(
            geometry, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)
        ):
        # if isinstance(
        #     geometry, (shapely.geometry.Polygon)
        # ):
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        return geometry

    # Apply the fix and orientation function to geometries
    geo["geometry"] = geo["geometry"].apply(fix_and_orient_geometry)

    # Extract polygon coordinates
    def get_polygon_coordinates(geometry):
        return (
            [[[x, y] for x, y in geometry.exterior.coords]]
            if isinstance(geometry, shapely.geometry.Polygon)
            else [
                [[x, y] for x, y in polygon.exterior.coords]
                for polygon in geometry.geoms
            ]
        )

    # Apply the coordinate conversion and store in a new column
    geo["geometry"] = geo["geometry"].apply(get_polygon_coordinates)

    return geo

@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)


df = carregar_dados_limpos()
geo = carregar_dados_geo()
modelo = carregar_modelo()


st.title("Previsão de preços de imóveis")

coluna_1, coluna_2 = st.columns(2)

# valores de entrada

with coluna_1:

    with st.form(key="formulario"):

        condados = sorted(geo['name'].unique())
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

        # df_entrada_modelo = pd.DataFrame(entrada_modelo, 
        #                                 index=[0])

        df_entrada_modelo = pd.DataFrame(entrada_modelo)

        botao_previsao = st.form_submit_button("Prever Preço")

        if botao_previsao:
            preco = modelo.predict(df_entrada_modelo)
            # st.write(f"Preço previsot: US$ {round(preco[0][0], -2):,.0f}")
            st.metric(f"Preço previsto: ", value=f"US$ {round(preco[0][0], -2):,.0f}")


with coluna_2:
    
    view_state = pdk.ViewState(
        latitude=float(latitude[0]), longitude=float(longitude[0]),
        zoom=7, min_zoom=5, max_zoom=10
    )

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=geo[["name", "geometry"]],
        get_polygon='geometry',
        get_fill_color=[0, 0, 255, 100], #RGB
        get_line_color=[255, 255, 255], #RGB
        get_line_width=50,
        pickable=True, auto_highlight = True
    )

    condado_selecionado = geo.query("name==@selecionar_condado")
    highlight_layer = pdk.Layer(
        "PolygonLayer",
        data=condado_selecionado[["name", "geometry"]],
        get_polygon='geometry',
        get_fill_color=[0, 0, 255, 100], #RGB
        get_line_color=[255, 255, 255], #RGB
        get_line_width=50,
        pickable=True, auto_highlight = True
    )

    tooltip = {
        "html" : "<b>Condado:</b> {name}",
        "style" : {"backgroundColor": "steelblue",
                   "color": "white",
                   'fontsize' : '10px'}
    }

    mapa = pdk.Deck(
        initial_view_state=view_state,
        # map_style='light',
        layers=[polygon_layer, highlight_layer],
        tooltip=tooltip
    )

    st.pydeck_chart(mapa)