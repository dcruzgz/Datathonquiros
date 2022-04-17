import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import json
import folium
import numpy as np
from pyvis.network import Network
import altair as alt
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from collections import Counter
from itertools import groupby



#Páginas de la aplicación web

PAGES = [
    'Nivel geográfico y temporal',
    'Nivel de producto',
    'Nivel de cliente'    
]


#Configuración de la página
st.set_page_config(
    page_title="Datathonquiros",
    page_icon=":brain:",
    initial_sidebar_state="expanded",
    layout = 'wide',
        menu_items={
         'About': " Datathonquiros: Daniel Jesús Cruz Garzón Antonio González Rodríguez"
     }
)


#Extracción de los datasheet, almacenados en caché para mejorar la experiencia de usuario

@st.experimental_memo(ttl=30)
def get_data_clean():
    data = pd.read_csv('https://www.dropbox.com/s/bj5mnpqkq70itd1/tickets_data.csv?dl=1'
                         ,header=0, encoding="ISO-8859-1")  # read a CSV file inside the 'data" folder next to 'app.py'
    return data

@st.experimental_memo(ttl=30)
def get_data_geo():
    data = json.load(
        open('Data/spain_provinces.geojson',
             encoding="utf8"))
    return data


@st.experimental_memo(ttl=30)
def get_data_pobl():
    data_raw = pd.read_csv('Data/poblacion.csv', sep=";"
                       , header=0, encoding="ISO-8859-1")  # read a CSV file inside the 'data" folder next to 'app.py'
    data_raw['Provincias'] = data_raw['Provincias'].str[:2]
    data_raw = data_raw.sort_values('Provincias')
    data_raw = pd.DataFrame(data_raw.iloc[0:52, [0, 3]])
    data_raw['CODIGO'] = data_raw['Provincias'].astype(int)
    data_raw['Total'] = data_raw['Total'].str.replace('.','', regex=False)
    data_raw['Total'] = data_raw['Total'].astype(int)
    return data_raw.iloc[:, [1, 2]]

@st.experimental_memo(ttl=30)
def get_data_prov():
    data = pd.read_excel(
        'Data/codprov.xls')
    data_pobl = get_data_pobl()
    data = data.merge(data_pobl, on="CODIGO", how="left")
    return data

@st.experimental_memo(ttl=30)
def get_descuentos():
    data = pd.read_csv('Data/descuentos.csv', sep=","
                           , header=0,
                           encoding="ISO-8859-1")

    return data

@st.experimental_memo(ttl=30)
def get_rules():
    url = 'Data/rules.csv' #Datos del Arules
    df_raw = pd.read_csv(url, encoding="UTF-8") 
    return df_raw
    
#Fin de funciones de extracción de los datos

def pretty(s: str) -> str:
    try:
        return dict(js="JavaScript")[s]
    except KeyError:
        return s.capitalize()


#region RULES

#Funciones para mostrar las asociaciones entre productos

def df_rules(min, max):
    df1 = get_rules()[["lhs", "rhs", "confidence"]]

    df_rules = pd.DataFrame()

    df_rules["Si el cliente compró"] = df1.loc[:, "lhs"].str.replace(r'[{]', '', regex=True).replace(r'[}]', '',
                                                                                                     regex=True)
    df_rules["Comprará"] = df1.loc[:, "rhs"].str.replace(r'[{]', '', regex=True).replace(r'[}]', '', regex=True)
    df_rules["Con una confianza del (%)"] = round(pd.to_numeric(df1["confidence"]) * 100, 2)
    df_rules = df_rules.loc[df_rules.loc[:, 'Con una confianza del (%)'] >= min]
    df_rules = df_rules.loc[ df_rules.loc[:, 'Con una confianza del (%)'] <= max]
    return df_rules

def rules(df_rules):
    # Relaciones gráficas

    r_net = Network(height='450px', width='1000px', bgcolor='#FFFFFF.', font_color='black')
            
    # set the physics layout of the network
    r_net.barnes_hut(spring_length=1,  central_gravity=10, overlap=0.3)
    sources = df_rules['Si el cliente compró']
    targets = df_rules['Comprará']
    weights = df_rules['Con una confianza del (%)']

    edge_data = zip(sources, targets, weights)

    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2]

        r_net.add_node(src, src, title=src, size=10, color='#EF7E62')
        r_net.add_node(dst, dst, title=dst, size=25, color='#93C9F7')
        r_net.add_edge(src, dst, value=w, size=25, color='#E2DCDB')

    neighbor_map = r_net.get_adj_list()

    # add neighbor data to node hover data
    for node in r_net.nodes:
        node['title'] += ' Relacionado:<br>' + '<br>'.join(neighbor_map[node['id']])
        node['value'] = len(neighbor_map[node['id']])

    r_net.save_graph('rules.html')

    HtmlFile = open("rules.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    return source_code
#endregion


# Run de la app

def run_UI():
    
    if st.session_state.page:
        with st.sidebar:
            image = Image.open('Data/mifarma.png')
            st.image(image)
            page = st.radio('Selecciona el nivel de análisis: ', PAGES, index=st.session_state.page)

    else:
        with st.sidebar:
            image = Image.open('Data/mifarma.png')
            st.image(image)
            page = st.selectbox('Selecciona el nivel de análisis: ', PAGES, index=0)
            

    #Página MAPA
    
    if page == 'Nivel geográfico y temporal':
             
        #Creación de los DataFrames
        datos_clean_or = get_data_clean() #General con los datos de los tickets
        data_code = get_data_prov() #Datos de las provincias (código y población)
        data_geo = get_data_geo()   #Datos de las coordenadas de las provincias



        # A partir del DataFrame original los NA en categoría de producto pasan a designase 'Sin Clasificar'

        datos_clean_or['productcat1'] = datos_clean_or['productcat1'].fillna('Sin clasificar')
        datos_clean_or['productcat2'] = datos_clean_or['productcat2'].fillna('Sin clasificar')
        datos_clean_or['productcat3'] = datos_clean_or['productcat3'].fillna('Sin clasificar')

        # DataFrame donde descartamos los datos sin código postal sólo para la representación del mapa y gráficas de provincias
        datos_clean_map = datos_clean_or[datos_clean_or['zp_sim'].notna()]


        dat_1 = data_code.sort_values('CODIGO')
        dat_1['cod_prov'] = data_code['CODIGO'].astype(int).astype(str)
        dat_1['cod_prov'] = dat_1['cod_prov']
        data_all = dat_1.set_index('CODIGO')

        #Tipos de variables para el mapa 

        dicts = {"Balance total (€)": 'GAIN',
                 "Balance relativo (€/100 mil hab.)": 'GAIN'}


        # Creación del mapa con folium
        map_sby = folium.Map(tiles='OpenStreetMap', location=[40.15775718967773, -3.9205941038156285], zoom_start=5.5)
        folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(map_sby)

        st.sidebar.write("""
          ## Nivel geográfico y temporal
          :pushpin: En esta página puedes visualizar las ventas de Atida Mifarma a nivel provincial. \n
          :calendar: Selecciona las categorías que te resulten de interés, y observa como cambian los balances de las ventas según el territorio y el mes. \n
          :man-woman-girl-boy:  Puedes consultar el balance de las ventas ajustado a los habitantes de cada provincia seleccionando 
          'Balance relativo (€/100 mil hab.)' como 'Dato mostrado', o el balance total generado en ese territorio, 
          seleccionando 'Balance total (€)' en ese mismo apartado. \n
          :chart_with_upwards_trend: Más abajo podrás encontrar la evolución temporal de la categoría seleccionada en 
          las provincias que desees y en todo el conjunto del territorio español. 
          """)
        st.title(":earth_africa: Ventas a nivel geográfico y temporal :clock130:")
        my_expander = st.expander(label='Filtros de la búsqueda', expanded = True)
        with my_expander:
            cols = st.columns((1, 1, 1))

            #SELECCION DE FECHAS
            month = datos_clean_map['Month'].unique()
            month = np.append(month, ['Todo el año'])

            year_1 = datos_clean_map['Year'].unique()
            year_1 = np.append(year_1, ['Todos los años'])

            mes = cols[0].selectbox("Mes:",
                              month, index=len(month)-1)
            year = cols[1].selectbox("Año:",
                              year_1, index=len(year_1)-1)

            variable_map = cols[2].selectbox("Dato mostrado:",
                                             ("Balance total (€)", "Balance relativo (€/100 mil hab.)"), index=1)

            prod1 = datos_clean_map['productcat1'].unique()
            prod1 = np.append(prod1, ['Todas las Categorías'])

            cat1 = cols[0].selectbox("Categoría:",
                                     prod1, index=len(prod1) - 1)

            df_cat1 = datos_clean_map.loc[datos_clean_map.loc[:, 'productcat1'] == cat1]
            prod2 = df_cat1['productcat2'].unique()
            prod2 = np.append(prod2, ['Toda la Categoría'])

            cat2 = cols[1].selectbox("Subcategoría 1:",
                                     prod2, index=len(prod2) - 1)

            df_cat2 = datos_clean_map.loc[datos_clean_map.loc[:, 'productcat2'] == cat2]
            prod3 = df_cat2['productcat3'].unique()
            prod3 = np.append(prod3, ['Toda la Categoría'])
            cat3 = cols[2].selectbox("Subcategoría 2:",
                                     prod3, index=len(prod3) - 1)

            if mes == 'Todo el año' and year != 'Todos los años':
                datos_clean = datos_clean_map[datos_clean_map['Year'] == int(year)]


            elif mes != 'Todo el año' and year == 'Todos los años':
                datos_clean = datos_clean_map[datos_clean_map['Month'] == int(mes)]

            elif mes == 'Todo el año' and year == 'Todos los años':
                datos_clean = datos_clean_map
            else:
                datos_clean = datos_clean_map[(datos_clean_map['Month'] == int(mes)) & (datos_clean_map['Year'] == int(year))]


            # Seleccion de Ganancias netas o por 100 mil habitantes 
            
            if variable_map == 'Balance total (€)':
                nombre_valor = "Balance (k€): "
            else:
                nombre_valor = " Balance relativo (€/100 mil hab.): "

            if cat1 == 'Todas las Categorías':

                if variable_map == 'Balance total (€)':
                    nombre_valor = "Balance (k€): "
                    df_sum = datos_clean.groupby(['zp_sim'])['Precio_calculado', 'productcat1'].sum() / 1000
                else:
                    nombre_valor = " Balance relativo (€/100 mil hab.):"
                    df_sum = datos_clean.groupby(['zp_sim'])['Precio_calculado', 'productcat1'].sum()
                    df_sum['Poblacion'] = data_code.sort_values('CODIGO').set_index('CODIGO')['Total']
                    df_sum['Precio_calculado'] = (df_sum['Precio_calculado'] / df_sum['Poblacion'])*100000

                data_all['GAIN'] = df_sum['Precio_calculado']
            
            #Seleccion de categorías
            
            else:
                df_va1 = datos_clean.loc[datos_clean.loc[:, 'productcat1'] == cat1]

                if cat2 == 'Toda la Categoría':

                    if variable_map == 'Balance total (€)':
                        nombre_valor = "Balance (k€): "
                        df_sum = df_va1.groupby(['zp_sim'])['Precio_calculado', 'productcat1'].sum() / 1000
                    else:
                        nombre_valor = " Balance relativo (€/100 mil hab.):"
                        df_sum = df_va1.groupby(['zp_sim'])['Precio_calculado', 'productcat1'].sum()
                        df_sum['Poblacion'] = data_code.sort_values('CODIGO').set_index('CODIGO')['Total']
                        df_sum['Precio_calculado'] = (df_sum['Precio_calculado'] / df_sum['Poblacion'])*100000

                    data_all['GAIN'] = df_sum['Precio_calculado']

                else:

                    if variable_map == 'Balance total (€)':
                        nombre_valor = "Balance (k€): "
                        df_sum = df_va1.groupby(['zp_sim'])['Precio_calculado', 'productcat2'].sum() / 1000
                    else:
                        nombre_valor = " Balance relativo (€/100 mil hab.):"
                        df_sum = df_va1.groupby(['zp_sim'])['Precio_calculado', 'productcat2'].sum()
                        df_sum['Poblacion'] = data_code.sort_values('CODIGO').set_index('CODIGO')['Total']
                        df_sum['Precio_calculado'] = (df_sum['Precio_calculado'] / df_sum['Poblacion'])*100000


                    data_all['GAIN'] = df_sum['Precio_calculado']

                    df_va2 = datos_clean.loc[datos_clean.loc[:, 'productcat2'] == cat2]

                    if cat3 == 'Toda la Categoría':

                        if variable_map == 'Balance total (€)':
                            nombre_valor = "Balance (k€): "
                            df_sum = df_va2.groupby(['zp_sim'])['Precio_calculado', 'productcat2'].sum() / 1000
                        else:
                            nombre_valor = " Balance relativo (€/100 mil hab.):"
                            df_sum = df_va2.groupby(['zp_sim'])['Precio_calculado', 'productcat2'].sum()
                            df_sum['Poblacion'] = data_code.sort_values('CODIGO').set_index('CODIGO')['Total']
                            df_sum['Precio_calculado'] = (df_sum['Precio_calculado'] / df_sum['Poblacion'])*100000


                        data_all['GAIN'] = df_sum['Precio_calculado']
                    else:
                        df_va3 = df_va2.loc[datos_clean.loc[:, 'productcat3'] == cat3]

                        if variable_map == 'Balance total (€)':
                            nombre_valor = "Balance (k€): "
                            df_sum = df_va3.groupby(['zp_sim'])['Precio_calculado', 'productcat3'].sum()/ 1000
                        else:
                            nombre_valor = " Balance relativo (€/100 mil hab.):"
                            df_sum = df_va3.groupby(['zp_sim'])['Precio_calculado', 'productcat3'].sum()
                            df_sum['Poblacion'] = data_code.sort_values('CODIGO').set_index('CODIGO')['Total']
                            df_sum['Precio_calculado'] = (df_sum['Precio_calculado'] / df_sum['Poblacion'])*100000

                        data_all['GAIN'] = df_sum['Precio_calculado']
                        
            # Incorporamos los datos calculados de las ganancias al geojson con las coordenadas de cada provincia 
            for idx in range(51):
                if pd.isna(data_all['GAIN'][idx + 1]):
                    data_all['GAIN'][idx + 1] = 0
                data_geo['features'][idx]['properties']['GAIN'] = round(data_all['GAIN'][idx + 1], 3)
                data_geo['features'][idx]['properties']['cod_prov'] = data_all['cod_prov'][idx + 1]  # igualar los codigos los 0 a la izq dan problemas
                
                
       
        # Texto de la búsqueda
        
        if cat1 == 'Todas las Categorías':
            cat_txt = variable_map  + " en todas las categorías,"
        elif cat1 != 'Todas las Categorías' and cat2 == 'Toda la Categoría':
            cat_txt = variable_map + " en la categoría " + cat1
        elif cat1 != 'Todas las Categorías' and cat2 != 'Toda la Categoría' and cat3 == 'Toda la Categoría':
            cat_txt = variable_map + " en " + cat1 + " - " + cat2 
        else:
            cat_txt = variable_map + " en " + cat1 + " - " + cat2 + " - " + cat3
            
        if mes == 'Todo el año' and year == 'Todos los años':
            time_txt = " durante todo el registro temporal (2017/2018)"
        elif mes == 'Todo el año' and year != 'Todos los años':
            time_txt = " durante el año " + year
        elif mes != 'Todo el año' and year == 'Todos los años':
             time_txt = " durante el mismo mes " + "(" + mes + ")" + " en los años 2017 y 2018  "
        else:
            time_txt = " en el mes/año, "  + mes +"/"+ year
       
        st.write(cat_txt + time_txt)
        
        #Función que nos genera los colores para identificar las variables en cada provincia
        
        threshold_scale = np.linspace(data_all[dicts[variable_map]].min(),
                                  data_all[dicts[variable_map]].max(),
                                  10, dtype=float)
        # change the numpy array to a list
        threshold_scale = threshold_scale.tolist()
        threshold_scale[-1] = threshold_scale[-1]
        
        maps = folium.Choropleth(geo_data=data_geo,
                             data=data_all,
                             columns=['cod_prov',dicts[variable_map]],
                             key_on='feature.properties.cod_prov',
                             threshold_scale=threshold_scale,
                             fill_color='YlOrRd',
                             fill_opacity=0.7,
                             line_opacity=0.2,
                             highlight=True,
                             reset=True, 
                             prefer_canvas=True).add_to(map_sby)

        style_function = lambda x: {'fillColor': '#ffffff',
                                    'color': '#000000',
                                    'fillOpacity': 0.1,
                                    'weight': 0.1}
        highlight_function = lambda x: {'fillColor': '#000000',
                                        'color': '#000000',
                                        'fillOpacity': 0.50,
                                        'weight': 0.1}



        NIL = folium.features.GeoJson(
            data_geo,
            style_function=style_function,
            control=False,
            highlight_function=highlight_function,
            tooltip=folium.features.GeoJsonTooltip(
                fields=['name', 'GAIN'],
                aliases=['Provincia : ', nombre_valor],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
            )
        )
        map_sby.add_child(NIL)
        map_sby.keep_in_front(NIL)
        folium.LayerControl().add_to(map_sby)
        folium_static(map_sby, width=1030, height=500)
        
        
        ##GRAFICOS TEMPORALES

        datos_clean_map['date'] = datos_clean_map["Month"].astype(str) + "/" + datos_clean_map["Year"].astype(str)
        fechas = datos_clean_map['date'].unique()
        datos_clean_map['date'] = pd.to_datetime(datos_clean_map['date'])

        # Seleccion de categoría

        if cat1 == 'Todas las Categorías':
            df_sum = datos_clean_map.groupby(['zp_sim', 'date'])['Precio_calculado', 'productcat1'].sum()
            df_total = datos_clean_map.groupby(['date'])['Precio_calculado', 'productcat1'].sum()
        else:
            df_va1 = datos_clean_map.loc[datos_clean_map.loc[:, 'productcat1'] == cat1]
            if cat2 == 'Toda la Categoría':
                df_sum = df_va1.groupby(['zp_sim', 'date'])['Precio_calculado', 'productcat1'].sum()
                df_total = df_va1.groupby(['date'])['Precio_calculado', 'productcat1'].sum()
            else:
                df_va2 = df_va1.loc[datos_clean_map.loc[:, 'productcat2'] == cat2]

                if cat3 == 'Toda la Categoría':
                    df_sum = df_va2.groupby(['zp_sim', 'date'])['Precio_calculado', 'productcat2'].sum()
                    df_total = df_va2.groupby(['date'])['Precio_calculado', 'productcat2'].sum()
                else:
                    df_va3 = df_va2.loc[datos_clean_map.loc[:, 'productcat3'] == cat3]
                    df_sum = df_va3.groupby(['zp_sim', 'date'])['Precio_calculado', 'productcat3'].sum()
                    df_total = df_va3.groupby(['date'])['Precio_calculado', 'productcat2'].sum()

        array = df_sum.index  # Solo permitimos la selección de provincias que contienen datos
        codigos = []
        
        for e in array:
            codigos.append(int(e[0]))
        
        prov_ok = data_code.loc[data_code['CODIGO'].isin(codigos)]['LITERAL'].to_numpy()
        
        st.markdown('**Evolución temporal**')
        
        seleccion = st.multiselect(
            "Selecciona o elimina las provincias deseadas para consultar la evolución temporal. Las provincias no disponibles en la lista no han tenido ventas de la categoría seleccionada:", options=prov_ok,
            default=prov_ok, format_func=pretty #Seleccion por defecto de la provincia con mas meses de compra en dicha categoría
        )

        fig1 = go.Figure()
        fig1.add_hline(y=0)
        for provincia in seleccion:
            cd_prov = data_code.loc[data_code.loc[:, 'LITERAL'] == provincia]['CODIGO'].values[0]
            if variable_map == 'Balance total (€)':
                x_axis = df_sum.loc[cd_prov, :].index
                y_axis = df_sum.loc[cd_prov, :]["Precio_calculado"]
                fig1.add_trace(go.Scatter(x=x_axis, y=y_axis,
                                          mode='lines',
                                          name=provincia))
            else:
                poblacion = data_code.loc[data_code.loc[:, 'LITERAL'] == provincia]['Total'].values[0]
                x_axis = df_sum.loc[cd_prov, :].index
                y_axis = ((df_sum.loc[cd_prov, :]["Precio_calculado"]) / poblacion) * 100000
                fig1.add_trace(go.Scatter(x=x_axis, y=y_axis,
                                          mode='lines',
                                          name=provincia))
        fig1.update_layout(xaxis=dict(tickformat="%m-%Y"))
        fig1.update_layout(
            title="Evolución del " + cat_txt +" en las provincias seleccionadas",
            xaxis_title="Fecha",
            yaxis_title=variable_map,
            legend_title="Provincia"
        )

        st.plotly_chart(fig1, use_container_width=True)
        df = pd.DataFrame()

        df['Fecha'] = df_total.index
        df['Ganancia'] = df_total['Precio_calculado']

        # Ganancias en categoría en todo el territorio

        if variable_map == 'Balance total (€)':
            fig = px.line(df_total, x=df_total.index, y="Precio_calculado")

        else:
            fig = px.line(df_total, x=df_total.index, y=(df_total["Precio_calculado"] / 46722980) * 100000)

        fig.update_layout(xaxis=dict(tickformat="%m-%Y"))
        fig.update_layout(
            title="Evolución en todo el país del "  + cat_txt ,
            xaxis_title="Fecha",
            yaxis_title=variable_map
        )

        fig.add_hline(y=0)

        st.plotly_chart(fig, use_container_width=True)
         
    #Página sobre los productos y marcas
    
    elif page == 'Nivel de producto':
    
        #Creación de los DataFrames
        datos_clean_or = get_data_clean() #General con los datos de los tickets
        
        # A partir del DataFrame original los NA en categoría de producto pasan a designase 'Sin Clasificar'
        
        datos_clean_or['productcat1'] = datos_clean_or['productcat1'].fillna('Sin clasificar')
        datos_clean_or['productcat2'] = datos_clean_or['productcat2'].fillna('Sin clasificar')
        datos_clean_or['productcat3'] = datos_clean_or['productcat3'].fillna('Sin clasificar')
        
        st.sidebar.write("""
            ## Nivel de producto

            :star: En este apartado se muestra el balance de las ventas producidas durante los años 2017 y 2018 para cada tipo de producto. \n
            :bar_chart: Deslizando hacia abajo podrás consultar las marcas más vendidas, las marcas que mayor beneficio generan, y 
            las marcas que menor beneficio generan.
            Utiliza los filtros deseados para seleccionar las categorías de interés. \n
            :ok_hand: Al final de la página podrás observar el balance de las ventas 
            en la categoría seleccionada dependiendo del descuento ofrecido. 
          """)
        st.title(" :baby_bottle: :lipstick: :pill: Nivel de producto")
        
        descuentos = get_descuentos() #Para descuentos creación del DATAFRAME

        ##CATEGORIAS TREEMAP
        st.write("")
        st.markdown('**Balance de las ventas para cada categoría y subcategoría**')
        st.write('Puedes navegar por las categorías y subcategorías haciendo click en ellas para tener una vista mas detallada.')  
        
        df_categorias = datos_clean_or.groupby(['productcat1', 'productcat2', 'productcat3'])['Precio_calculado'].sum().reset_index(
            name='Balance (€)')

        df_categorias = df_categorias.loc[df_categorias['Balance (€)'] >= 0]
        df_categorias = df_categorias.loc[df_categorias['productcat1'] != 'Sin clasificar']
        df_categorias['Balance (€)'] = round(df_categorias['Balance (€)'], 0)
        df = pd.DataFrame(
            dict(productcat1=df_categorias['productcat1'],productcat2=df_categorias['productcat2'],
                 productcat3=df_categorias['productcat3'], Balance=df_categorias['Balance (€)'])
        )
        df["Todas"] = "Todas"  # in order to have a single root node
        fig = px.treemap(df, path=[px.Constant("Todas las categorías"), 'productcat1', 'productcat2', 'productcat3'], values='Balance'
                         )
        fig.update_traces(root_color="lightgrey", hovertemplate='Categoría=%{id}<br>Balance(€)=%{value:.0f}<extra></extra>')
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), width= 800, height= 700)
        st.plotly_chart(fig, use_container_width=True)

        
        st.write("")
        st.markdown('**Ventas según marcas**')

        #Como en mapa filtrar por categoría según marca_gain
        
        cols = st.columns((1, 1, 1))

        prod1 = datos_clean_or['productcat1'].unique()
        prod1 = np.append(prod1, ['Todas las Categorías'])

        cat1 = cols[0].selectbox("Categoría:",
                                 prod1,  index=len(prod1)-1)
        if cat1 == 'Todas las Categorías':
            cat2 = cols[1].selectbox("Subcategoría 1:",
                              ['Toda la Categoría'])
            cat3 = cols[2].selectbox("Subcategoría 2:",
                              ['Toda la Categoría'])
            df_filter = datos_clean_or

        else:
            df_va1 = datos_clean_or.loc[datos_clean_or.loc[:, 'productcat1'] == cat1]
            prod2 = df_va1['productcat2'].unique()
            prod2 = np.append(prod2, ['Toda la Categoría'])

            cat2 = cols[1].selectbox("Subcategoría 1:",
                                     prod2, index=len(prod2)-1)

            if cat2 == 'Toda la Categoría':

                df_filter = datos_clean_or.loc[datos_clean_or['productcat1'] == cat1]

                cat3 = cols[2].selectbox("Subcategoría 2:",
                                  ['Toda la Categoría'])
            else:
                df_va2 = datos_clean_or.loc[datos_clean_or.loc[:, 'productcat2'] == cat2]
                prod3 = df_va2['productcat3'].unique()


                prod3 = np.append(prod3, ['Toda la Categoría'])
                cat3 = cols[2].selectbox("Subcategoría 2:",
                                         prod3, index=len(prod3)-1)

                if cat3 == 'Toda la Categoría':
                    df_filter1 = datos_clean_or.loc[datos_clean_or['productcat1'] == cat1]
                    df_filter = df_filter1.loc[df_filter1['productcat2'] == cat2]
                else:
                    df_filter0 = datos_clean_or.loc[datos_clean_or['productcat1'] == cat1]
                    df_filter1 = df_filter0.loc[df_filter0['productcat2'] == cat2]
                    df_filter = df_filter1.loc[df_filter1['productcat3'] == cat3]

        #Sumatoria de todas las compras por marca
        
        plot_df1 = df_filter.groupby('productmarca')['qty_ordered'].sum().rename_axis('Marca').reset_index(name='Ventas')

        top = plot_df1.sort_values(by='Ventas', ascending=False)['Marca'] #Selección de las más vendidas
        plot_df1['Ventas'] = round((plot_df1['Ventas'] / (plot_df1['Ventas'].sum())) * 100, 2)
        st.expander(label='Campo a consultar')

        marcas1 = datos_clean_or['productmarca'].astype(str).unique()

        marcas = np.delete(marcas1, np.where(marcas1 == 'nan'))

        # Texto de la búsqueda
        
        if cat1 == 'Todas las Categorías':
            cat_txt = " en todas las categorías"
        elif cat1 != 'Todas las Categorías' and cat2 == 'Toda la Categoría':
            cat_txt = " en la categoría " + cat1
        elif cat1 != 'Todas las Categorías' and cat2 != 'Toda la Categoría' and cat3 == 'Toda la Categoría':
            cat_txt = " en " + cat1 + " - " + cat2 
        else:
            cat_txt = " en " + cat1 + " - " + cat2 + " - " + cat3
        
        seleccion = st.multiselect(
            "Selección de marcas:", options=marcas, default=top[:20], format_func=pretty
        )

        plot_df = plot_df1[plot_df1.Marca.isin(seleccion)]

        
        #Representación
        
        chart = (
            alt.Chart(
                plot_df,
                title="Las marcas más vendidas" + cat_txt ,
            )
                .mark_bar()
                .encode(
                x=alt.X("Ventas", title="% Ventas"),
                y=alt.Y(
                    "Marca",
                    sort=alt.EncodingSortField(field="Ventas", order="descending"),
                    title="",
                ),
                color=alt.Color(
                    "Marca",
                    legend=None,
                    scale=alt.Scale(scheme="category10"),
                ),
                tooltip=["Ventas", "Marca"],
            )
        )

        st.altair_chart(chart, use_container_width=True)
        
        
        
        #Ganancias de las marcas mas vendidas

        marca_gain = df_filter.groupby('productmarca')['Precio_calculado'].sum().rename_axis('Marca').reset_index(
            name='Ventas €')

        plot_df = marca_gain[marca_gain.Marca.isin(seleccion)].sort_values(by="Ventas €", ascending=False)

        plot_df['Ventas €'] = round(plot_df['Ventas €'], 0)
        chart = (
        alt.Chart(
            plot_df,
            title="Balance de las ventas de las marcas más vendidas" + cat_txt,
        )
            .mark_bar()
            .encode(
            x= alt.Y( "Marca",
                sort=alt.EncodingSortField(field="Ventas €", order="descending"),
                title="",
            ),
            y=alt.X("Ventas €", title="Balance(€)"),

            color=alt.Color(
                "Marca",
                legend=None,
                scale=alt.Scale(scheme="category10"),
            ),
            tooltip=["Ventas €", "Marca"],
        )
        )

        yrule = (
            alt.Chart().mark_rule(color="red", strokeWidth=2).encode(y=alt.datum(0))
        )

        st.altair_chart(chart + yrule, use_container_width=True)
        
        #Marcas con menos ganancias y/o pérdidas en dicha categoría
        
        plot_df = marca_gain.sort_values(by="Ventas €", ascending=True)
        plot_df['Ventas €'] = round(plot_df['Ventas €'], 0)
        plot_df = plot_df[0:20]
        chart = (
            alt.Chart(
                plot_df,
                title="Marcas con menos beneficio generado en " + cat_txt,
            )
                .mark_bar()
                .encode(
                x=alt.Y("Marca",
                        sort=alt.EncodingSortField(field="Ventas €", order="ascending"),
                        title="",
                        ),
                y=alt.X("Ventas €", title="Balance(€)"),

                color=alt.Color(
                    "Marca",
                    legend=None,
                    scale=alt.Scale(scheme="category10"),
                ),
                tooltip=["Ventas €", "Marca"],
            )
        )

        st.altair_chart(chart, use_container_width=True)
        
        #Ganancias/Pérdidas por categoría seleccionada en función del descuento
        
        
        if cat1 == 'Todas las Categorías':
            des1 = descuentos
        else:
            des1 = descuentos.loc[descuentos.loc[:, 'productcat1'] == cat1]
            if cat2 != 'Toda la Categoría':
                des1 = des1.loc[des1.loc[:, 'productcat2'] == cat2]
                if cat3 != 'Toda la Categoría':
                    des1 = des1.loc[des1.loc[:, 'productcat3'] == cat3]

        etiquetas = des1['descuentolabel'].unique()
        print(etiquetas)
        des1 = des1.groupby('descuentolabel')['Precio_calculado'].sum().rename_axis('Descuento').reset_index(
            name='Balance')
        
        des1['Balance'] = round(des1['Balance'], 0)
        
        chart =  (alt.Chart(
                des1,
                title="Balance de las ventas según el descuento ofertado"  + cat_txt,
            )
                .mark_bar()
                .encode(
                x=alt.Y("Descuento",
                        sort = etiquetas),
                y=alt.X("Balance", title="Balance"), 

                color=alt.condition(
                    alt.datum.Balance > 0,
                    alt.value("steelblue"),  # The positive color
                    alt.value("red")  # The negative color
                    ),
                tooltip=["Balance", "Descuento"],
            )
            )
        st.altair_chart(chart, use_container_width=True)
        
            
    else:
    
    #Página sobre Asociaciones
    
        get_data_clean.clear()
        st.sidebar.write("""
            ## Nivel de cliente
           :cloud: En esta sección puedes ver las palabras más presentes en las descripciones de los productos que más han comprado los clientes. \n
           :rocket: Más abajo se muestran los tipos de productos que se compran conjuntamente con mayor probabilidad. \n 
           :mag: Puedes hacer zoom para navegar por la red de manera más detallada y cambiar el rango de confianza como desees.
           """)
        st.title(":heavy_heart_exclamation_mark_ornament: Nivel de cliente")
        
           ## NUBE DE PALABRAS
        
        st.markdown(':thought_balloon: **Nube de palabras**')
       
        st.write("Selecciona una categoría y comprueba las palabras más presentes en los productos favoritos de los clientes.")
        cloud = st.selectbox("Categoría:", ['Cosmética y Belleza', 'Higiene y cuidado personal', 'Infantil', 'Nutrición', 'Salud', 'Veterinaria'])
        
        if cloud == 'Cosmética y Belleza':
            HtmlCloud = open("Data/Nube Cosmetica y Belleza.html", 'r', encoding='utf-8')          
        elif cloud == 'Higiene y cuidado personal':
            HtmlCloud = open("Data/Nube Higiene y cuidado personal.html", 'r', encoding='utf-8')
        elif cloud == 'Infantil':
            HtmlCloud = open("Data/Nube Infantil.html", 'r', encoding='utf-8')
        elif cloud == 'Nutrición':
            HtmlCloud = open("Data/Nube Nutricion.html", 'r', encoding='utf-8')     
        elif cloud == 'Salud':
            HtmlCloud = open("Data/Nube Salud.html", 'r', encoding='utf-8')  
        else:
            HtmlCloud = open("Data/Nube Veterinaria.html", 'r', encoding='utf-8')
        
        cloud_source = HtmlCloud.read()
        components.html(cloud_source, height=480, width=1050)   

        # Llamada a la creación del html con la Network y representación
        st.write("")
        st.markdown(':bulb: **¿Cómo se relacionan los productos?**')
        
        values = st.slider(
            'Selecciona el rango de confianza entre productos comprados a la vez para visualizar la red:',
            10.0, 60.0, (10.0, 30.0))

        components.html(rules(df_rules(values[0], values[1])), height=480, width=1050)

        st.write(df_rules(values[0], values[1]).style.format(({"Con una confianza del (%)": "{:.2f}"})))
        
        
     
        
def run_shell():
    st.write("Cargando...")


#Main ejecución 

if __name__ == '__main__':

    if not os.path.exists('Output'):
        os.makedirs('Output')
    if st._is_running_with_streamlit:
        url_params = st.experimental_get_query_params()
        if 'loaded' not in st.session_state:
            if len(url_params.keys()) == 0:
                st.experimental_set_query_params(page='Nivel geográfico y temporal')
                url_params = st.experimental_get_query_params()

            st.session_state.page = PAGES.index(url_params['page'][0])
            st.session_state['data_type'] = 'County Level'
            st.session_state['data_format'] = 'Raw Values'
            st.session_state['loaded'] = False

        run_UI()
    else:
        run_shell()
