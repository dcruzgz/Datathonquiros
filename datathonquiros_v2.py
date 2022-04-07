import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import json
import folium
import numpy as np
from pyvis.network import Network
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import altair as alt
from streamlit_folium import folium_static

PAGES = [
    'Nuestras ventas en el territorio',
    'TOP MARCAS',
    'Próximas promociones'
]

datos_clean_or = pd.read_csv('https://www.dropbox.com/s/mp1zbrj68ccz0oc/datos_f.csv?dl=1'
                         , header=0, encoding="ISO-8859-1")  # read a CSV file inside the 'data" folder next to 'app.py'

datos_clean_or['productcat1'] = datos_clean_or['productcat1'].fillna('Sin clasificar')
datos_clean_or['productcat2'] = datos_clean_or['productcat2'].fillna('Sin clasificar')
datos_clean_or['productcat3'] = datos_clean_or['productcat3'].fillna('Sin clasificar')
datos_clean_or = datos_clean_or[datos_clean_or['zp_sim'].notna()]

data_code = pd.read_excel(
    'codprov.xls')
data_geo = json.load(
    open('spain_provinces.geojson',
         encoding="utf8"))


#region MAPA
# ---------------------------------
#     MAPA
# ---------------------------------

dat_1 = data_code.sort_values('CODIGO')
dat_1['cod_prov'] = data_code['CODIGO'].astype(int).astype(str)
dat_1['cod_prov'] = dat_1['cod_prov']
data_all = dat_1.set_index('CODIGO')




dicts = {"Ganancia": 'GAIN'}

# showing the maps
map_sby = folium.Map(tiles='OpenStreetMap', location=[39.59130262109639, -3.933016292135545], zoom_start=6, width=700, height=570)
folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(map_sby)


def threshold(data):
    threshold_scale = np.linspace(data_all[dicts[data]].min(),
                                  data_all[dicts[data]].max(),
                                  10, dtype=float)
    # change the numpy array to a list
    threshold_scale = threshold_scale.tolist()
    threshold_scale[-1] = threshold_scale[-1]
    return threshold_scale


def show_maps(data, threshold_scale, nombre_valor):
    maps = folium.Choropleth(geo_data=data_geo,
                             data=data_all,
                             columns=['cod_prov',dicts[data]],
                             key_on='feature.properties.cod_prov',
                             threshold_scale=threshold_scale,
                             fill_color='YlOrRd',
                             fill_opacity=0.7,
                             line_opacity=0.2,
                            #legend_name=dicts[data],
                             highlight=True,
                             reset=True).add_to(map_sby)

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
    folium_static(map_sby)

# endregion


#region RULES

def df_rules():
    url = 'https://raw.githubusercontent.com/dcruzgz/Datahonquiros/09bafb9e6c376e6b6754fba40b55bb7925d8bd19/rules.csv?token=GHSAT0AAAAAABSZT435MD2WTR7XIW36ZK7GYSSRUOA'

    df_raw = pd.read_csv(url, encoding="ISO-8859-1")  # read a CSV file inside the 'data" folder next to 'app.py'

    df1 = df_raw[["lhs", "rhs", "confidence"]]
    df_rules = pd.DataFrame()

    df_rules["Si el cliente compró"] = df1.loc[:, "lhs"].str.replace(r'[{]', '', regex=True).replace(r'[}]', '',
                                                                                                     regex=True)
    df_rules["Comprará"] = df1.loc[:, "rhs"].str.replace(r'[{]', '', regex=True).replace(r'[}]', '', regex=True)
    df_rules["Con una confianza del (%)"] = round(pd.to_numeric(df1["confidence"]) * 100, 2)
    return df_rules


def rules(df_rules):
    # Relaciones gráficas

    r_net = Network(height='450px', width='100%', bgcolor='#FFFFFF.', font_color='black')

    # set the physics layout of the network

    sources = df_rules['Si el cliente compró']
    targets = df_rules['Comprará']
    weights = df_rules['Con una confianza del (%)']

    edge_data = zip(sources, targets, weights)

    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2]

        r_net.add_node(src, src, title=src)
        r_net.add_node(dst, dst, title=dst)
        r_net.add_edge(src, dst, value=w)

    neighbor_map = r_net.get_adj_list()

    # add neighbor data to node hover data
    for node in r_net.nodes:
        node['title'] += ' Relacionado:<br>' + '<br>'.join(neighbor_map[node['id']])
        node['value'] = len(neighbor_map[node['id']])

    r_net.save_graph('rules.html')

    HtmlFile = open("rules.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    return source_code


def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection
#endregion

#region MARCAS

def pretty(s: str) -> str:
    try:
        return dict(js="JavaScript")[s]
    except KeyError:
        return s.capitalize()
#endregion

def run_UI():
    st.set_page_config(
        page_title="Datathonquiros",
        page_icon=":brain:",
        initial_sidebar_state="expanded",
        menu_items={
            'Report a bug': "https://github.com/arup-group/social-data/issues/new/choose",
            'About': """            
        Antonio
        Daniel
        """
        }
    )
    st.sidebar.title('Datathonquiros - Análisis')

    if st.session_state.page:
        page = st.sidebar.radio('Navigation', PAGES, index=st.session_state.page)
    else:
        page = st.sidebar.radio('Navigation', PAGES, index=1)

    if page == 'Nuestras ventas en el territorio':
        st.sidebar.write("""
            ## About
            AQUI VA EL MAPA
            Seleccione en los filtros de búsqueda. 
            Si alguna provincia aparece en gris con valor NaN en esa provincia no hubo ventas en ese mes.
          """)
        st.title(":earth_africa: Las ventas por el territorio")
        my_expander = st.expander(label='Filtros de la búsqueda')
        with my_expander:
            cols = st.columns((1, 1, 1))

            #SELECCION DE FECHAS
            month = datos_clean_or['Month'].unique()
            month = np.append(month, ['Todo el año'])

            year_1 = datos_clean_or['Year'].unique()
            year_1 = np.append(year_1, ['Todos los años'])

            mes = cols[0].selectbox("Mes",
                              month)
            year = cols[1].selectbox("Año",
                              year_1)

            variable_map = cols[2].selectbox("Dato",
                                             ("Ganancia", "Codigo"))

            if mes == 'Todo el año' and year != 'Todos los años':
                datos_clean = datos_clean_or[datos_clean_or['Year'] == int(year)]
              

            elif mes != 'Todo el año' and year == 'Todos los años':
                datos_clean = datos_clean_or[datos_clean_or['Month'] == int(mes)]

            elif mes == 'Todo el año' and year == 'Todos los años':
                datos_clean = datos_clean_or
            else:
                datos_clean = datos_clean_or[(datos_clean_or['Month'] == int(mes)) & (datos_clean_or['Year'] == int(year))]


            ####

            prod1 = datos_clean['productcat1'].unique()
            prod1 = np.append(prod1, ['Toda la Categoría'])

            cat1 = cols[0].selectbox("Categoría:",
                                        prod1)
            if cat1 == 'Toda la Categoría':
                cols[1].selectbox("Subcategoría 1:",
                                  ['Toda la Categoría'])
                cols[2].selectbox("Subcategoría 3:",
                                  ['Toda la Categoría'])

                df_sum = datos_clean.groupby(['zp_sim'])['Precio_calculado', 'productcat1'].sum()/1000
                data_all['GAIN'] = df_sum['Precio_calculado']

            else:
                df_va1 = datos_clean.loc[datos_clean.loc[:, 'productcat1'] == cat1]
                prod2 = df_va1['productcat2'].unique()
                prod2 = np.append(prod2, ['Toda la Categoría'])

                cat2 = cols[1].selectbox("Categoría 2",
                                prod2)

                if cat2 == 'Toda la Categoría':
                    df_sum = df_va1.groupby(['zp_sim'])['Precio_calculado', 'productcat1'].sum()/1000
                    data_all['GAIN'] = df_sum['Precio_calculado']
                    cols[2].selectbox("Categoría 3",
                                      ['Toda la Categoría'])
                else:

                    df_sum = df_va1.groupby(['zp_sim'])['Precio_calculado', 'productcat2'].sum()/1000
                    data_all['GAIN'] = df_sum['Precio_calculado']


                    df_va2 = datos_clean.loc[datos_clean.loc[:, 'productcat2'] == cat2]
                    prod3 = df_va2['productcat3'].unique()
                    prod3 = np.append(prod3, ['Toda la Categoría'])
                    cat3 = cols[2].selectbox("Categoría 3",
                                             prod3)

                    if cat3 == 'Toda la Categoría':
                        df_sum = df_va2.groupby(['zp_sim'])['Precio_calculado', 'productcat2'].sum()
                        data_all['GAIN'] = df_sum['Precio_calculado']
                    else:
                        df_va3 = df_va2.loc[datos_clean.loc[:, 'productcat3'] == cat3]
                        df_sum = df_va3.groupby(['zp_sim'])['Precio_calculado', 'productcat3'].sum()/1000
                        data_all['GAIN'] = df_sum['Precio_calculado']
                        

            for idx in range(51):
                data_geo['features'][idx]['properties']['GAIN'] = round(data_all['GAIN'][idx + 1], 2)
                data_geo['features'][idx]['properties']['cod_prov'] = data_all['cod_prov'][idx + 1]  # igualar los codigos los 0 a la izq dan problemas        
                      
    ### LLAMAR A LA GESTION PARA CREAR GAIN A PARTIR DE VARIABLE MAP
        select_data = variable_map
        nombre_valor = "Balance (k€): "
        show_maps(select_data, threshold(select_data), nombre_valor)

    elif page == 'TOP MARCAS':
        st.sidebar.write("""
            ## About
          Bayes
          """)
        st.title(" TOP MARCAS :star:")
        cols = st.columns((1, 1, 1))

        prod1 = datos_clean_or['productcat1'].unique()
        prod1 = np.append(prod1, ['Toda la Categoría'])

        cat1 = cols[0].selectbox("Categoría:",
                                 prod1)
        if cat1 == 'Toda la Categoría':
            cols[1].selectbox("Subcategoría 1:",
                              ['Toda la Categoría'])
            cols[2].selectbox("Subcategoría 3:",
                              ['Toda la Categoría'])
            df_filter = datos_clean_or

        else:
            df_va1 = datos_clean_or.loc[datos_clean_or.loc[:, 'productcat1'] == cat1]
            prod2 = df_va1['productcat2'].unique()
            prod2 = np.append(prod2, ['Toda la Categoría'])

            cat2 = cols[1].selectbox("Categoría 2",
                                     prod2)

            if cat2 == 'Toda la Categoría':

                df_filter = datos_clean_or.loc[datos_clean_or['productcat1'] == cat1]

                cols[2].selectbox("Categoría 3",
                                  ['Toda la Categoría'])
            else:
                df_va2 = datos_clean_or.loc[datos_clean_or.loc[:, 'productcat2'] == cat2]
                prod3 = df_va2['productcat3'].unique()


                prod3 = np.append(prod3, ['Toda la Categoría'])
                cat3 = cols[2].selectbox("Categoría 3",
                                         prod3)

                if cat3 == 'Toda la Categoría':
                    df_filter1 = datos_clean_or.loc[datos_clean_or['productcat1'] == cat1]
                    df_filter = df_filter1.loc[df_filter1['productcat2'] == cat2]
                else:
                    df_filter0 = datos_clean_or.loc[datos_clean_or['productcat1'] == cat1]
                    df_filter1 = df_filter0.loc[df_filter0['productcat2'] == cat2]
                    df_filter = df_filter1.loc[df_filter1['productcat3'] == cat3]


        plot_df1 = df_filter.groupby('productmarca')['qty_ordered'].sum().rename_axis('Marca').reset_index(name='Ventas')
       # plot_df1 = df_filter['productmarca'].value_counts(dropna=True).rename_axis('Marca').reset_index(
        #name='Ventas')
        top = plot_df1['Marca']
        plot_df1['Ventas'] = round((plot_df1['Ventas'] / (plot_df1['Ventas'].sum())) * 100, 2)
        st.expander(label='Campo a consultar')

        marcas1 = datos_clean_or['productmarca'].astype(str).unique()

        marcas = np.delete(marcas1, np.where(marcas1 == 'nan'))



        langs = st.multiselect(
            "Selección de marcas:", options=marcas, default=top[:20], format_func=pretty
        )

        plot_df = plot_df1[plot_df1.Marca.isin(langs)]


        chart = (
            alt.Chart(
                plot_df,
                title="Las marcas más vendidas",
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
                    legend=alt.Legend(title="Marca"),
                    scale=alt.Scale(scheme="category10"),
                ),
                tooltip=["Ventas", "Marca"],
            )
        )

        st.altair_chart(chart, use_container_width=True)

    else:
        st.sidebar.write("""
            ## About
           ARULES
            Para mas información sobre los métodos usados:
            https://cran.r-project.org/web/packages/arules/arules.pdf.
           """)
        st.title(":rocket: Próximas promociones")

        components.html(rules(df_rules()), height=480, width=650)
        df_rules().style.format(({"Con una confianza del (%)": "{:.2f}"}))
        aggrid_interactive_table(df=df_rules())

def run_shell():
    st.write("Cargando...")


if __name__ == '__main__':

    if not os.path.exists('Output'):
        os.makedirs('Output')
    if st._is_running_with_streamlit:
        url_params = st.experimental_get_query_params()
        if 'loaded' not in st.session_state:
            if len(url_params.keys()) == 0:
                st.experimental_set_query_params(page='Nuestras ventas en el territorio')
                url_params = st.experimental_get_query_params()

            st.session_state.page = PAGES.index(url_params['page'][0])
            st.session_state['data_type'] = 'County Level'
            st.session_state['data_format'] = 'Raw Values'
            st.session_state['loaded'] = False

        run_UI()
    else:
        run_shell()
