import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt
import numpy as np
from urllib.error import URLError
import os
from pyvis.network import Network

st.set_page_config(layout="wide")

#---------------------------------
#     MAPA
#---------------------------------

#---------------------------------
#     ASOCIACIONES
#---------------------------------


df_raw = pd.read_csv("C:/Users/dcruzg/Desktop/Datathon/Atmira_Pharma_Visualization/dathon/rules.csv",  encoding = "ISO-8859-1")  # read a CSV file inside the 'data" folder next to 'app.py'
df1 = df_raw[["lhs", "rhs", "confidence"]]
df1["lhs"] = df1["lhs"].str.replace('{', '')
df1["lhs"] = df1["lhs"].str.replace('}', '')
df1["rhs"] = df1["rhs"].str.replace('{', '')
df1["rhs"] = df1["rhs"].str.replace('}', '')

df1["confidence"]= round(pd.to_numeric(df1["confidence"])*100, 2)

df1.columns = ["Si el cliente compró", "Comprará", "Con una confianza del (%)"]




    #Relaciones gráficas

r_net = Network(height='450px', width='100%', bgcolor='#f8f8f8', font_color='black')

# set the physics layout of the network

sources = df1['Si el cliente compró']
targets = df1['Comprará']
weights = df1['Con una confianza del (%)']

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



#Ordenar página

row1_1, row1_2 = st.columns((3, 2))

with row1_1:
    st.title("¿CUÁL ES LA SIGUIENTE PROMOCIÓN?")  # add a title


with row1_2:
    st.write(
        """
    ##
    Reglas obtenidas en el tratamiento de los datos sobre los clientes.
    Es el resultado de aplicar arules en R.
    """
    )

row2_1, row2_2 = st.columns((2, 2))

with row2_1:
    st.write('\n')
    st.dataframe(df1.style.format(({"Con una confianza del (%)": "{:.2f}"})), height=470, width=650) # visualize my dataframe in the Streamlit app

with row2_2:
    components.html(source_code, height=480, width=650)

st.write(
    """
      ##
      Para mas información sobre los métodos usados:
      https://cran.r-project.org/web/packages/arules/arules.pdf.
      """
)
#--------------------------------------
