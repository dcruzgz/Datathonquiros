#######PREPROCESSING#######

####tickets####

#cargamos el dataset
tickets_df <- read.csv("items_ordered_2yearsClean.csv")

#product_id=foreign de products

names(tickets_df)#nombres de las variables
which(is.na(tickets_df$base_cost))#hay datos nulos en precios base
which(is.na(tickets_df$price))#no hay datos nulos en precios
which(is.na(tickets_df$zipcode))#no hay datos nulos en códigos postales
which(is.na(tickets_df$qty_ordered))#no hay datos nulos en cantidades
which(is.na(tickets_df$discount_percent))#no hay datos nulos en descuentos

sort(unique(tickets_df$city))#misma ciudad escrita de muchas formas distintas, usar CP


#duplicados según item_id:
1:nrow(tickets_df)%in%which(duplicated(tickets_df$item_id))==F
#dataframe sin elementos duplicados:
tickets_df2=tickets_df[1:nrow(tickets_df)%in%which(duplicated(tickets_df$item_id))==F,]

#borramos el df que no vamos a usar más
rm(tickets_df)
gc()


#separación fecha
library(stringr)

vectorfechas=strtrim(paste(tickets_df2$created_at),10)

#incluimos la fecha separando year, month, y day
tickets_df2[,"Day"]=str_split_fixed(vectorfechas,"/",3)[,1]
tickets_df2[,"Month"]=str_split_fixed(vectorfechas,"/",3)[,2]
tickets_df2[,"Year"]=str_split_fixed(vectorfechas,"/",3)[,3]



#números erroneos CP
length(which(nchar(tickets_df2$zipcode)>5))
which(nchar(tickets_df2$zipcode)>5)
length(which(nchar(tickets_df2$zipcode)==5))
length(which(nchar(tickets_df2$zipcode)!=5))

#añadir provincia
  
#obtenemos solo los que tienen 4 o 5 dígitos
zipbienfilas=c(which(nchar(tickets_df2$zipcode)==5),
               which(nchar(tickets_df2$zipcode)==4))
tickets_df2zipbien=tickets_df2[zipbienfilas,]
nrow(tickets_df2zipbien)

#solo números en el zip
which(!grepl("\\D", tickets_df2$zipcode))


tickets_df2zipbien=tickets_df2zipbien[which(!grepl("\\D", tickets_df2zipbien$zipcode)),]

#introducimos el 0 que se ha borrado en los de 4 dígitos
zipcodes5=ifelse(nchar(tickets_df2zipbien$zipcode)==4,paste("0",tickets_df2zipbien$zipcode,sep=""),tickets_df2zipbien$zipcode)


tickets_df2zipbien$zipcode=zipcodes5

#eliminamos zipcodes no españoles
borrarzips=c(which(as.numeric(zipcodes5)<00999),
             which(as.numeric(zipcodes5)>52999))

allrows=1:nrow(tickets_df2zipbien)
validrows=allrows[allrows%in%borrarzips==F]



tickets_df2zipbien=tickets_df2zipbien[validrows,]

tickets_df2zipbien$zp_sim=substring(tickets_df2zipbien$zipcode,1,2)

#nos quedamos solo con los dos primeros dígitos, ya que son los necesarios para conocer
#la provincia
tickets_df2[rownames(tickets_df2zipbien),"zp_sim"]=tickets_df2zipbien$zp_sim

#calcular precio tras descuento
tickets_df2$precioreal=tickets_df2$price*(1-(tickets_df2$discount_percent/100))

which(is.na(tickets_df2$base_cost))#datos nulos en precio base

tickets_df2=tickets_df2[(1:nrow(tickets_df2))%in%which(is.na(tickets_df2$base_cost))==F,]
tickets_df2=tickets_df2[(1:nrow(tickets_df2))%in%which((tickets_df2$base_cost<0))==F,]

#algunos productos tienen precio base muy alto (>500). Probablemente sea un error en aquellos
#que se venden por mucho menos
tickets_df2=tickets_df2[(tickets_df2$base_cost<500 & tickets_df2$base_cost!=202.0000),]


#calculamos el beneficio resultante de cada transacción, restándole al precio de venta
#el precio del coste
tickets_df2$Precio_calculado=(tickets_df2$qty_ordered*tickets_df2$precioreal)-
  (tickets_df2$qty_ordered*tickets_df2$base_cost)

#descuentos por rangos

tickets_df2$descuento=0
tickets_df2$descuento=ifelse(tickets_df2$discount_percent<6,1,tickets_df2$descuento)
tickets_df2$descuento=ifelse((tickets_df2$discount_percent>5 &
                                  tickets_df2$discount_percent<11),
                               2,tickets_df2$descuento)
tickets_df2$descuento=ifelse((tickets_df2$discount_percent>10 &
                                  tickets_df2$discount_percent<16),
                               3,tickets_df2$descuento)
tickets_df2$descuento=ifelse((tickets_df2$discount_percent>15 &
                                  tickets_df2$discount_percent<21),
                               4,tickets_df2$descuento)
tickets_df2$descuento=ifelse((tickets_df2$discount_percent>20 &
                                  tickets_df2$discount_percent<26),
                               5,tickets_df2$descuento)
tickets_df2$descuento=ifelse((tickets_df2$discount_percent>25 &
                                  tickets_df2$discount_percent<31),
                               6,tickets_df2$descuento)
tickets_df2$descuento=ifelse((tickets_df2$discount_percent>30 &
                                  tickets_df2$discount_percent<36),
                               7,tickets_df2$descuento)
tickets_df2$descuento=ifelse((tickets_df2$discount_percent>35 &
                                  tickets_df2$discount_percent<41),
                               8,tickets_df2$descuento)
tickets_df2$descuento=ifelse((tickets_df2$discount_percent>40 &#aqui no hay
                                  tickets_df2$discount_percent<46),
                               9,tickets_df2$descuento)
tickets_df2$descuento=ifelse((tickets_df2$discount_percent>45 &
                                  tickets_df2$discount_percent<51),
                               9,tickets_df2$descuento)
tickets_df2$descuento=ifelse((tickets_df2$discount_percent>50 &
                                  tickets_df2$discount_percent<56),
                               10,tickets_df2$descuento)
tickets_df2$descuento=ifelse((tickets_df2$discount_percent>55 &
                                  tickets_df2$discount_percent<61),
                               11,tickets_df2$descuento)
tickets_df2$descuento=ifelse((tickets_df2$discount_percent>60 &
                                  tickets_df2$discount_percent<66),
                               12,tickets_df2$descuento)
tickets_df2$descuento=ifelse((tickets_df2$discount_percent>65 &
                                  tickets_df2$discount_percent<71),
                               13,tickets_df2$descuento)
tickets_df2$descuento=ifelse((tickets_df2$discount_percent>70),#este es 100%
                               14,tickets_df2$descuento)

tickets_df2$descuentolabel=factor(tickets_df2$descuento,labels=c("1-5%",
      "6-10%","11-15%","16-20%","21-25%","26-30%","31-35%","36-40%",
      "46-50%","51-55%","56-60%","61-65%","66-70%","100%"
                                                            ))

levels(tickets_df2$descuentolabel)


#####products####


productos_df <- read.csv("products.csv")

#sku=foreign de categorias.
which(is.na(productos_df$product_id))
which(is.na(productos_df$sku))


sort(unique(productos_df$analytic_category))#mejor utilizamos las del otro dataset
sort(unique(productos_df$marca_value))#algunas marcas tienen mal la codificación

#AposÃ¡n=aposán
#DrBrownÂ´s=drbrown's
#BegobaÃ±o=begobaño
#ArmonÃ­a=armonía
#NutribÃ©n=nutribén
#BebÃ©Due=bebédue
#BiManÃ¡n=bimanán
#FortÃ© Pharma=forté pharma
#Ãªtre belle=être belle
#NÃ»by=nûby
#AbeÃ±ula=abeñula
#M2 BeautÃ©=m2 beauté
#Gine-canestÃ©n=gine-canestén
#NeositrÃ­n=neositrín
#BabyBjÃ¶rn=babybjörn
#Acqua Colonia NÂº4711= acqua colonia Nº4711
#MÃ¡dara=mádara
#DermoestÃ©tica del Sur=dermoestética del sur
#PÃ©rez GimÃ©nez=pérez giménez
#DietÃ©ticos Intersa=dietéticos intersa
#BabybaÃ±o=babybaño
#Laboratorios RubiÃ³=laboratorios rubió
#Laboratorios ViÃ±as=laboratorios viñas

productos_df$marca_value=ifelse(productos_df$marca_value=="AposÃ¡n","Aposán",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="DrBrownÂ´s","DrBrown's",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="BegobaÃ±o","Begobaño",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="ArmonÃ­a","Armonía",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="NutribÃ©n","Nutribén",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="BebÃ©Due","Bebédue",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="BiManÃ¡n","Bimanán",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="FortÃ© Pharma","Forté pharma",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="Ãªtre belle","Ëtre belle",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="NÃ»by","Nûby",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="AbeÃ±ula","Abeñula",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="M2 BeautÃ©","M2 Beauté",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="Gine-canestÃ©n","Gine-Canestén",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="NeositrÃ­n","Neositrín",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="BabyBjÃ¶rn","Babybjörn",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="Acqua Colonia NÂº4711","Acqua colina Nº4711",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="MÃ¡dara","Mádara",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="DermoestÃ©tica del Sur","Dermoestética del Sur",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="PÃ©rez GimÃ©nez","Pérez Giménez",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="DietÃ©ticos Intersa","Dietéticos Intersa",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="BabybaÃ±o","Babybaño",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="Laboratorios RubiÃ³","Laboratorios Rubió",productos_df$marca_value)
productos_df$marca_value=ifelse(productos_df$marca_value=="Laboratorios ViÃ±as","Laboratorios Viñas",productos_df$marca_value)

sort(unique(productos_df$marca_value))#ahora ya están bien


sort(unique(productos_df$name))#
sort(unique(productos_df$short_description))#



sort(unique(productos_df$marca_value))#
sort(unique(productos_df$name))#


#####categories####

categorias_df <- read.csv("products_categories.csv")

which(is.na(categorias_df$cat1))
which(categorias_df$cat1=="")#hay categorias en blanco y algunas aparecen dentro 
#de la variable SKU, hay que separarlas

categorias_dfblanco <- read.csv("products_categories_sincomillas.csv")#sin comillas problemáticas
categorias_dfblanco=categorias_dfblanco[which(categorias_dfblanco$cat1==""),]
#write.csv(categorias_dfblanco,"categorias_dfblanco.csv")
categorias_dfnoblanco=categorias_df[which(categorias_df$cat1!=""),]
categorias_dfblanco=read.csv("categorias_dfblanco.csv")
categorias_dfblanco=categorias_dfblanco[,1:4]

categorias_df=categorias_dfnoblanco
categorias_df[10742:11159,]=categorias_dfblanco#los numeros son las filas

sort(unique(categorias_df$cat1))#bien menos el valor 0630
filamalcat=which(categorias_df$cat1=='0630')
categorias_df[filamalcat,]=c("800630","Infantil","Lactancia","Accesorios de extracción")
sort(unique(categorias_df$cat2))#bien
sort(unique(categorias_df$cat3))#bien

#arreglar minúsculas y mayúsculas

#categorias bien

unique(categorias_df$cat2)

firstup <- function(x) {#solo primera mayusula
  y=tolower(x)
  substr(y, 1, 1) <- toupper(substr(y, 1, 1))
  y
}

firstup(unique(categorias_df$cat1))

duplicadoscat1=(unique(categorias_df$cat1))[duplicated(firstup(unique(categorias_df$cat1)),fromLast = T)|duplicated(firstup(unique(categorias_df$cat1)))]
categorias_df$cat1=ifelse(categorias_df$cat1%in%duplicadoscat1,firstup(categorias_df$cat1),categorias_df$cat1)

duplicadoscat2=(unique(categorias_df$cat2))[duplicated(firstup(unique(categorias_df$cat2)),fromLast = T)|duplicated(firstup(unique(categorias_df$cat2)))]
categorias_df$cat2=ifelse(categorias_df$cat2%in%duplicadoscat2,firstup(categorias_df$cat2),categorias_df$cat2)

#algunas categorias empiezan con espacios en blanco

categorias_df$cat3=trimws(categorias_df$cat3)
duplicadoscat3=(unique(categorias_df$cat3))[duplicated(firstup(unique(categorias_df$cat3)),fromLast = T)|duplicated(firstup(unique(categorias_df$cat3)))]
categorias_df$cat3=ifelse(categorias_df$cat3%in%duplicadoscat3,firstup(categorias_df$cat3),categorias_df$cat3)


sort(unique(categorias_df$cat1))#
sort(unique(categorias_df$cat2))#
sort(unique(categorias_df$cat3))#

categorias_df$cat3=firstup(categorias_df$cat3)#algunas empezaban por minuscula


categorias_df$cat4=ifelse(categorias_df$cat3=="",categorias_df$cat2,categorias_df$cat3)
categorias_df$cat4=ifelse(categorias_df$cat2=="",categorias_df$cat1,categorias_df$cat4)



sort(unique(categorias_df$cat4))



#####todo en el mismo df####
#tarda bastante, como ya está generado se puede cargar para los siguientes pasos el
#archivo "tickets_dfbigenglishcleandefinitivo.csv"

tickets_dfbig=tickets_df2

for (x in unique(tickets_dfbig$product_id)) {
  useddata=subset(tickets_dfbig,product_id==x)
  useddata1=subset(productos_df,product_id==x)
  if (length(useddata1$sku>0)){
    tickets_dfbig[rownames(useddata),"productname"]=useddata1$name[1]
    tickets_dfbig[rownames(useddata),"productmarca"]=useddata1$marca_value[1]
    tickets_dfbig[rownames(useddata),"productdescr"]=useddata1$short_description[1]
    }
  useddata2=subset(categorias_df,sku==useddata1$sku[1])
  if (length(useddata2$sku>0)){
    tickets_dfbig[rownames(useddata),"productcat1"]=useddata2$cat1
    tickets_dfbig[rownames(useddata),"productcat2"]=useddata2$cat2
    tickets_dfbig[rownames(useddata),"productcat3"]=useddata2$cat3
    tickets_dfbig[rownames(useddata),"productcat4"]=useddata2$cat4
  }
  if (which(unique(tickets_dfbig$product_id)==x)%%10==0) {
    #para saber el progreso
    print(length(unique(tickets_dfbig$product_id)))
    print(which(unique(tickets_dfbig$product_id)==x))
  }
}

#write.csv(tickets_dfbig,"tickets_dfbigenglishcleandefinitivo.csv")



#######ANALISIS#######

#cada uno de estos apartados se puede iniciar de manera independiente cargando
#el dataframe que contiene toda la información al inicio.


#####dataframe reducido#####

dfbigenglish=read.csv("tickets_dfbigenglishcleandefinitivo.csv")

#guardamos el dataframe con la información necesaria para la plataforma streamlit
#write.csv(dfbigenglish[,c(6,14:16,18,20,22:24)],"tickets_data.csv")


####Compras conjuntas####

dfbigenglish=read.csv("tickets_dfbigenglishcleandefinitivo.csv")

#solo filas sin categorías en blanco
dfbigenglish=dfbigenglish[1:nrow(dfbigenglish)%in%which(is.na(dfbigenglish$productcat4)==F),]

#convertimos las variables en factores
dfbigenglish$product_id <- as.factor(dfbigenglish$product_id)
dfbigenglish$customer_id <- as.factor(dfbigenglish$customer_id)
dfbigenglish$productmarca <- as.factor(dfbigenglish$productmarca)
dfbigenglish$productcat1 <- as.factor(dfbigenglish$productcat1)
dfbigenglish$productcat2 <- as.factor(dfbigenglish$productcat2)
dfbigenglish$productcat3 <- as.factor(dfbigenglish$productcat3)
dfbigenglish$productcat4 <- as.factor(dfbigenglish$productcat4)


#cargamos los paquetes necesarios
library(dplyr)
library(arules)
dat_t <- dfbigenglish %>% select(customer_id, productcat4) %>% unique()
dat_t <- as(split(dfbigenglish$productcat4, dfbigenglish$customer_id), "transactions")
inspect(head(dat_t, 3))

#observamos las categorías más consumidas
itemFrequencyPlot(dat_t,topN=50,type="absolute")

#obtenemos las relaciones más probables entre categorías
fm_rules <-  apriori(dat_t, parameter = list(support = 0.01, confidence = 0.10,maxtime=30))

rules=inspect(head(sort(fm_rules, by = "confidence"), 1000))

#write.csv(rules,"rules.csv)


######sumatorio descuento agregado#####

dfdisccategories=(aggregate(Precio_calculado ~ productcat1+productcat2+productcat3+descuentolabel+descuento, data = dfbigenglish, FUN = sum, na.rm = TRUE))
#generamos el archivo para calcular los beneficios de cada categoria según el descuento ofertado
#write.csv(dfdisccategories,"Ganancias y pérdidas descuentos y categorías.csv")

#

#####nube de palabras####
#cargamos el documento completo
doctextobig <- read.csv("tickets_dfbigenglishcleandefinitivo.csv")

#seleccionamos solo las columnas relevantes
doctextobig=doctextobig[,c("productcat1","productdescr","qty_ordered")]
#y aquellas descripciones y categorías que no estén vacías
doctextobig=subset(doctextobig,is.na(productdescr)==F)
doctextobig=subset(doctextobig,is.na(productcat1)==F)
doctextobig=subset(doctextobig,productdescr!="")

#cagamos los paquetes necesarios
library(tidyr)
library(tm)
library(SnowballC)
library(wordcloud2)
library(RColorBrewer)
library(RCurl)
library(XML)
library(rsconnect)
library(htmltools)

#hacemos un bucle con tantos niveles como categorías de nivel 1
for (y in 1:6) {
  #obtenemos un dataframe temporal con las cantidades de cada producto dependiendo
  #de la categoría 1, así como su descripción
  doctexto=(aggregate(qty_ordered ~ productcat1+productdescr, data = doctextobig, FUN = sum, na.rm = TRUE))
  
  #creamos un nuevo dataframe temporal y un índice para las filas en las que añadiremos
  #datos
  dfnuevo=data.frame(0)
  indexrow=0
  
  #creamos el dataframe temporal con la categoría objetivo
  doctexto=subset(doctexto,productcat1==unique(doctextobig$productcat1)[y])

  #localizamos el percentil 90 de las cantidades vendidas (el 10% más vendidas)
  perc90=quantile(doctexto$qty_ordered,.9)
  
  #seleccionamos solo el 10% de productos más vendidos
  doctexto=subset(doctexto,qty_ordered>perc90)
  
  
  #hacemos un bucle que complete el nuevo dataframe con la información repetida para cada
  #descripción tantas veces como cada producto ha sido comprado
  for (x in 1:nrow(doctexto)) {
    indextarget=doctexto[x,"qty_ordered"]
    dfnuevo[(indexrow+1):(indexrow+1+indextarget-1),1]=doctexto[x,"productdescr"]
    dfnuevo[(indexrow+1):(indexrow+1+indextarget-1),2]=doctexto[x,"qty_ordered"]
    dfnuevo[(indexrow+1):(indexrow+1+indextarget-1),3]=doctexto[x,"productcat1"]
    indexrow=(indexrow+1+indextarget)
  }
  
  
  #codificamos correctamente los acentos que faltan y eliminamos caracteres no necesarios
  dfnuevo$X0=gsub("Ã¡","á",dfnuevo$X0)
  dfnuevo$X0=gsub("Ã©","é",dfnuevo$X0)
  dfnuevo$X0=gsub("Ã­","í",dfnuevo$X0)
  dfnuevo$X0=gsub("Ã³","ó",dfnuevo$X0)
  dfnuevo$X0=gsub("Ãº","ú",dfnuevo$X0)
  dfnuevo$X0=gsub("Ã±","ñ",dfnuevo$X0)
  dfnuevo$X0=gsub("<p>","",dfnuevo$X0)
  dfnuevo$X0=gsub("<strong>","",dfnuevo$X0)
  dfnuevo$X0=gsub("</strong>","",dfnuevo$X0)
  dfnuevo$X0=gsub("</em>","",dfnuevo$X0)
  dfnuevo$X0=gsub("</p>","",dfnuevo$X0)
  
  #transformamos los datos al formato correspondiente
  docs <- iconv(dfnuevo[,1], to = "UTF-8")
  
  docs <- Corpus(VectorSource(docs))
  
  
  #eliminamos caracteres no necesarios o que pueden generar confusión en la nube
  docs <- docs %>%
    tm_map(removeNumbers) %>%
    tm_map(removePunctuation) %>%
    tm_map(stripWhitespace)
  docs <- tm_map(docs, content_transformer(tolower))
  docs <- tm_map(docs, removeWords, stopwords("spanish"))
  
  #obtenemos el dataframe con el formato requerido por la función wordcloud2
  dtm <- TermDocumentMatrix(docs) 
  matrix <- as.matrix(dtm) 
  words <- sort(rowSums(matrix),decreasing=TRUE)
  df <- data.frame(word = names(words),freq=words)
  
  #creamos la nube de palabras en html
  nubepalabras=(wordcloud2(df,shape="circle"))
  #la mostramos en el visualizador para guardarla desde ahí
  html_print(nubepalabras)
  
}

