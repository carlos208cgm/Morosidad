# Script de preparación de datos

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import os

def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw', filename), sep = ';')
    print(filename, ' cargado correctamente')
    return df

def data_preparation(df):
  df.rename(index=str,
          columns={"NRO_VEC_COB":"VECES_EN_COBRANZAS",
                   "ESTADO_PDP":"ESTADO_PROMESAS",
                   "NRO_CUOTAS":"CUOTAS_ADEUDADAS",
                   "MES_0":"DEUDAVENCIDA_ACTUAL",
                   "MES_1":"DEUDAVENCIDA_1MES",
                   "MES_2":"DEUDAVENCIDA_2MESES",
                   "ACTIVACION":"AÑOACTIVACION"
                   },inplace=True)
  df[df.duplicated(subset=None)]
  df.head()
  df.FECHALLAMADA = pd.to_datetime(df.FECHALLAMADA, format="%d/%m/%Y")
  df_cat = df.select_dtypes(include=["object","datetime"])
  df_num = df.select_dtypes("number")
  imputer_cat = SimpleImputer(strategy="most_frequent")
  imputer_cat.fit(df_cat)
  df_cat_imputed = pd.DataFrame(imputer_cat.transform(df_cat), columns = df_cat.columns)
  knn = KNNImputer(n_neighbors=5)
  df_num_knn = pd.DataFrame(data=knn.fit_transform(df_num),columns=df_num.columns)
  df2=pd.concat([df_num_knn,df_cat_imputed], axis=1)
  df2["TIPOCONTACTO"] = df2["TIPOCONTACTO"].map({"CNE":0,"CEF":1})
  df2["PERIODO_DIA"]=pd.cut(df2["HORA"], bins=[6,12,18,24], labels=["mañana","tarde","noche"])
  df2=df2.drop(["HORA"],axis=1)
  df2["DIA_SEMANA"]=df2["FECHALLAMADA"].dt.day_name()
  df2=df2.drop(["FECHALLAMADA"],axis=1)
  df2["ANTIGUEDAD"]=2015-df2["AÑOACTIVACION"]
  df2=df2.drop(["AÑOACTIVACION"],axis=1)
  num_cols=['DEUDAVENCIDA_ACTUAL','DEUDAVENCIDA_1MES', 'DEUDAVENCIDA_2MESES', 'DEUDA_TOTAL']
  df2_num = pd.DataFrame(data=df2[num_cols])
  df2_num[num_cols]=df2[num_cols].apply(lambda x: np.log(x+1))
  df2=df2.drop(num_cols,axis=1)
  #LABEL ENCODING
  df2["VECES_EN_COBRANZAS"] = df2["VECES_EN_COBRANZAS"].map({"<=10":0,">10":1})
  df2["CUOTAS_ADEUDADAS"] = df2["CUOTAS_ADEUDADAS"].map({"<=24":0,"<24, 48]":1,">48":2})
  #ONE HOT ENCODING
  cat_cols=['CUOTAS_ADEUDADAS',"DIA_SEMANA",'PERIODO_DIA']
  df2_cat = pd.get_dummies(data=df2[cat_cols], columns=cat_cols)
  df2=df2.drop(cat_cols,axis=1)
  df3 = pd.concat([df2,df2_num,df2_cat], axis=1)
  df3 = df3.astype({"ESTADO_PROMESAS":"int64","MORA":"int64","ANTIGUEDAD":"int64"})
  #ESCALAMIENTO
  escalar_cols=['ANTIGUEDAD','DEUDAVENCIDA_ACTUAL','DEUDAVENCIDA_1MES', 'DEUDAVENCIDA_2MESES', 'DEUDA_TOTAL']
  scaler=MinMaxScaler()
  df3_escalar=pd.DataFrame(data=scaler.fit_transform(df3[escalar_cols]), columns=escalar_cols)
  df3=df3.drop(escalar_cols,axis=1)
  df3=pd.concat([df3,df3_escalar], axis=1)
  cols_to_convert = [
    'CUOTAS_ADEUDADAS_0',
    'CUOTAS_ADEUDADAS_1',
    'CUOTAS_ADEUDADAS_2',
    'DIA_SEMANA_Friday',
    'DIA_SEMANA_Monday',
    'DIA_SEMANA_Thursday',
    'DIA_SEMANA_Tuesday',
    'DIA_SEMANA_Wednesday',
    'PERIODO_DIA_mañana',
    'PERIODO_DIA_tarde',
    'PERIODO_DIA_noche'
  ]
  for col in cols_to_convert:
    df3[col] = df3[col].astype(int)
  return df3

# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename), sep = ';')
    print(filename, 'exportado correctamente en la carpeta processed')

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('Morosidad_raw_train.csv')
    df1.head()
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['ESTADO_PROMESAS',	'MORA',	'VECES_EN_COBRANZAS',	'TIPOCONTACTO',	'CUOTAS_ADEUDADAS_0',
                          'CUOTAS_ADEUDADAS_1',	'CUOTAS_ADEUDADAS_2',	'DIA_SEMANA_Friday',	'DIA_SEMANA_Monday',	'DIA_SEMANA_Thursday',
                         'DIA_SEMANA_Tuesday',	'DIA_SEMANA_Wednesday',	'PERIODO_DIA_mañana',	'PERIODO_DIA_tarde',	'PERIODO_DIA_noche',
                          'ANTIGUEDAD',	'DEUDAVENCIDA_ACTUAL',	'DEUDAVENCIDA_1MES',	'DEUDAVENCIDA_2MESES',	'DEUDA_TOTAL'],'Morosidad_train.csv')
    df2 = read_file_csv('Morosidad_raw_valid.csv')
    tdf2 = data_preparation(df1)
    data_exporting(tdf2, ['ESTADO_PROMESAS',	'MORA',	'VECES_EN_COBRANZAS',	'TIPOCONTACTO',	'CUOTAS_ADEUDADAS_0',
                         'CUOTAS_ADEUDADAS_1',	'CUOTAS_ADEUDADAS_2',	'DIA_SEMANA_Friday',	'DIA_SEMANA_Monday',	'DIA_SEMANA_Thursday',
                         'DIA_SEMANA_Tuesday',	'DIA_SEMANA_Wednesday',	'PERIODO_DIA_mañana',	'PERIODO_DIA_tarde',	'PERIODO_DIA_noche',
                         'ANTIGUEDAD',	'DEUDAVENCIDA_ACTUAL',	'DEUDAVENCIDA_1MES',	'DEUDAVENCIDA_2MESES',	'DEUDA_TOTAL'],'Morosidad_valid.csv')
    df1 = read_file_csv('Morosidad_raw_score.csv')
    tdf3 = data_preparation(df1)
    data_exporting(tdf3, ['ESTADO_PROMESAS',	'MORA',	'VECES_EN_COBRANZAS',	'CUOTAS_ADEUDADAS_0',
                         'CUOTAS_ADEUDADAS_1',	'CUOTAS_ADEUDADAS_2',	'DIA_SEMANA_Friday',	'DIA_SEMANA_Monday',	'DIA_SEMANA_Thursday',
                         'DIA_SEMANA_Tuesday',	'DIA_SEMANA_Wednesday',	'PERIODO_DIA_mañana',	'PERIODO_DIA_tarde',	'PERIODO_DIA_noche',
                         'ANTIGUEDAD',	'DEUDAVENCIDA_ACTUAL',	'DEUDAVENCIDA_1MES',	'DEUDAVENCIDA_2MESES',	'DEUDA_TOTAL'],'Morosidad_score.csv')
    
if __name__ == "__main__":
    main()