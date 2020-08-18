def b():
  # Something
  return 1
# Gabriel Moyano
# Leonardo Rebola
# Pablo Leiva


# Importación de las librerías necesarias
import numpy as np
import pandas as pd
# Puede que nos sirvan también
import matplotlib as mpl
mpl.get_cachedir()
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from datetime import datetime
import os
import warnings
from zipfile import ZipFile

pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
pd.set_option('max_colwidth', 151)

#Parsing auxiliar
dateparse = lambda x: dt.datetime.strptime(x.strip(), '%Y-%m-%d %H:%M:%S')

# para que los numeros flotantes me los muestre solo con dos decimales
pd.options.display.float_format = '{:.2f}'.format

df_logs = pd.DataFrame(columns=['datetime', 'message'])

def log(message):
    global df_logs
    log_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print('{} {}'.format(log_time, message))
    df_logs = df_logs.append({'datetime':log_time, 'message':message}, ignore_index=True)

# Información del ultimo dataset.
# Esta información es utilizada para saber que no se esta importando un dataset con menor cantidad de registros.

ENERGIA_FILAS=44363
ENERGIA_COLUMNAS=9
ENERGIA_DIAS=154

CLIMA_FILAS=3001
CLIMA_COLUMNAS=3
CLIMA_DIAS=194

####################################################################################
def load_energia(url_or_path):
    
    global _ds_energia, energia_origin_col_names, energia_work_col_names, energia_origin_col_names_dic, interesting_col
    
    _ds_energia = pd.read_csv(url_or_path,
                              dtype={'Amper fase T-A': float},
                              parse_dates=['Fecha'],
                              date_parser=dateparse,
                              float_precision='round_trip')

    # nombres de columnas originales
    energia_origin_col_names = ['Fecha', 'Amper fase T-A', 'Amper fase S-A', 'Amper fase R-A', 'Vab', 'Vca', 'Vbc', 'Kwatts 3 fases', 'Factor de Poten-A']
    # nombre de columnas para trabajar el dataset de forma mas simple
    energia_work_col_names = ['fecha','ta','sa','ra','vab','vca','vbc','kwatts','potencia']
    # mapeo de nombre de columnas de trabajo y las originales. Pandas no tiene un "alias" para las variables.
    energia_origin_col_names_dic = {'fecha':'Fecha', 'ta':'Amper fase T-A', 'sa':'Amper fase S-A', 'ra':'Amper fase R-A', 'vab':'Vab', 'vca':'Vca', 'vbc':'Vbc', 'kwatts':'Kwatts 3 fases', 'potencia':'Factor de Poten-A'}
    # columnas de interes
    interesting_col = ['ta','sa','ra','vab','vca','vbc','kwatts','potencia']
    # asignacion de nombres para trabajar
    _ds_energia.columns = energia_work_col_names

####################################################################################
def load_clima(url_or_path):
    
    global _ds_clima, clima_origin_col_names_dic
    
    # cargo el ds del clima y me quedo con las columnas de fechas, temperatura y velocidad de viento
    _ds_clima = pd.read_csv(url_or_path, parse_dates=['time'], date_parser=dateparse)

    _ds_clima = _ds_clima.sort_values(by='time').reset_index()
    _ds_clima = _ds_clima[['time','temperature','windspeed']]
    _ds_clima.columns =['fecha','temperature','windspeed']

    clima_origin_col_names_dic = {'fecha':'time', 'temperature':'temperature','windspeed':'windspeed'}


###### 1.1
####################################################################################
def energia_check_date_and_rows():
    is_valid = True
    
    # el dataset inicial tiene 155 días de registros. Por lo cual se verifica que no se reduzca esa cantidad
    if (_ds_energia.fecha.max() - _ds_energia.fecha.min()).days < ENERGIA_DIAS:
        is_valid = False
        log('Error: Hay menos días ({}) que lo esperado ({})'.format((_ds_energia.fecha.max() - _ds_energia.fecha.min()).days), ENERGIA_DIAS)
    
    # el dataset inicial tiene 44363 filas. Por lo cual se verifica que no se reduzca esa cantidad
    if _ds_energia.shape[0] < ENERGIA_FILAS:
        is_valid = False
        log('Error: Hay menos filas ({}) de las esperadas ({})'.format(_ds_energia.shape[0]), ENERGIA_FILAS)
    
    return is_valid
        
def energia_is_num_cols_rows_valid():
    if len(_ds_energia.columns) < ENERGIA_COLUMNAS:
        log('Error: Faltan columnas')
        return False
    elif len(_ds_energia.columns) > ENERGIA_COLUMNAS:
        log('Error: Hay mas cantidad de columnas')
        return False
    
    if _ds_energia.shape[0] < 1:
        log('Error: No hay filas')
        return False
    
    return True
    

def energia_is_valid_types():
    is_valid_types = True

    if _ds_energia.dtypes.fecha != np.dtype('datetime64[ns]'):
        is_valid_types = False
        log('Tipo de dato no esperado para: Fecha -> {}'.format(_ds_energia.fecha.dtype))


    if _ds_energia.dtypes.ta != np.float64:
        is_valid_types = False
        log('Tipo de dato no esperado para: Amper fase T-A -> {}'.format(_ds_energia.dtypes.ta))

    if _ds_energia.dtypes.sa != np.float64:
        is_valid_types = False
        log('Tipo de dato no esperado para: Amper fase S-A -> {}'.format(_ds_energia.dtypes.sa))

    if _ds_energia.dtypes.ra != np.float64:
        is_valid_types = False
        log('Tipo de dato no esperado para: Amper fase R-A -> {}'.format(_ds_energia.dtypes.ra))


    if _ds_energia.dtypes.vab != np.float64:
        is_valid_types = False
        log('Tipo de dato no esperado para: Vab -> {}'.format(_ds_energia.dtypes.vab))
    if _ds_energia.dtypes.vca != np.float64:
        is_valid_types = False
        log('Tipo de dato no esperado para: Vca -> {}'.format(_ds_energia.dtypes.vca))
    if _ds_energia.dtypes.vbc != np.float64:
        is_valid_types = False
        log('Tipo de dato no esperado para: Vbc -> {}'.format(_ds_energia.dtypes.vbc))


    if _ds_energia.dtypes.kwatts != np.float64:
        is_valid_types = False
        log('Tipo de dato no esperado para: Kwatts 3 fases -> {}'.format(_ds_energia.dtypes.kwatts))

    if _ds_energia.dtypes.potencia != np.float64:
        is_valid_types = False
        log('Tipo de dato no esperado para: Factor de Poten-A -> {}'.format(_ds_energia.dtypes.potencia))
        
    return is_valid_types




####################################################################################
def clima_check_date_and_rows():
    is_valid = True
    
    # el dataset inicial tiene 155 días de registros. Por lo cual se verifica que no se reduzca esa cantidad
    if (_ds_clima.fecha.max() - _ds_clima.fecha.min()).days < CLIMA_DIAS:
        is_valid = False
        log('Error: Hay menos días ({}) que lo esperado ({})'.format((_ds_clima.fecha.max() - _ds_energia.fecha.min()).days), CLIMA_DIAS)
    
    # el dataset inicial tiene 3001 filas. Por lo cual se verifica que no se reduzca esa cantidad
    if _ds_clima.shape[0] < CLIMA_FILAS:
        is_valid = False
        log('Error: Hay menos filas ({}) de las esperadas ({})'.format(_ds_clima.shape[0]), CLIMA_FILAS)
    
    return is_valid

def clima_is_num_cols_rows_valid():
    if len(_ds_clima.columns) < CLIMA_COLUMNAS:
        log('Error: Faltan columnas')
        return False
    elif len(_ds_clima.columns) > CLIMA_COLUMNAS:
        log('Error: Hay mas cantidad de columnas')
        return False
    
    if _ds_clima.shape[0] < 1:
        log('Error: No hay filas')
        return False
    
    return True

def clima_is_valid_types():
    is_valid_types = True

    if _ds_clima.dtypes.fecha != np.dtype('datetime64[ns]'):
        is_valid_types = False
        log('Tipo de dato no esperado para: Fecha -> {}'.format(_ds_clima.fecha.dtype))

    if _ds_clima.dtypes.temperature != np.float64:
        is_valid_types = False
        log('Tipo de dato no esperado para: Temperature -> {}'.format(_ds_clima.dtypes.temperature))

    if _ds_clima.dtypes.windspeed != np.float64:
        is_valid_types = False
        log('Tipo de dato no esperado para: Windspeed -> {}'.format(_ds_clima.dtypes.windspeed))
    
    return is_valid_types




###### 1.2
####################################################################################
def exist_datetimes_duplicates(ds):
    if len(ds[ds.fecha.duplicated()]) == 0:
        return False
    else:
        return True




###### 2.2
####################################################################################

def summary_na_values_pct(ds, interesting_col):
    count_total_rows = ds.shape[0]
    count_total_cols = ds.shape[1]
    count_total_na = ds.isna().sum()
    
    for col in interesting_col:
        if count_total_cols > 3:
            real_col_name = energia_origin_col_names_dic[col]
        else:
            real_col_name = clima_origin_col_names_dic[col]
            
        log('Cantidad de valores nulos en la columna: {:_<30} {:_<10} {}%'
            .format(real_col_name, ds[col].isnull().sum(), round(((100 * ds[col].isnull().sum()) / count_total_rows), 2)))

    



##### 2.5
####################################################################################
def validar_potencia(ds):
    
    validate = False
    
    if ds[(ds['potencia'].abs() > 1) | (ds['potencia'].abs() < 0)].shape[0] > 0:
        log('Error: La potencia es un valor entre 0 y 1.')
    else:
        validate = True
    
    return validate


    
def validar_negativos(ds):
    
    ambos_negativos = len(ds[((ds.kwatts < 0) & (ds.potencia < 0)) == True])
    kwatts_pos_potencia_neg = len(ds[((ds.kwatts >= 0) & (ds.potencia < 0)) == True])
    kwatts_neg_potencia_pos = len(ds[((ds.kwatts < 0) & (ds.potencia >= 0)) == True])

    if (ambos_negativos > 0) | (kwatts_pos_potencia_neg > 0) | (kwatts_neg_potencia_pos > 0):
        log('Hay valores negativos. Se los tratará posteriormente')

    log('Potencia negativa y factor de potencia negativo: {}'.format(ambos_negativos))
    log('Potencia positiva y factor de potencia negativo: {}'.format(kwatts_pos_potencia_neg))
    log('Potencia negativa y factor de potencia positiva: {}'.format(kwatts_neg_potencia_pos))
    



##### 2.6
####################################################################################

### Cargar el dataset _ds_energia
### Renombrar las columnas como energia_work_col_names y definir las columnas intersting_col

### energia_work_col_names = ['fecha','ta','sa','ra','vab','vca','vbc','kwatts','potencia']
### interesting_col = ['ta','sa','ra','vab','vca','vbc','kwatts','potencia']

### Cargar el dataset _ds_clima
### Seleccionar las columnas ['time','temperature','windspeed']
### Renombrar las columnas como ['fecha','temperature','windspeed']


### Se genera el dataset _ds_total, limpiando outliers de ambos ds y agregando 
### las columnas del ds clima al ds energía además de completar datos de estas
### columnas 

def merge_datasets():
    _ds_energia_tomerge = _ds_energia

    log('Se convierten todos los valores en positivos')
    ### limpieza de los valores del ds de energía
    _ds_energia_tomerge[interesting_col] = np.abs(_ds_energia[interesting_col])

    log('Se completan datos faltantes en el dataset de energia en caso de tener un valor NaN entre 2 valores numéricos')
    # completo datos faltantes en el dataset de energia en caso de tener un valor NaN entre 2 valores numéricos
    for col in interesting_col:
      _ds_energia_tomerge[col].interpolate(method='linear',inplace=True,limit=1,limit_area='inside')


    log('Se reemplazan outliers en el dataset de Energia:')
    log('En caso de ser mayores al valor mediana + 1.5*CI eliminamos el registo')
    log('En caso de ser menores al valor mediana - 1.5*CI reemplazamos el valor por 0')
    # reemplazo outliers
    for col in interesting_col:
      CI =  (_ds_energia_tomerge[col].quantile(0.75)-_ds_energia_tomerge[col].quantile(0.25))
      _ds_energia_tomerge[col] = _ds_energia_tomerge[col].mask((_ds_energia_tomerge[col] - _ds_energia_tomerge[col].median()) > 1.5*CI , np.nan)
      _ds_energia_tomerge[col] = _ds_energia_tomerge[col].mask((_ds_energia_tomerge[col].median() - _ds_energia_tomerge[col]) > 1.5*CI , 0)

    #### limpieza de los valores del ds de clima

    log('Se reemplazan outliers en el dataset de Clima:')
    # reemplazo outliers
    _ds_clima_tomerge = _ds_clima
    CI =  (_ds_clima_tomerge['temperature'].quantile(0.75)-_ds_clima_tomerge['temperature'].quantile(0.25))
    _ds_clima_tomerge['temperature'] = _ds_clima_tomerge['temperature'].mask(abs((_ds_clima_tomerge['temperature'] - _ds_clima_tomerge['temperature'].median())) > 2*CI , np.nan)
    CI =  (_ds_clima['windspeed'].quantile(0.75)-_ds_clima['windspeed'].quantile(0.25))
    _ds_clima_tomerge['windspeed'] = _ds_clima_tomerge['windspeed'].mask(abs((_ds_clima_tomerge['windspeed'] - _ds_clima_tomerge['windspeed'].median())) > 2*CI , np.nan)

    log('Se ordenan por fecha los datasets')
    # ordeno por fecha los ds
    _ds_energia_tomerge = _ds_energia_tomerge.sort_values(by='fecha').reset_index()
    _ds_clima_tomerge = _ds_clima_tomerge.sort_values(by='fecha').reset_index()
    _ds_energia_tomerge.drop(columns=['index'], inplace=True)
    _ds_clima_tomerge.drop(columns=['index'], inplace=True)

    log('Se unen ambos datasets agregando las mediciones del dataset clima al dataset energia')
    # uno los 2 ds, agregando las mediciones del dataset clima al dataset energia
    _ds_total = pd.merge(_ds_energia_tomerge,_ds_clima_tomerge,on='fecha',how='left')

    log('Se completan datos faltantes interpolando las mediciones de temperature y windspeed')
    # completo datos faltantes interpolando las mediciones de temperature y windspeed
    _ds_total['temperature'].interpolate(method='pchip',inplace=True,limit_direction='forward')
    _ds_total['windspeed'].interpolate(method='pchip',inplace=True,limit_direction='forward')

    _ds_total = _ds_total.rename(columns=energia_origin_col_names_dic)
    
    return _ds_total


##### 2.8
####################################################################################

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_zip(zip_filepath, ds_filepath, logs_filepath):
    
    # create a ZipFile object
    zipObj = ZipFile(zip_filepath, 'w')
    
    # Add multiple files to the zip
    zipObj.write(ds_filepath, os.path.basename(ds_filepath))
    zipObj.write(logs_filepath, os.path.basename(logs_filepath))
    
    # close the Zip File
    zipObj.close()


def guardar_dataset(ds, prefix):
    
    date = datetime.today().strftime('%Y%m%d')
    time = datetime.today().strftime('%H%M%S')

    dir = os.path.join('generated_out', date)
    create_dir(dir)
    
    filename = prefix + date + '_' + time
    ds_filepath = os.path.join(dir, filename + '.csv')
    logs_filepath = os.path.join(dir, filename + '.log')
    zip_filepath = os.path.join(dir, filename + '.zip')
    
    ds.to_csv(ds_filepath, index = False, header=True, encoding='utf-8', float_format='%.2f')
    df_logs.to_csv(logs_filepath, index = False, header=False, encoding='utf-8')

    create_zip(zip_filepath, ds_filepath, logs_filepath)

    os.remove(ds_filepath)
    os.remove(logs_filepath)
    
    print('Dataset y registro de cambios guardado en: {}'.format(zip_filepath))





####################################################################################

def get_ds(url_energia, url_clima, saveInFile=False):
    
    load_energia(url_energia)
    load_clima(url_clima)
    
    if energia_is_num_cols_rows_valid() & energia_is_valid_types() & energia_check_date_and_rows():
        log('Dataset Energía importado correctamente')
        
    if clima_is_valid_types() & clima_is_num_cols_rows_valid() & clima_check_date_and_rows():
        log('Dataset Clima importado correctamente')

    if exist_datetimes_duplicates(_ds_energia) & exist_datetimes_duplicates(_ds_clima):
        log('Verificacion de Fecha y Hora únicas: Hay duplicados.')
    else:
        log('Verificacion de Fecha y Hora únicas: OK')
        
    log('Resumen de valores nulos para dataset Clima')
    summary_na_values_pct(_ds_clima, ['fecha','temperature','windspeed'])
    print('')
    log('Resumen de valores nulos para dataset Energia')
    summary_na_values_pct(_ds_energia, interesting_col)
    
    if validar_potencia(_ds_energia):
        log('El rango de valores de la potencia es correcto.')
    
    validar_negativos(_ds_energia)
    
    if saveInFile == True:
        guardar_dataset(merge_datasets(), 'ds_energia_clima_')
        return 0
    else:
        return merge_datasets()

    

ENERGIA_DS_URL = 'https://raw.githubusercontent.com/alaain04/diplodatos/master/data/energia_completo.csv'
CLIMA_DS_URL = 'https://raw.githubusercontent.com/alaain04/diplodatos/master/data/clima_posadas_20192020.csv'

#get_ds(ENERGIA_DS_URL, CLIMA_DS_URL, True)
