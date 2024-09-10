# Código de Entrenamiento

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename), sep = ';')
    X_train = df.drop(['TIPOCONTACTO'],axis=1)
    y_train = df[['TIPOCONTACTO']]
    print(filename, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    model_rf = RandomForestClassifier()
    model_rf.fit(X_train, y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(model_rf, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')

# Entrenamiento completo
def main():
    read_file_csv('Morosidad_train.csv')
    print('Finalizó el entrenamiento del Modelo')

if __name__ == "__main__":
    main()