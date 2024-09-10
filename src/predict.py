# Código de Scoring - Modelo de Riesgo de Default en un Banco de Corea
############################################################################

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os


# Cargar la tabla transformada
def score_model(filename, scores):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de Scoring    
    res = model.predict(df).reshape(-1,1)
    pred = pd.DataFrame(res, columns=['PREDICT'])
    pred.to_csv(os.path.join('../data/scores/', scores))
    print(scores, 'exportado correctamente en la carpeta scores')


# Scoring desde el inicio
def main():
    df = score_model('Morosidad_score.csv','final_score.csv')
    print('Finalizó el Scoring del Modelo')


if __name__ == "__main__":
    main()