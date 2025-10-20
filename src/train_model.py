from sklearn.tree import DecisionTreeClassifier

# convertir variables categoricas
from sklearn.preprocessing import LabelEncoder


def codificar_dataset(df):
    """
    Convierte columnas de texto a numeros usando
    labelEncoder y devuelve un df codificado
    """
    df_codificado = df.copy()
    # codifica solo columnas texto (object) y la clase
    for col in df_codificado.columns:
        if df_codificado[col].dtype == "object":
            labelEncoder = LabelEncoder()
            df_codificado[col] = labelEncoder.fit_transform(df_codificado[col])

    return df_codificado


def entrenar_modelo(arbol, training_atributes, trainig_class):
    """
    Entrena el modelo en base a los atributos y clase
    """
    arbol.fit(training_atributes, trainig_class)
    return arbol


def evaluar_modelo(arbol, test_atributes, test_class):
    """
    Calcula predicciones, aciertos y precision
    """
    predicciones = arbol.predict(test_atributes)
    aciertos = (predicciones == test_class).sum()
    precision = aciertos / len(test_class)
    return predicciones, aciertos, precision

