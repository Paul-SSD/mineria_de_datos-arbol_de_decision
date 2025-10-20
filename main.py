from src.visualize import generar_reporte_html
from src.preprocess import aplicarRegla
from src.data_loader import cargar_dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from src.train_model import (
    codificar_dataset,
    entrenar_modelo,
    evaluar_modelo,
)


def main():
    """
    Función principal que ejecuta todo el proceso de análisis
    y genera el reporte completo.
    """
    print("Iniciando análisis de árbol de decisión...")

    # visualizar el dataset
    df = cargar_dataset("data/weather_dataset.xlsx")
    print("Dataset original:")
    print(df)

    # visualizar el dataset luego de aplicar la regla de decision
    prediccion_columna_objetivo, aciertos, precision = aplicarRegla(df)
    print("\nPredicciones de la regla manual:")
    print(prediccion_columna_objetivo)
    print("Número de aciertos:", aciertos)
    print("Precisión:", precision)

    # dataset codificado: columnas categoricas a columnas numericas
    df_codificado = codificar_dataset(df)

    # separar el dataset en atributos y columna_objetivo
    columna_objetivo = "Play"
    atributos = df_codificado.drop(columns=[columna_objetivo])
    clase = df_codificado[columna_objetivo]

    # dividir dataset en entrenamiento y prueba
    random_seed = 10
    porcion_datos_entrenamiento = 0.2
    atributos_training, atributos_test, clase_training, clase_test = train_test_split(
        atributos,
        clase,
        test_size=porcion_datos_entrenamiento,
        random_state=random_seed,
    )

    # generar arbol
    arbol = DecisionTreeClassifier(random_state=random_seed)

    # entrenar arbol
    modelo = entrenar_modelo(arbol, atributos_training, clase_training)

    # evaluar arbol
    predicciones_arbol, aciertos_arbol, precision_arbol = evaluar_modelo(
        arbol, atributos_test, clase_test
    )

    print("\nResultados del árbol de decisión:")
    print(predicciones_arbol)
    print("Aciertos arbol:", aciertos_arbol)
    print("Precisión arbol:", precision_arbol)


    generar_reporte_html()


if __name__ == "__main__":
    main()
