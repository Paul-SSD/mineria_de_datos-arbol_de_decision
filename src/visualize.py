import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import base64
from io import BytesIO
from datetime import datetime
import os
from .data_loader import cargar_dataset
from .preprocess import aplicarRegla
from .train_model import codificar_dataset, entrenar_modelo, evaluar_modelo


def generar_reporte_html(dataset_path="data/weather_dataset.xlsx", output_dir="outputs"):
    """
    Función para generar reporte HTML del análisis de árbol de decisión
    con comparación entre regla manual y árbol de decisión.
    """
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Cargar dataset
    print("Cargando dataset...")
    df = cargar_dataset(dataset_path)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # 2. Aplicar regla de decisión manual
    print("Aplicando regla de decisión manual...")
    prediccion_columna_objetivo, aciertos, precision = aplicarRegla(df)
    print(f"Regla manual - Aciertos: {aciertos}, Precisión: {precision:.2f}")
    
    # 3. Preparar datos para árbol de decisión
    print("Preparando datos para árbol de decisión...")
    df_codificado = codificar_dataset(df)
    
    # Separar atributos y clase objetivo
    columna_objetivo = "Play"
    atributos = df_codificado.drop(columns=[columna_objetivo])
    clase = df_codificado[columna_objetivo]
    
    # Dividir dataset en entrenamiento y prueba
    random_seed = 10
    porcion_datos_entrenamiento = 0.2
    atributos_training, atributos_test, clase_training, clase_test = train_test_split(
        atributos,
        clase,
        test_size=porcion_datos_entrenamiento,
        random_state=random_seed,
    )
    
    # 4. Entrenar árbol de decisión
    print("Entrenando árbol de decisión...")
    arbol = DecisionTreeClassifier(random_state=random_seed)
    modelo = entrenar_modelo(arbol, atributos_training, clase_training)
    
    # 5. Evaluar árbol de decisión
    print("Evaluando árbol de decisión...")
    predicciones_arbol, aciertos_arbol, precision_arbol = evaluar_modelo(
        arbol, atributos_test, clase_test
    )
    print(f"Árbol (conjunto de prueba) - Aciertos: {aciertos_arbol}, Precisión: {precision_arbol:.2f}")
    
    
    # 6. Generar visualización del árbol
    print("Generando visualización del árbol...")
    plt.figure(figsize=(15, 10))
    tree.plot_tree(
        arbol,
        feature_names=atributos.columns,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True,
        fontsize=8
    )
    
    # Guardar la imagen en memoria
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    buffer.seek(0)
    imagen_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # 7. Crear reporte HTML
    print("Generando reporte HTML...")
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Crear tabla de comparación (sin evaluación en dataset completo)
    tabla_comparacion = pd.DataFrame({
        'Método': ['Regla Manual', 'Árbol (Prueba)'],
        'Aciertos': [aciertos, aciertos_arbol],
        'Precisión': [f"{precision:.2f}", f"{precision_arbol:.2f}"],
        'Total Instancias': [len(df), len(clase_test)]
    })
    
    reporte_html = f"""
<html>
<head>
    <title>Análisis Comparativo: Reglas de Decisión vs Árbol de Decisión</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            margin: 40px;
            background-color: #ffffff;
            line-height: 1.6;
            color: #333;
        }}
        .header {{
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #2c3e50;
            font-size: 28px;
            margin-bottom: 10px;
            text-align: center;
        }}
        h2 {{
            color: #34495e;
            font-size: 20px;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #2c3e50;
            font-size: 16px;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin-bottom: 25px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        th, td {{
            border: 1px solid #bdc3c7;
            padding: 12px;
            text-align: center;
        }}
        th {{
            background-color: #34495e;
            color: white;
            font-weight: bold;
        }}
        .results-table {{
            margin: 20px 0;
        }}
        .method-comparison {{
            background-color: #f8f9fa;
            padding: 20px;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            margin-bottom: 25px;
        }}
        .method-section {{
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .performance-metrics {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-box {{
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #007bff;
            border-radius: 3px;
        }}
        .tree-visualization {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        .model-info {{
            background-color: #e8f4fd;
            padding: 20px;
            border-radius: 5px;
            margin-top: 25px;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #2c3e50;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 10px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Análisis Comparativo: Reglas de Decisión vs Árbol de Decisión</h1>
        <div class="metadata">
            <p><strong>Fecha de ejecución:</strong> {fecha_actual}</p>
            <p><strong>Dataset:</strong> {dataset_path}</p>
            <p><strong>Método de evaluación:</strong> Comparación de precisión entre regla manual y algoritmo de árbol de decisión</p>
        </div>
    </div>

    <h2>1. Descripción del Dataset</h2>
    <p><strong>Dimensiones:</strong> {df.shape[0]} instancias × {df.shape[1]} atributos</p>
    <p><strong>Variables:</strong> Outlook, Temperature, Humidity, Windy, Play</p>
    <div class="results-table">
        {df.head().to_html(index=False, classes='table')}
    </div>

    <h2>2. Comparación de Métodos</h2>
    <div class="method-comparison">
        {tabla_comparacion.to_html(index=False, classes='table')}
    </div>

    <h2>3. Análisis de Resultados</h2>
    
    <div class="method-section">
        <h3>Regla de Decisión Manual</h3>
        <div class="performance-metrics">
            <div class="metric-box">
                <strong>Aciertos:</strong> {aciertos} de {len(df)} instancias
            </div>
            <div class="metric-box">
                <strong>Precisión:</strong> {precision:.3f} ({precision*100:.1f}%)
            </div>
        </div>
        <p><strong>Evaluación:</strong> Aplicada sobre el conjunto completo de datos</p>
    </div>

    <div class="method-section">
        <h3>Árbol de Decisión (Conjunto de Prueba)</h3>
        <div class="performance-metrics">
            <div class="metric-box">
                <strong>Aciertos:</strong> {aciertos_arbol} de {len(clase_test)} instancias
            </div>
            <div class="metric-box">
                <strong>Precisión:</strong> {precision_arbol:.3f} ({precision_arbol*100:.1f}%)
            </div>
        </div>
        <p><strong>Datos de entrenamiento:</strong> {len(clase_training)} instancias</p>
        <p><strong>Datos de prueba:</strong> {len(clase_test)} instancias</p>
    </div>

    

    <h2>4. Visualización del Árbol de Decisión</h2>
    <div class="tree-visualization">
        <img src="data:image/png;base64,{imagen_base64}" alt="Árbol de decisión generado" style="max-width: 100%; height: auto; border: 1px solid #dee2e6;">
    </div>

    <h2>5. Características del Modelo</h2>
    <div class="model-info">
        <p><strong>Profundidad del árbol:</strong> {arbol.get_depth()} niveles</p>
        <p><strong>Número de nodos hoja:</strong> {arbol.get_n_leaves()}</p>
        <p><strong>Atributos utilizados:</strong> {', '.join(atributos.columns.tolist())}</p>
        <p><strong>Semilla aleatoria:</strong> {random_seed}</p>
        <p><strong>Proporción de entrenamiento:</strong> {(1-porcion_datos_entrenamiento)*100:.0f}%</p>
        <p><strong>Proporción de prueba:</strong> {porcion_datos_entrenamiento*100:.0f}%</p>
    </div>

    <div class="footer">
        <p>Reporte generado automáticamente por el sistema DecisionTreeRules</p>
        <p>Fecha de generación: {fecha_actual}</p>
    </div>
</body>
</html>
"""
    
    # 8. Guardar el reporte
    output_path = os.path.join(output_dir, "reporte_arbol_decision.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(reporte_html)
    
    print(f"Reporte HTML generado en: {output_path}")
    print(f"Resumen:")
    print(f"   - Regla Manual: {aciertos}/{len(df)} aciertos ({precision:.2f})")
    print(f"   - Árbol (Prueba): {aciertos_arbol}/{len(clase_test)} aciertos ({precision_arbol:.2f})")
    
    return {
        'regla_manual': {'aciertos': aciertos, 'precision': precision, 'total': len(df)},
        'arbol_decision_prueba': {'aciertos': aciertos_arbol, 'precision': precision_arbol, 'total': len(clase_test)},
        'reporte_path': output_path
    }


# Función para ejecutar desde línea de comandos
if __name__ == "__main__":
    resultado = generar_reporte_html()
    print("Proceso completado exitosamente!")
