# Proyecto: Árbol de Decisión con Dataset del Clima

Este proyecto aplica **técnicas de minería de datos y aprendizaje supervisado** para la **construcción de un árbol de decisión** capaz de predecir si se debe jugar al aire libre basándose en las condiciones climáticas, utilizando un dataset de clima.

El objetivo principal es **generar reglas de decisión interpretables** y comparar el rendimiento entre reglas manuales y algoritmos automáticos de árbol de decisión. El proyecto incluye visualización del árbol resultante, evaluación del modelo mediante métricas de desempeño y generación de reportes HTML completos.

---

## Objetivo del Proyecto

El proyecto tiene como objetivo:

1. **Comparar métodos de decisión**: Evaluar la efectividad de reglas de decisión manuales vs. algoritmos automáticos de árbol de decisión
2. **Generar reglas interpretables**: Crear reglas claras que expliquen cuándo es apropiado jugar al aire libre según las condiciones climáticas
3. **Visualizar el proceso de decisión**: Mostrar gráficamente cómo el algoritmo toma decisiones
4. **Evaluar el rendimiento**: Medir la precisión de ambos enfoques mediante métricas cuantitativas
5. **Documentar resultados**: Generar reportes HTML profesionales con análisis comparativo

---

## Descripción del Dataset

El conjunto de datos contiene información sobre condiciones climáticas y decisiones de juego al aire libre. Las variables incluyen:

- **Outlook**: Condiciones del cielo (sunny, overcast, rain)
- **Temperature**: Temperatura (hot, mild, cool)  
- **Humidity**: Humedad (high, normal)
- **Windy**: Condiciones de viento (True, False)
- **Play**: Decisión objetivo (yes, no) - si se juega al aire libre o no

El dataset se utiliza para entrenar un modelo que pueda predecir si es apropiado jugar al aire libre basándose en las condiciones meteorológicas.

---

## Estructura del Proyecto

```bash
decisionTreeRules/
│
├── data/
│   └── weather_dataset.xlsx        # Dataset de condiciones climáticas
│
├── notebooks/                      # Directorio para notebooks de análisis
│
├── outputs/
│   └── reporte_arbol_decision.html # Reporte HTML generado automáticamente
│
├── src/
│   ├── __init__.py                 
│   ├── data_loader.py              # Función para cargar datasets (CSV/Excel)
│   ├── preprocess.py               # Reglas de decisión manuales y preprocesamiento
│   ├── train_model.py              # Entrenamiento y evaluación del árbol de decisión
│   ├── visualize.py                # Generación de reportes HTML y visualizaciones
│   └── _init__.py                  # Archivo de inicialización del paquete
│
├── main.py                         # Script principal que ejecuta todo el flujo
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Documentación del proyecto
```

---

## Instalación y Uso

### Requisitos del Sistema

- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Instalación de Dependencias

```bash
# Clonar o descargar el proyecto
cd decisionTreeRules

# Instalar las dependencias
pip install -r requirements.txt
```

### Ejecución del Proyecto

```bash
# Ejecutar el análisis completo
python main.py

# O ejecutar directamente la generación de reportes
python -m src.visualize
```

---

## Salidas del Proyecto

### 1. Reporte HTML (`outputs/reporte_arbol_decision.html`)

El proyecto genera un reporte HTML completo que incluye:

- **Descripción del dataset**: Información sobre las variables y dimensiones
- **Comparación de métodos**: Tabla comparativa entre regla manual y árbol de decisión
- **Métricas de rendimiento**: Precisión, aciertos y total de instancias para cada método
- **Visualización del árbol**: Imagen del árbol de decisión generado
- **Características del modelo**: Profundidad, número de nodos, atributos utilizados
- **Metadatos**: Fecha de ejecución, configuración del modelo

### 2. Salida en Consola

Durante la ejecución, el programa muestra:

- Dataset original cargado
- Predicciones de la regla manual con métricas
- Resultados del árbol de decisión con métricas
- Confirmación de generación del reporte HTML

### 3. Métricas Generadas

- **Precisión**: Porcentaje de predicciones correctas
- **Aciertos**: Número de instancias clasificadas correctamente
- **Total de instancias**: Tamaño del conjunto evaluado
- **Características del árbol**: Profundidad y número de nodos hoja

---

## Metodología

### 1. Regla de Decisión Manual

Se implementa una regla simple basada en condiciones climáticas:
- Si hay lluvia y no hay viento → Jugar (yes)
- En cualquier otro caso → No jugar (no)

### 2. Árbol de Decisión Automático

- **Preprocesamiento**: Codificación de variables categóricas usando LabelEncoder
- **División de datos**: 80% entrenamiento, 20% prueba
- **Algoritmo**: DecisionTreeClassifier de scikit-learn
- **Evaluación**: Métricas de precisión en conjunto de prueba

### 3. Comparación de Métodos

El proyecto compara directamente:
- Efectividad de reglas manuales vs. algoritmos automáticos
- Interpretabilidad de las decisiones
- Precisión en la predicción

---

## Mejoras Futuras

### 1. Expansión del Dataset
- **Más variables climáticas**: Agregar presión atmosférica, velocidad del viento, índice UV
- **Datos temporales**: Incluir información de estaciones del año, hora del día
- **Datos geográficos**: Considerar ubicación geográfica y microclimas

### 2. Mejoras en el Modelo
- **Validación cruzada**: Implementar k-fold cross-validation para evaluación más robusta
- **Múltiples algoritmos**: Comparar con Random Forest, SVM, Naive Bayes
- **Optimización de hiperparámetros**: Grid search para encontrar mejores parámetros
- **Métricas adicionales**: Recall, F1-score, matriz de confusión detallada

### 3. Funcionalidades Adicionales
- **API REST**: Crear endpoint para predicciones en tiempo real
- **Interfaz web**: Dashboard interactivo para visualizar resultados
- **Predicciones en tiempo real**: Integración con APIs meteorológicas
- **Notificaciones**: Sistema de alertas basado en condiciones climáticas

### 4. Mejoras Técnicas
- **Logging**: Sistema de logs para monitoreo y debugging
- **Configuración**: Archivo de configuración para parámetros del modelo
- **Tests unitarios**: Suite de pruebas para validar funcionalidades
- **Docker**: Containerización para facilitar despliegue

### 5. Análisis Avanzado
- **Análisis de importancia de características**: Identificar qué variables son más influyentes
- **Análisis de errores**: Estudiar casos donde el modelo falla
- **Visualizaciones interactivas**: Gráficos dinámicos con Plotly o Bokeh
- **Análisis de tendencias**: Identificar patrones temporales en los datos

### 6. Documentación y Usabilidad
- **Tutorial interactivo**: Jupyter notebook con ejemplos paso a paso
- **Documentación de API**: Documentación detallada de todas las funciones
- **Ejemplos de uso**: Casos de uso reales y ejemplos prácticos
- **Guía de contribución**: Instrucciones para colaboradores

---

## Tecnologías Utilizadas

- **Python 3.7+**: Lenguaje de programación principal
- **pandas**: Manipulación y análisis de datos
- **scikit-learn**: Algoritmos de machine learning
- **matplotlib**: Visualización de datos
- **numpy**: Computación numérica
- **openpyxl**: Lectura de archivos Excel
- **seaborn**: Visualizaciones estadísticas avanzadas

---

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

