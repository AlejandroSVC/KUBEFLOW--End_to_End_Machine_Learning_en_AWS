# XGBoost en AWS con Kubeflow

## Clasificación binaria XGBoost mínima, robusta y escalable en AWS con Kubeflow y PySpark

Esta guía describe un flujo de trabajo práctico para realizar una clasificación binaria XGBoost escalable en AWS, utilizando Kubeflow Pipelines para la orquestación y PySpark para el procesamiento distribuido de datos. Demostraremos las mejores prácticas para garantizar la robustez y la escalabilidad, incluyendo la lógica condicional basada en el tamaño de los datos y el manejo correcto de los archivos Parquet.

## Prerrequisitos y configuración

Clúster de AWS y Kubernetes

•  Aprovisione un clúster de EKS (Elastic Kubernetes Service) en AWS con los nodos de trabajo necesarios (CPU/GPU según las necesidades de datos/computación previstas).

•  Asegúrese de que los roles y permisos de AWS IAM estén configurados para su uso de EKS y Kubeflow.

Implementación de Kubeflow

•  Implemente Kubeflow siguiendo la documentación oficial de Kubeflow.

•  Confirme que las tuberías, notebooks y Katib de Kubeflow (para el ajuste de hiperparámetros) estén operativas.

Almacenamiento S3

•  Configure un bucket S3 para cargar y guardar datos, artefactos intermedios y modelos entrenados (esto garantiza la escalabilidad y la portabilidad).

Bibliotecas de PySpark y XGBoost

Instale los siguientes paquetes en su contenedor de entrenamiento de Docker o notebook de Kubeflow:
```
bash
pip install pyspark xgboost xgboost4j findspark
```
# Creación de una pipeline de Kubeflow

## Paso 1: Carga de Datos con PySpark

Objetivo: Cargar un archivo Parquet local, determinar el tamaño de los datos y enrutar la pipeline según corresponda.

## data_loader.py
```
from pyspark.sql import SparkSession         # Importar SparkSession para operaciones con DataFrames de Spark

def load_and_check_size(input_path: str, size_threshold: int = 1000000) -> str:
    # Inicializar SparkSession
    spark = SparkSession.builder.appName("XGBoostBinaryDataLoader").getOrCreate()
    # ^ Crea/obtiene una SparkSession con nombre de aplicación para gestión de recursos en clúster

    # Cargar archivo Parquet desde disco local (o S3 si es necesario)
    df = spark.read.parquet(input_path)
    # ^ Lee el archivo Parquet en un DataFrame de Spark (se admite ruta de AWS S3 mediante s3a://)

    # Contar el número de filas para determinar el tamaño del conjunto de datos
    row_count = df.count()
    # ^ Operación de acción que desencadena cómputo distribuido (costoso para conjuntos de datos grandes)

    # Bandera de enrutamiento basada en umbral
    if row_count < size_threshold:
        return "small"        # El conjunto de datos cabe en procesamiento de un solo nodo
    else:
        return "large"         # Requiere procesamiento distribuido
```
Mejor práctica: Iniciar siempre SparkSession en modo distribuido dentro de un contenedor gestionado y cerrarla después del trabajo para liberar recursos.

## Paso 2: Preprocesamiento de Datos con PySpark

• Para conjuntos de datos pequeños, recopilar como DataFrame de pandas para operaciones eficientes en un solo nodo.

• Para conjuntos de datos grandes, procesar en una cadena de transformaciones distribuida con PySpark, escribiendo la salida de nuevo en Parquet (particionado si es necesario).

Ejemplo:
```
def preprocess_data(input_path: str, output_path: str, mode: str):
    spark = SparkSession.builder.appName("Preprocessing").getOrCreate()
    df = spark.read.parquet(input_path)       # Cargar datos de origen

    if mode == "small":
        pdf = df.toPandas()                      # Convertir a DataFrame de pandas (recopila datos en el nodo principal)
        # ... cualquier preprocesamiento basado en pandas
        # ^ Usar pandas para operaciones complejas en un solo nodo (ej., preprocesamiento con scikit-learn)
        pdf.to_parquet(output_path)    # Guardar como Parquet (local/S3)
    else:
        # ... Transformaciones basadas en PySpark (ingeniería de características, fillna, filtrado, etc.)
        df = df.dropna()                            # Ejemplo: Manejo distribuido de valores nulos
        # ^ Añadir ingeniería de características, escalado, etc. usando la API de DataFrame de Spark
        df.write.mode("overwrite").parquet(output_path)      # Escritura distribuida
        # ^ El modo 'overwrite' garantiza ejecuciones de pipeline idempotentes

    spark.stop()             # Cerrar la sesión de Spark de manera limpia (importante para la gestión de recursos)

# Rutas de código separadas manejan diferencias de volumen de datos eficientemente; no usar collect() en conjuntos de datos grandes.
```

## Paso 3: Lógica Condicional del Pipeline (Kubeflow DSL)

Usar dsl.Condition o dsl.IfElse para bifurcar el flujo de trabajo de Kubeflow basado en el tamaño de los datos:
```
import kfp.dsl as dsl       # Importar DSL de Kubeflow Pipelines

@dsl.pipeline(
    name="Robust XGBoost Binary Classifier Pipeline",
    description="Pipeline condicional de XGBoost basado en el tamaño de los datos"
)
def binary_classification_pipeline(input_path: str, size_threshold: int):
    data_size_task = data_loader_op(input_path, size_threshold)
     # ^ ComponentOp para verificación del tamaño de datos (debe definirse como componente de Kubeflow)

    # Rama de ejecución condicional
    with dsl.Condition(data_size_task.output == "small"):
        # Flujo de trabajo para datos pequeños
        preprocess_task = preprocess_data_op(input_path, "output_small.parquet", "small")
        xgboost_train_task = train_xgboost_op("output_small.parquet", mode="single_node")
        # ^ Usa XGBoost de un solo nodo para conjuntos de datos pequeños

    with dsl.Condition(data_size_task.output == "large"):
        # Flujo de trabajo para datos grandes
        preprocess_task = preprocess_data_op(input_path, "output_large.parquet", "large")
        xgboost_train_task = train_xgboost_op("output_large.parquet", mode="distributed")
        # ^ Usa SparkXGBClassifier para entrenamiento distribuido

# Evitar sentencias if de Python en el código del pipeline; usar condiciones DSL de Kubeflow.
```

## Paso 4: Entrenamiento Distribuido de XGBoost

Componente de Entrenamiento

• Usar la integración XGBoost4J-Spark para XGBoost distribuido real dentro de PySpark.

• Para un solo nodo (datos pequeños), se puede usar el XGBClassifier estándar de XGBoost.

• Para datos grandes, entrada Parquet particionada, lanzar SparkXGBClassifier distribuido.

Ejemplo de esqueleto de entrenamiento distribuido:
```
# Entrenamiento Distribuido de XGBoost
from xgboost.spark import SparkXGBClassifier  # Integración XGBoost4J-Spark

def train_xgboost(input_path: str, mode: str, model_output: str):
    spark = SparkSession.builder.appName("XGBoostTrain").getOrCreate()
    df = spark.read.parquet(input_path)      # Cargar datos preprocesados

    if mode == "single_node":
        # Convertir a Pandas para entrenamiento en un solo nodo
        pdf = df.toPandas()
        import xgboost as xgb
        model = xgb.XGBClassifier()    # Clasificador estándar con API de scikit-learn
        model.fit(pdf.drop('label',axis=1), pdf['label'])    # Entrenamiento en un solo nodo
        # Guardar modelo
        model.save_model(model_output)    # Guardar en formato nativo de XGBoost
    else:
        # XGBoost distribuido usando XGBoost4J-Spark
        xgb_classifier = SparkXGBClassifier(
            featuresCol="features",     # Nombre de la columna de características en Spark ML
            labelCol="label",                  # Nombre de la columna de etiquetas
            numWorkers=4,                   # Número de ejecutores de Spark para entrenamiento: ajustar según el clúster
            maxDepth=6,                        # Hiperparámetro del modelo
            objective='binary:logistic'   # Objetivo de clasificación binaria
        )
        model = xgb_classifier.fit(df)                                          # Entrenamiento distribuido en clúster de Spark
        model.nativeBooster.save_model(model_output)    # Acceder al modelo subyacente

    spark.stop()


# Ajustar numWorkers, particiones, etc. según el tamaño del clúster y el volumen de datos.
# Asegurar siempre que los recursos del clúster de Spark estén configurados adecuadamente para el volumen de datos anticipado.

# Mejores Prácticas de Kubeflow
```
• Containerizar cada componente en una imagen Docker con todas las dependencias y subirla a un registro.

• Usar Parámetros de Pipeline para entradas dinámicas, como ruta del archivo de datos o tamaño del clúster.

• Los datos de entrada/salida deben residir en ubicaciones accesibles desde S3 para portabilidad y escalabilidad del pipeline.

• Aprovechar la visualización del pipeline para depuración, seguimiento de experimentos y reproducibilidad en la interfaz de Kubeflow.

• Usar Conditionals y ParallelFor en el pipeline para flujos de trabajo escalables y modulares, como se muestra arriba.

• Registrar métricas en cada etapa, especialmente artefactos de entrenamiento y evaluación, para seguimiento y mejores prácticas de registro de modelos.

# Ejemplo: Esqueleto Completo del Pipeline (Kubeflow DSL)
```
import kfp
import kfp.dsl as dsl

@dsl.pipeline(
    name='XGBoost Binary Classification Pipeline',
    description='Pipeline minimalista robusto con PySpark y XGBoost distribuido'
)
def pipeline(input_parquet: str, size_threshold: int = 1000000):
    # Paso 1: Verificar tamaño de los datos
    check_task = data_loader_op(input_parquet, size_threshold)
    # ^ Primera etapa del pipeline - devuelve 'small' o 'large'
    
    # Paso 2: Ejecución de Ramas Condicionales
    with dsl.Condition(check_task.output == 'small'):
        # Ruta para datos pequeños
        preprocess_task = preprocess_data_op(input_parquet, 'prep_small.parquet', 'small')
        train_task = train_xgboost_op('prep_small.parquet', 'single_node', 'model_small.bin')

    with dsl.Condition(check_task.output == 'large'):
        # Ruta para datos grandes
        preprocess_task = preprocess_data_op(input_parquet, 'prep_large.parquet', 'large')
        train_task = train_xgboost_op('prep_large.parquet', 'distributed', 'model_large.bin')


# Reemplazar funciones _op con kfp.components.create_component_from_func o contenedores personalizados.
# Parametrizar la asignación de recursos según los tipos de instancia de AWS.

```
Consideraciones Técnicas Clave:

Spark en Kubernetes:

• SparkSession se ejecuta en Kubernetes mediante el operador Spark-on-K8s

• Asignación de recursos controlada mediante parámetros de spark-submit (no mostrado)

• Referencia: Integración de Spark con Kubernetes

XGBoost Distribuido:

• Usa XGBoost4J-Spark para entrenamiento distribuido

• Requiere Java 8+ y versiones compatibles de Spark/XGBoost

• Referencia: Guía de XGBoost4J-Spark

Mejores Prácticas de Kubeflow:

• dsl.Condition permite enrutamiento dinámico del pipeline

• Las salidas de los componentes se pasan mediante el atributo output

• Referencia: DSL de Kubeflow Pipelines

Integración con AWS:

• Usar rutas S3 (s3a://) para datos de entrada/salida

• Configurar credenciales de AWS para Hadoop en la configuración de Spark

• Referencia: Integración de Spark con S3

Consideraciones de Rendimiento:

• df.count() es costoso para conjuntos de datos grandes - considerar muestreo

• Parquet particionado permite lecturas/escrituras paralelas

• Ajustar numWorkers según el tamaño del clúster

Compensaciones de Escalabilidad:

• Ruta de un solo nodo: Más simple pero limitada por la memoria del driver

• Ruta distribuida: Mayor sobrecarga pero maneja datos a escala de TB

• El valor del umbral (1M filas) debe ajustarse según:

1. Dimensionalidad de las características
2. Tipos de instancia
3. Recursos del clúster

Para despliegue en producción:

• Containerizar componentes con Docker

• Configurar plantillas de recursos de Spark

• Añadir etapas de validación/despliegue del modelo

• Implementar monitoreo con Amazon CloudWatch

Recomendaciones

• Usar PySpark para procesamiento de datos escalable y distribuido. Nunca usar collect() o convertir conjuntos de datos grandes a Pandas.

• Aprovechar XGBoost4J-Spark para entrenamiento distribuido real, especialmente para conjuntos de datos que no caben en memoria.

• Diseñar pipelines de Kubeflow con condicionales para enrutar automáticamente datos pequeños o grandes a los recursos y rutas de código apropiados.

• Siempre containerizar componentes y usar S3 para entrada/salida para portabilidad.

• Ajustar el clúster y el paralelismo de entrenamiento según el tamaño del conjunto de datos y los nodos EC2/GPU/CPU disponibles.

• Monitorear y registrar todas las métricas para un seguimiento robusto de experimentos y reproducibilidad.

• Usar la documentación oficial de XGBoost con Spark y los patrones distribuidos y condicionales de Kubeflow como referencias fundamentales.
