# Proyecto Final - Knowledge Distillation y tectécnicas de XAI para el aprendizaje

## Knowledge destillation sobre imaganes naturales

1. En principio se utilizo la red tutora VGG19, la red aprendiz era VGG19 con la mitad con la mitad de filtros convolucionales, sobre el 3% del dataset de ImageNet (1.4 millones de imagenes y 1000 clases).
2. Surge el problema del gran cuello de botella de parametros en ambas redes, la gran necesidad de memoria y tiempo de entrenamiento por el tamaño de las redes y el tamaño del dataset a pesar de ser una fraccion.
3. Se propone la utilizacion de redes que distribuyan mejor los parametros, como la familia de modelos ResNet y el dataset en principio completo de CIFAR-10 (60 mil imagenes y 100 clases).
4. Para poder comparar resultados y hacer consistente la comparacion se aplicará el mismo esquema de entrenamiento para la red tutora y la red aprendiz, y se estudiarán técnicas para mejorar este baseline mediante knowledge destillation.
5. Se aplican principalmente 3 técnicas de destilación de conocimiento.
    - Soft targets: Consiste en entrenar la red aprendiz con las salidas de la red tutora (logits), en lugar de las etiquetas originales.
    - Similarity preserving: Consiste en entrenar la red aprendiz con las salidas de la red tutora (logits) y las salidas de la red aprendiz, para que sean similares.
    - Attention transfer: Consiste en entrenar la red aprendiz con las salidas de las capas intermedias de la red tutora, para que la red aprendiz preste atención a las mismas características que la red tutora.
6. En caso de resultados prometedores, se utilizará la técnica de mejor desempeño en un escenario de escasez de recursos, es decir, la red aprendiz con menos parametros y el dataset mas pequeño.
7. Se compararán los resultados de las redes tutoras y aprendices con las métricas de accuracy y tiempo de entrenamiento además de técnias de XAI para entender el comportamiento de las redes como Grad-CAM, LIME y SHAP.

```plaintext
Debido a la cantidad de parámetros, se debieron considerar estos cambios debido a la capacidad computacional y la aplicabilidad del dataset al entrenar las diferentes arquitecturas de VGG.
```

### Detalles de la impelementación

#### Conjunto de datos

Los datos de entrenamiento se componen principalmente de imágenes del conjunto de datos CIFAR-100. Para el conjunto de entrenamiento 45.000 y validación 5.000, se aplican transformaciones como volteo horizontal aleatorio, recorte aleatorio, ajuste de color, rotación aleatoria y recorte redimensionado aleatorio para aumentar la diversidad de los datos y mejorar la capacidad del modelo para generalizar. Además, se normalizan las imágenes utilizando la media y la desviación estándar proporcionadas para ImageNet (valor comúnmente usado por su gran diversidad de imagenes). Para el conjunto de prueba, se aplica una transformación estándar sin aumentación de datos. Se utilizan DataLoader para cargar los conjuntos de datos de entrenamiento, validación y prueba con el tamaño de lote especificado, y se configuran para el procesamiento paralelo y la asignación de memoria.

#### Entrenamiento de la red tutora y la red aprendiz

El experimento se enfoca en entrenar un modelo utilizando el optimizador SGD con una tasa de aprendizaje inicial de 0.5 y un momento de 0.9. Implementa un esquema de programación de la tasa de aprendizaje con un calentamiento inicial de 5 épocas lineal, seguido de una disminución cíclica basada en la función coseno hasta que se alcanza el límite de épocas definido. Se ajusta el peso decaimiento para no afectar a las capas de normalización por lotes. Durante el entrenamiento, se evalúa la precisión y la pérdida utilizando la entropía cruzada, con un suavizado de etiquetas de 0.1.

#### Entrenamiento de la red aprendiz con destilación de conocimiento

Se implementan tres técnicas de destilación de conocimiento: soft targets, similarity preserving y attention transfer. Para cada técnica, se entrena un modelo utilizando el optimizador SGD con una tasa de aprendizaje inicial de 0.5 y un momento de 0.9. Implementa un esquema de programación de la tasa de aprendizaje con un calentamiento inicial de 5 épocas lineal, seguido de una disminución cíclica basada en la función coseno hasta que se alcanza el límite de épocas definido. Se ajusta el peso decaimiento para no afectar a las capas de normalización por lotes. Durante el entrenamiento, se evalúa la precisión y la pérdida utilizando la entropía cruzada, sin suavizado de etiquetas.

#### Evaluación de la red tutora y la red aprendiz

Se evalúan los modelos entrenados utilizando el conjunto de prueba. Se calcula la precisión y la pérdida utilizando la entropía cruzada. Además, se aplican técnicas de XAI para comprender el comportamiento de los modelos.
    - Grad-CAM: Genera mapas de activación de clase para visualizar las regiones importantes de las imágenes.
    - LIME: Explica las predicciones de los modelos utilizando un modelo localmente interpretable.
    - SHAP: Explica las predicciones de los modelos utilizando valores Shapley.

#### Resultados

Se comparan los resultados de los modelos entrenados utilizando las métricas de precisión y pérdida. Además, se analizan las visualizaciones generadas por las técnicas de XAI para comprender el comportamiento de los modelos.

### Referencias

- [Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.](https://arxiv.org/abs/1503.02531)
- [Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2014). Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550.](https://arxiv.org/abs/1412.6550)
- [Zagoruyko, S., & Komodakis, N. (2016). Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928.](https://arxiv.org/abs/1612.03928)
- [Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision (pp. 618-626).](https://arxiv.org/abs/1610.02391)
- [Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).](https://arxiv.org/abs/1602.04938)
- [Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In Advances in neural information processing systems (pp. 4765-4774).](https://arxiv.org/abs/1705.07874)
- [Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., ... & Berg, A. C. (2015). ImageNet large scale visual recognition challenge. International Journal of Computer Vision, 115(3), 211-252.](http://www.image-net.org/challenges/LSVRC/)
- [He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).](https://arxiv.org/abs/1512.03385)
- [Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.](https://arxiv.org/abs/1409.1556)

#### Reproducibilidad

1. Clonar el repositorio

    ```bash
    git clone https://github.com/M4thinking/DestillML.git && cd DestillML
    ```

2. Crear ambiente virtual, activar, updatear pip e instalar dependencias:

    ```bash
    python -m venv env
    source ./env/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. Ejecutar dataset a utilizar (cifar10, cifar100, imagenet):

    ```bash
    python dataset.py --dataset cifar100
    ```

4. Entrenar red tutora:

    ```bash
    python trainer.py --dataset cifar100 --architecture ResNet101 --epochs 600 --batch-size 128
    ```

    Además, puedes utilizar ```--show_versions``` para ver si existen más modelos entrenados bajo la misma configuración de dataset y arquitectura. Con ```--version {version}``` puedes continuar el entrenamiento de un modelo existente entregando su respectiva versión.

    Por último para ver las principales métricas de entrenamiento y validación, además de guardar el onnx del modelo, puedes utilizar

    ```bash
    python metrics.py --dataset cifar100 --architecture ResNet101 --select_version 0
    ```

5. Entrenar la red aprendiz de dos formas:
   1. Entrenar red aprendiz como modelo base igual a la red tutora:

        ```bash
        python trainer.py --dataset cifar100 --architecture ResNet18 --epochs 600 --batch-size 128
        ```

   2. Entrenar red aprendiz con destilación de conocimiento, para esto, primero debe guardar el modelo onnx de la red tutora en la carpeta de pretrained_models (puede usar metrics.py y mover el archivo onnx desde el checkpoint del experimento a la carpeta pretrained_models). Luego, puede entrenar la red aprendiz de la siguiente manera:

        ```bash
        python destiller.py --dataset cifar100 --student_architecture ResNet18 --epochs 600 --batch-size 128 --distillation soft_targets --teacher_architecture ResNet101
        ```

        Igual que antes, puedes utilizar ```--show_versions``` para ver si existen más modelos entrenados bajo la misma configuración de dataset y arquitectura. Con ```--version {version}``` puedes continuar el entrenamiento de un modelo existente entregando su respectiva versión.
