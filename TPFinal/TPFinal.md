# Trabajo Final - Redes convolucionales

Para este trabajo vamos a utilizar el dataset `CIFAR-10`, que tiene imágenes de 10 clases distintas de objetos. 
Las implementaciones de redes profundas que usaremos serán de TensorFlow con keras. La librería es fácil de instalar y usar (https://www.tensorflow.org/install?hl=es-419). 
Un ejemplo inicial se puede ver en: https://www.tensorflow.org/tutorials/keras/classification
En el ejemplo de redes convolucionales de la librería se puede ver como obtener el dataset e implementar una red convolucional sobre el mismo:
https://www.tensorflow.org/tutorials/images/cnn?hl=en
Entrenar una red como las que vamos a utilizar en un i7 toma una media hora de tiempo. Si tienen acceso a una GPU el tiempo se reduce por un factor 10 al menos. Pueden intentar usar Google Colab, y suelen dar acceso gratuito a las GPUs de google por unas horas al día.

## Para Entregar:

### 1. Resolver un problema de visión.  
Entrene una red convolucional como la mostrada en el tutorial anterior sobre el dataset CIFAR-10. La red tiene que tener las siguientes capas:
Conv, filtro 3x3, 32 filtros, con padding para que conserve el tamaño de 32x32, Relu.
Maxpooling 2x2 (es el valor por defecto)
Conv, filtro 3x3, 64 filtros, con padding, Relu.
Conv, filtro 3x3, 64 filtros, con padding, Relu.
Maxpooling 2x2
Conv, filtro 3x3, 64 filtros, con padding, Relu.
Conv, filtro 3x3, 64 filtros, con padding, Relu.
Densa, 64
Densa, 128
Densa, 128
Densa, 10 (salida)

Usen como base el ejemplo del tutorial (ver la capa de aplanado al pasar de convolución a densa!), con los mismos parámetros para el entrenamiento. Obtengan las curvas de errror en train y validación, y el valor de error en test del modelo final. 

### 2. Dropout.
Agregue regularización por Dropout en dos formas, primero con una capa Dropout luego de la última capa convolucional, y después cree otro modelo agregando a la anterior una segunda capa de dropout luego de la primer capa densa. Pruebe dos valores de `p` en cada caso, 0.2 y 0.5 (al agregar dropout se recomienda duplicar las épocas de entrenamiento porque el azar dificulta el aprendizaje). Obtenga las curvas de entrenamiento y validación, y el error de test. Analice los resultados.

### 3. Data augmentation.

Siguiendo otro ejemplo de keras, definamos una capa de data augmentation así:

```
img_height = 32
img_width = 32
data_augmentation = keras.Sequential(
    [
          layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
          layers.RandomRotation(0.1),
          layers.RandomZoom(0.1),
     ]
)
```

Agréguenla como primer capa del modelo anterior (con el dropout óptimo). Reentrenen el modelo, con las curvas y el error de test (de nuevo, duplicando las épocas del original). Hay mejoras? Menos sobreajuste?

## Opcional:

### 4. Opcional: Diseñar una red (2 puntos)

Mejore el diseño de la CNN como considere adecuado para obtener una mejora importante en el error (al menos 3%) sobre los resultados anteriores. Trate de justificar por qué su diseño mejora lo anterior.

### 5. Opcional: Mejorar una capa (1 punto)

Mejore la capa de Data augmentation, incorporando al menos alguna otra operación. Reentrene, compare, analice los resultados.

--

Entregar un notebook con código, figuras y comentarios adecuados.