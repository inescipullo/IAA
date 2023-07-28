# Trabajo Práctico 4 - K-vecinos

Vamos a usar las implementaciones de Nearest Neighbours de la librería Sklearn, con todas las opciones por defecto salvo las que se discutan en cada punto. Como siempre, hay que implementar código que nos permita seleccionar el k óptimo sobre un conjunto de validación, nos devuelva el clasificador o regresor entrenado, y nos permita hacer curvas de entrenamiento, validación y test en función del número de vecinos.


## Para entregar:

### a) 
Resuelva el problema de las espirales-anidadas usando k-nn. Utilice el datasets de "espirales con ruido" de esta página. Hay una versión "original" y otra que tiene agregadas dos variables que contienen ruido uniforme. Realice gráficas de las predicciones sobre el conjunto de test, y gráficas de errores vs número de vecinos. Compare el resultado con el obtenido con árboles de decisión, los dos métodos sobre las dos versiones del dataset.

### b) Dimensionalidad: 
Repita el punto 4 del Práctico 1, usando k-nn. Utilice dos valores de `k`: el número de vecinos que se obtiene como mínimo de validación, y 1 vecino. Genere una gráfica incluyendo también los resultados de redes, árboles y naive-Bayes con Gaussianas.

### c) Distancias pesadas: 
Aplique la versión con votación pesada por distancia (inversa) con dos valores de k: `k=50` o un valor de `k` optimizado con un conjunto de validación. Aplique las dos versiones al problema de la dimensionalidad. Tomando `d=32`, genere las curvas de error de entrenamiento, validación y test y coméntelas. Genere también nuevas curvas de error vs dimensiones para los dos clasificadores (`k` fijo y `k` óptimo) y compárelas con las del clasificador con peso uniforme del punto anterior (también con `k=1` y con `k` óptimo).

### d) Regresión con k_nn: 
Prepare código para usar k-nn en regresión con pesos uniformes y con pesado inverso a la distancia, como se describió en la teoría. Aplique las dos versiones al problema de Sunspots (SSP) y al problema de Ikeda, y compare los resultados con los obtenidos con ANN.

### e) 
Otra variante de k-nn que se suele utilizar es usar en la votación a todos los patrones que estén a una distancia menor a un dado valor `D` del patrón que se quiere clasificar, en lugar de usar un número fijo `k`. El único parámetro del algoritmo, ahora, es la distancia máxima `D`, la que se optimiza utilizando un conjunto de validación. 
Implemente código que optimice el valor de `D` de forma razonable en función de los datos de entrenamiento, utilizando un conjunto de validación si es necesario. Lo más interesante de este punto es discutir y definir cómo encontrar el valor de D, sin googlearlo.
Aplique su método al problema de Dimensionalidad, y compare el resultado con los obtenidos en el punto c).

## OPCIONAL:

### f) Opcional (2 puntos): Recomendación con KNN.
Contamos con dos planillas de datos. "pelis" tiene una lista de 25 películas, cada una evaluada en 6 atributos distintos en una escala de 1 a 5 por un grupo de expertos en cine (puede haber errores como en todo dataset real). "usuarios" tiene la opinión de un grupo de usuarios (en columnas U1 a U8) sobre algunas de las 25 películas que tenemos disponibles, usando una escala de 1 a 5. En las columnas U9 y U10 tenemos 2 usuarios nuevos que opinaron sobre algunas pelis y a los cuales les tenemos que recomendar las 2 películas más apropiadas para ellos.
Desarrolle una solución propia para un sistema de recomendación simple que use la idea de distancia y vecinos. Explique la idea del sistema que implementó. 

--

Entregue un notebook con todos los códigos, figuras y comentarios.