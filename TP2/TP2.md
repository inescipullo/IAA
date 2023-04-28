# Trabajo Práctico 2 - ANN (Artificial Neural Networks)

## Trabajo previo:

Vamos a usar la implementación de los MLP estándar que viene en la librería Sklearn. Tanto la versión para problemas de clasificación (MLPClassifier) como para problemas de regresión (MLPRegressor). Este es un ejemplo de las redes que vamos a usar:

```
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

#defino parámetros de la red
epocas_por_entrenamiento=25    #numero de epocas que entrena cada vez
eta=0.01              #learning rate
alfa=0.9               #momentum
N2=60                 #neuronas en la capa oculta

#defino MLP para regresión
regr = MLPRegressor(hidden_layer_sizes=(N2,), activation='logistic', solver='sgd', alpha=0.0, batch_size=1, learning_rate='constant', learning_rate_init=eta,momentum=alfa,nesterovs_momentum=False,tol=0.0,warm_start=True,max_iter=epocas_por_entrenamiento)
#defino MLP para clasificación
clasif = MLPClassifier(hidden_layer_sizes=(N2,), activation='logistic', solver='sgd', alpha=0.0, batch_size=1, learning_rate='constant', learning_rate_init=eta,momentum=alfa,nesterovs_momentum=False,tol=0.0,warm_start=True,max_iter=epocas_por_entrenamiento)
print(regr)
```

Para los ejercicios del práctico hay que desarrollar una función que entrene la red un número de veces y que mida y devuelva los errores de la misma, para poder hacer curvas de entrenamiento. La función tiene que ser algo así:

```
#función que entrena una red ya definida previamente "evaluaciones" veces, cada vez entrenando un número de épocas elegido al crear la red y midiendo el error en train, validación y test al terminar ese paso de entrenamiento. 
#Guarda y devuelve la red en el paso de evaluación que da el mínimo error de validación
#entradas: la red, las veces que evalua, los datos de entrenamiento y sus respuestas, de validacion y sus respuestas, de test y sus respuestas
#salidas: la red entrenada en el mínimo de validación, los errores de train, validación y test medidos en cada evaluación
def entrenar_red(red, evaluaciones, X_train, y_train, X_val, y_val, X_test, y_test):
     #mi código
     #red.fit(X_train, y_train)
     #más código
     return best_red, error_train, error_val, error_test
```

Pueden hacer una función para problemas de regresión y una para clasificación, o todo en una. Pueden agregar lo que quieran, ese diseño es para darles una idea de lo que hay que hacer. Para hacer una copia de una red ya creada, para mantener la mejor que encontraron hasta el momento, una forma es con la función deepcopy() de la librería copy. La medida de error que vamos a usar en regresión es error cuadrático medio (`sk.metrics.mean_squared_error()`) y para clasificación es el error de clasificación (`sk.metrics.zero_one_loss()`)

La función que crearon la vamos a usar para entrenar y evaluar redes como la que definí más arriba, graficando curvas de error y las predicciones, con código como éste:

```
#epocas es la cantidad de veces que entreno la red y mido los errores
regr, e_train, e_val, e_test = entrenar_red(regr, epocas, X_train, y_train, X_val, y_val, X_test, y_test)
import matplotlib.pyplot as plt
plt.plot(range(epocas),e_train,label="train",linestyle=":")
plt.plot(range(epocas),e_val,label="validacion",linestyle="-.")
plt.plot(range(epocas),e_test,label="test",linestyle="-")
plt.legend()
plt.show()
```
Warning: el entrenamiento de las redes es computacionalmente pesado. Algunos de los entrenamientos que se plantean en el trabajo llevan varios minutos, dependiendo de la velocidad del procesador. El tiempo máximo en Google Colab fue de alrededor de 10 minutos. Tengan eso en cuenta para hacer el trabajo.

## Para Entregar:

### 1. Capacidad de modelado: 
Entrene redes neuronales para resolver el problema de clasificación de las espirales anidadas que creamos en el TP 0. Use un número creciente de neuronas en la capa intermedia: 2, 10, 20, 40. Valores posibles para los demás parámetros de entrenamiento: learning rate 0.1, momentum 0.9, 600 datos para ajustar los modelos (20% de ese conjunto separarlo al azar para conjunto de validación), 2000 para testear, 1000 evaluaciones del entrenamiento, cada una de 20 épocas. Para cada uno de los cuatro modelos obtenidos, graficar en el plano xy las clasificaciones sobre el conjunto de test. Comentar.

### 2. Mínimos locales: 
Baje el dataset dos-elipses de la descargas. Realice varios entrenamientos con los siguientes parámetros: 6 neuronas en la capa intermedia, 500 patrones en el training set, de los cuales 400 se usan para entrenar y 100 para validar el modelo (sacados del `.data`), 2000 patrones en el test set (del `.test`), 300 evaluaciones del entrenamiento, cada una de 50 épocas. 

Pruebe distintos valores de momentum y learning-rate (valores usuales son 0, 0.5, 0.9 para el momentum y 0.1, 0.01, 0.001 para el learning-rate, pero no hay por qué limitarse a esos valores), para tratar de encontrar el mejor mínimo posible de la función error. El valor que vamos a usar es el promedio de 10 entrenamientos iguales, dado que los entrenamientos incorporan el azar. Como guía, con los parámetros dados, hay soluciones entre 5% y 6% de error en test, y tal vez mejores. Confeccione una tabla con los valores usados para los parámetros y el resultado en test obtenido (la media de las 10 ejecuciones). Haga una gráfica de mse de train, validación y test en función del número de épocas para los valores seleccionados (los mejores valores de eta y alfa). 

### 3. Regularización: 
Baje el dataset Ikeda. Realice varios entrenamientos usando el 95% del archivo `.data` para entrenar, y el resto para validar. Realice otros entrenamientos cambiando la relación a 75%-25%, y a 50%-50%. En cada caso seleccione un resultado que considere adecuado, y genere gráficas del mse en train, validación y test. Comente sobre los resultados. 
Los otros parámetros para el entrenamiento son: learning rate 0.01, momentum 0.9, 2000 datos para testear, 400 evaluaciones del entrenamiento, cada una de 50 épocas, 30 neuronas en la capa oculta.

### 4. Regularización: 
Vamos a usar regularización por penalización, el weight-decay. Hay que tener cuidado con los nombres de los parámetros en este caso. El parámetro que nosotros llamamos `gamma` en la teoría corresponde en MLP de sklearn al parámetro `alpha`, mientras que nosotros usamos alfa para el momentum en general. Para activarlo tenemos que usar:

```
gamma=0.00001
regr = MLPRegressor(hidden_layer_sizes=(N2,), activation='logistic', solver='sgd', alpha=gamma, batch_size=1, learning_rate='constant', learning_rate_init=eta,momentum=alfa,nesterovs_momentum=False,tol=0.0,warm_start=True,max_iter=epocas_por_entrenamiento)
```

En este tipo de regularización no se usa un conjunto de validación, asi que hay que modificar la función que crearon para evaluar el entrenamiento de las redes, para que en lugar del error sobre el conjunto de validación, nos devuelva la suma de los valores absolutos o de los valores al cuadrado de todos los pesos de la red en la epoca correspondiente, y todo el resto igual que antes.

Una vez implementado, aplíquelo al dataset Sunspots (ssp). Busque el valor de `gamma` adecuado para conseguir un equilibrio entre evitar el sobreajuste y hacer el modelo demasiado rígido (el valor de `gamma` se debe variar en varios órdenes de magnitud, por ejemplo empezar en 10^-6 e ir hasta 10^0 (1) de a un orden cada vez). En este caso todos los registros del archivo `.data` se deben usar para el entrenamiento, ya que la regularización se realiza con la penalización a los pesos grandes. 
Los otros parámetros se pueden tomar: learning rate 0.05, momentum 0.3, 4000 evaluaciones del entrenamiento, cada una de 20 épocas, 6 neuronas en la capa intermedia.

Entregue curvas de los tres errores (entrenamiento y test en una figura, penalización en otra figura) para el valor de `gamma` elegido, y para algún otro valor en que haya observado sobreajuste. Comente los resultados.

### 5. Dimensionalidad: 
Repita el punto 4 del Práctico 1, usando ahora redes con 6 unidades en la capa intermedia. Los otros parámetros hay que setearlos adecuadamente, usando como guía los casos anteriores. Genere una gráfica que incluya los resultados de redes y árboles.

## OPCIONALES:

### 6. Multiclase: 
Busque en los datasets de sklearn los archivos del problema iris y entrene una red sobre ellos. Tome un tercio de los datos como test, y los dos tercios restantes para entrenar y validar. Ajusto los parámetros de la red de manera conveniente. Realice curvas de entrenamiento, validación y test.

Baje el dataset Faces. Entrene una red neuronal para resolver dicho problema, eligiendo adecuadamente todos los parámetros (hay que re-escalar los datos al rango [0,1] antes de usarlos). Muestre curvas de entrenamiento para asegurar la convergencia. Compare los resultados con los citados en Mitchell (pag. 112). 

### 7. Minibatch: 
La implementaciín de sklearn permite usar minibaths cambiando el parámetro batch_size. Realice una comparación de las curvas de aprendizaje para el problema de Sunspots, utilizando batches de longitud 1 y otros dos valores (3 curvas). Comente los resultados. Parametros del entrenamiento: 20% de validación, learning rate 0.05, momentum 0.3, 2000 evaluaciones del entrenamiento, cada una de 200 épocas, 6 neuronas en la capa intermedia. Los valores del minibatch se deben elegir para que en cada etapa de entrenamiento haya varias actualizaciones del gradiente, o sea que no debería ser mayor a 40.

--

En todos los puntos del práctico discutir los resultados que considere conveniente. Entregar como siempre un notebook con todo el código y los comentarios.