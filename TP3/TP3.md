# Trabajo Práctico 3 - Bayes

Vamos a usar la implementación de Naive Bayes que viene en la librería Sklearn, tanto con Gaussianas como con histogramas. Ejemplo:

```
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
```

## Para entregar:

### 1. Dimensionalidad: 
Repita el punto de dimensionalidad del Práctico 1, usando el Clasificador Naive-Bayes con Gaussianas. Genere una gráfica incluyendo también los resultados de redes y árboles.

### 2. Límites del clasificador: 
Resuelva el problema de dos-elipses utilizando el Clasificador Naive-Bayes con Gaussianas. Realice una gráfica de la predicción sobre el conjunto de test. Compare el resultado con el obtenido con redes. Resuelva el problema de las espirales-anidadas, y también realice la gráfica y compare con el resultado de redes. Explique por qué se obtienen esos resultados. Use la misma cantidad de datos de entrenamiento y test que en los problemas del práctico de redes.

### 3.
Para el algoritmo Naive-Bayes con histogramas vamos a usar la implementacion de CategoricalNB de Sklearn. Para poder usarla primero tenemos que convertir nuestras variables continuas en categóricas, o sea en un histograma, usando un discretizador. Desarrolle una función que entrene un clasificador Naive-Bayes con histogramas, usando un conjunto de validación para determinar el número óptimo de bins del histograma. Las opciones para el clasificador y el discretizador son las siguientes:

```
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import CategoricalNB
bins=5
discretizador = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
#
#código propio
#
clf = CategoricalNB(min_categories=bins)
clf.fit(X_discreto,y_train)
#
#más código propio
```

La función es similar a la que hicieron para entrenar redes. Tiene que recibir los conjuntos de entrenamiento, validación y test, el discretizador y el clasificador, y los valores a probar del número de bins, y tiene que devolver los errores para hacer curvas de error versus número de bins, y el discretizador y el clasificador óptimo entrenados.


### 4.
Usando la función implementada repita el trabajo sobre los problemas de dos-elipses y de espirales-anidadas, usando conjuntos de validación y cantidad de bines adecuados. Grafique el error de clasificación en ajuste, validación y test en función de dicho número de bins (hay sobreajuste?). Con el clasificador óptimo grafique las clasificaciones en test. Compárelos con los resultados del punto 2.

### 5. 
Uno de los usos más comunes del algoritmo de Naive-Bayes es para la clasificación de texto. En este punto vamos a usar un dataset muy conocido:

```
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
X, y = fetch_20newsgroups(subset="train",return_X_y=True, remove=["headers"])
X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=0.25, random_state=1)
X_test, y_test = fetch_20newsgroups(subset="test",return_X_y=True,remove=["headers"])
```

Y lo vamos a "vectorizar" contando las veces que ocurre cada palabra de un diccionario dado en cada texto:

```
from sklearn.feature_extraction.text import CountVectorizer
largo_diccionario=4000
vec = CountVectorizer(stop_words='english',max_features=largo_diccionario)
Xvec_train = vec.fit_transform(X_train).toarray()
Xvec_test = vec.transform(X_test).toarray()
```

y lo vamos a clasificar con el clasificador discreto multinomial (pag. 180 libro de Mitchell):

```
from sklearn.naive_bayes import MultinomialNB
alfa=1
clf = MultinomialNB(alpha=alfa)
```

Escriba código para entrenar este clasificador sobre los datos correspondientes y evaluarlo sobre los conjuntos de validación y test. Evalúe distintas combinaciones del largo del diccionario de palabras (entre 1000 y 4000 por lo menos) y del parámetro alfa (órdenes de magnitud, de 1 a 0.0001), buscando el mínimo en validación. Hay sobreajuste? El comportamiento en validación es representativo del conjunto de test? Calcule una matriz de confusión para el conjunto de test. Hay alguna particularidad que merece atención especial?

## OPCIONAL:

### 6. Opcional (2 puntos): 
Usando como base la implementación de CategoricalNB de Sklearn (contenida en el archivo adjunto `naive_bayes.py`), implemente una versión del clasificador Bayesiano con histogramas donde los atributos no sean considerados independientes (Not-Naive-Bayes). Es aceptable hacer un programa limitado a datasets de 2 dimensiones. Aplíquelo al problema de las espirales-anidadas, y compárelo con los resultados anteriores. Discuta los problemas de implementación y de aplicación práctica que tendría este algoritmo en datasets de muchas dimensiones.

--

Entregue un notebook con todo el código y los comentarios pedidos.