# Trabajo Práctico 0: Generar datos en Python

## Prepare funciones en Python que generen dataframes panda ( de longitud dada n ) de acuerdo a las siguientes descripciones:

### a) 
Los datos tienen d inputs, todos valores reales, correspondientes a la posición del punto en un espacio d-dimensional. El output es binario, y corresponde a la clase a la que pertenece el ejemplo. La clase 1 corresponde a puntos generados al azar, provenientes de una distribución normal, con centro en el ( 1, 1, 1, .... , 1 ) y matriz de covarianza diagonal, con desviación estándar igual a `C * SQRT(d)`. La clase 0 tiene la misma distribución, pero centrada en el ( -1, -1, -1, .... , -1 ). Se puede encontrar información sobre Gaussianas multidimensionales y el caso especial de una matriz diagonal en http://cs229.stanford.edu/section/gaussians.pdf (secciones 1 y 3). Los parámetros que se deben ingresar a la función son d y n (enteros) y C (real). De los n puntos generados, n/2 deben pertenecer a cada clase.

### b) 
Igual al punto anterior, pero las distribuciones tienen centro en el ( 1, 0, 0, .... , 0 ) y en el ( -1, 0, 0, .... , 0 ), respectivamente y la desviación estandar es igual a `C` independientemente de d.

### c) 
Espirales anidadas: Los datos tienen 2 inputs, x e y, que corresponden a puntos generados al azar con una distribución UNIFORME (en dicho sistema de referencia x-y) dentro de un circulo de radio 1. El output es binario, correspondiendo la clase 0 a los puntos que se encuentran entre las curvas `ro = theta/4pi` y `ro = (theta + pi)/4pi` (en polares) y la clase 1 al resto. De los n puntos generados, n/2 deben pertenecer a cada clase.


--

Para verificar los problemas a) y b), genere conjuntos con d=2, n=200 y C=0.75, y grafíquelos. También genere conjuntos con d=4, n=5000 y C=2.00, y verifique en el código que las medias y desviaciones estándar sean correctas.

Para el problema c), genere un gráfico con n=2000 y compárelo con el que está arriba.

Todos los trabajos prácticos se entregan como un notebook Jupiter en un archivo .ipynb, con el código que genere lo solicitado y cualquier comentario que quieran agregar.