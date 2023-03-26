import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MultidimensionalGaussianDistribution:
	"""Generate multidimensional points sample according to the Gaussian Distribution.

	Values generated equally in two different categories.
	Sample A:
	0: gaussian distribution with center (-1, -1, ..., -1)
	1: gaussian distribution with center (1, 1, ..., 1)
	Sample B:
	0: gaussian distribution with center (-1, 0, ..., 0)
	1: gaussian distribution with center (1, 0, ..., 0)
	"""

	def __init__(self, dimension: int, sample_size: int, C: float):
		self.dimension = dimension
		self.sample_size = sample_size
		self.C = C

	def generate_sample_a(self):
		center_0 = np.ones(self.dimension) * -1
		center_1 = np.ones(self.dimension)
		variance = (self.C * np.sqrt(self.dimension))**2

		sample_0 = self._sample_with_center_and_variance(center_0, variance, 0)
		sample_1 = self._sample_with_center_and_variance(center_1, variance, 1)

		columns = [f"Dim{i+1}" for i in range(self.dimension)]
		columns.append("Class")

		dataframe = pd.DataFrame(sample_0+sample_1, columns=columns)
		return dataframe

	def generate_sample_b(self):
		center_0 = np.append(np.ones(1), np.zeros(self.dimension-1))
		center_1 = np.append(np.ones(1)*-1, np.zeros(self.dimension-1))
		variance = self.C**2

		sample_0 = self._sample_with_center_and_variance(center_0, variance, 0)
		sample_1 = self._sample_with_center_and_variance(center_1, variance, 1)

		columns = [f"Dim{i+1}" for i in range(self.dimension)]
		columns.append("Class")

		dataframe = pd.DataFrame(sample_0+sample_1, columns=columns)
		return dataframe


	def _sample_with_center_and_variance(self, center, variance, category):
		covariance_matrix = np.identity(self.dimension) * (variance**2)
		
		sample = np.random.multivariate_normal(mean=center, cov=covariance_matrix, size=self.sample_size//2)

		sample_with_class = [point + [category] for point in sample.tolist()]
		return sample_with_class


def plot(dataframe):
	colors = dataframe["Class"].map({0: 'b', 1: 'r'})
	dataframe.plot(x="Dim1", y="Dim2", kind='scatter', c=colors)
	plt.show()


def testMultidimensionalGaussianDistribution(d: int, n: int , C: float):
  print(f"Test d={d}, n={n}, C={C}")
  data_gen = MultidimensionalGaussianDistribution(dimension=d, sample_size=n, C=C)
  print("Sample A:")
  df_a = data_gen.generate_sample_a()
  if d == 2:
    plot(df_a)
  else:
    print("Media: \n", df_a.groupby(["Class"]).mean())
    print("Desvio Estandar: \n", df_a.groupby(["Class"]).std())
  
  df_b = data_gen.generate_sample_b()
  print("Sample B:")
  if d == 2:
    plot(df_b)
  else:
    print("Media: \n", df_b.groupby(["Class"]).mean())
    print("Desvio Estandar: \n", df_b.groupby(["Class"]).std())


testMultidimensionalGaussianDistribution(d=2, n=2000, C=0.75)
testMultidimensionalGaussianDistribution(d=4, n=5000, C=2.00)


class Circular2DUniformDistribution:
	def __init__(self, sample_size: int):
		self.sample_size = sample_size
		self.dimension = 2
		self.radius = 1

	def generate_sample(self):
		count = {"0": 0, "1": 0}
		cat_size = self.sample_size // 2
		x_values = []
		y_values = []
		categories = []

		while count["0"] < cat_size and count["1"] < cat_size:
			theta = np.random.uniform(0, 2*np.pi, 1).tolist()[0]
			radius = np.random.uniform(0, self.radius, 1).tolist()[0]
			category = self._get_category(theta, radius)

			if count[f"{category}"] < cat_size:
				count[f"{category}"] += 1
				x_values.append(radius * np.cos(theta))
				y_values.append(radius * np.sin(theta))
				categories.append(category)


		return pd.DataFrame({"X": x_values, "Y": y_values, "Class": categories})

	def _get_category(self, theta, radius):
		r1 = theta / (4 * np.pi)
		r2 = (theta * np.pi) / (4 * np.pi)
		
		if r1 < radius and radius < r2:
			return 0

		r2p = (theta * np.pi + 2 * np.pi) / (4 * np.pi)
		if radius > r2p:
			return 0
		return 1


def plot2(dataframe):
	colors = dataframe["Class"].map({0: 'b', 1: 'r'})
	dataframe.plot(x="X", y="Y", kind='scatter', c=colors)
	plt.show()


def testCircular2DUniformDistribution(n: int):
  data_gen = Circular2DUniformDistribution(n)
  df = data_gen.generate_sample()
  plot2(df)


testCircular2DUniformDistribution(2000)