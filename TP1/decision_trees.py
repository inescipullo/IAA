from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score


class Ej2Sobreajuste:
  def __init__(self):
    self.df_generator = MultidimensionalGaussianDistribution

  def for_sample_sizes(self, ns):
    self._for_sample_a(ns)
    self._for_sample_b(ns)

  def _for_sample_a(self, ns):
    test_data, test_target = self._get_data_and_target(self.test_df_a)
    test_error = {}
    train_error = {}
    tree_sizes = {}
    for n in ns:
      test_error[f"{n}"] = []
      train_error[f"{n}"] = []
      tree_sizes[f"{n}"] = []
      for i in range(20):
        df = self.df_generator.generate_sample_a(n)
        data, target = self._get_data_and_target(df)
        trained_decision_tree = self._train_decision_tree(data, target)
        test_df_with_prediction = self._test_decision_tree(trained_decision_tree, self.test_df_a)
        train_df_with_prediction = self._test_decision_tree(trained_decision_tree, df)
        if i == 0:
          print(f"Tree prediction of Sample A trained with dataframe size {n}.\n")
          plot(test_df_with_prediction)
        test_error[f"{n}"].append(1 - accuracy_score(test_target, test_df_with_prediction["Class"]))
        train_error[f"{n}"].append(1 - accuracy_score(target, train_df_with_prediction["Class"]))
        tree_sizes[f"{n}"].append(trained_decision_tree.tree_.node_count)
    self._plot_errors(test_error, train_error)
    self._plot_tree_sizes(tree_sizes)

  def _for_sample_b(self, ns):
    test_data, test_target = self._get_data_and_target(self.test_df_a)
    test_error = {}
    train_error = {}
    tree_sizes = {}
    for n in ns:
      test_error[f"{n}"] = []
      train_error[f"{n}"] = []
      tree_sizes[f"{n}"] = []
      for i in range(20):
        df = self.df_generator.generate_sample_b(n)
        data, target = self._get_data_and_target(df)
        trained_decision_tree = self._train_decision_tree(data, target)
        test_df_with_prediction = self._test_decision_tree(trained_decision_tree, self.test_df_b)
        train_df_with_prediction = self._test_decision_tree(trained_decision_tree, df)
        if i == 0:
          print(f"Tree prediction of Sample B trained with dataframe size {n}.\n")
          plot(test_df_with_prediction)
        test_error[f"{n}"].append(1 - accuracy_score(test_target, test_df_with_prediction["Class"]))
        train_error[f"{n}"].append(1 - accuracy_score(target, train_df_with_prediction["Class"]))
        tree_sizes[f"{n}"].append(trained_decision_tree.tree_.node_count)
    self._plot_errors(test_error, train_error)
    self._plot_tree_sizes(tree_sizes)

  def _get_data_and_target(self, dataframe):
    """Given a dataframe generated with MultidimensionalGaussianDistribution with dimension=2, splits it into data and target."""
    data = list(map(list, zip(dataframe["Dim1"], dataframe["Dim2"])))
    target = dataframe["Class"].values.tolist()
    return data, target

  def _train_decision_tree(self, data, target):
    decision_tree = DecisionTreeClassifier(criterion="entropy", min_impurity_decrease=0.005, random_state=0, min_samples_leaf=5)
    return decision_tree.fit(data, target)
  
  def _test_decision_tree(self, decision_tree, test_df):
    test_data, _ = self._get_data_and_target(test_df)
    prediction = decision_tree.predict(test_data)
    for i in range(len(prediction)):
        test_data[i].append(prediction[i])
    return pd.DataFrame(test_data, columns=["Dim1", "Dim2", "Class"])

  def _plot_errors(self, test_errors, train_errors):
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for n in test_errors.keys():
      x1.append(int(n))
      y1.append(np.mean(test_errors[n]))
      x2.append(int(n))
      y2.append(np.mean(train_errors[n]))
    plt.plot(x1, y1, linestyle='dashed', marker='s', label="Test error")
    plt.plot(x2, y2, linestyle='dashed', marker='s', label="Train error")
    plt.legend(loc="upper right")
    plt.title("Promedio del error sobre conjuntos de test y de entreno s/ N")
    # plt.xscale('log')
    plt.show()

  def _plot_tree_sizes(self, tree_sizes):
    x = []
    y = []
    for n in tree_sizes.keys():
      x.append(int(n))
      y.append(np.mean(tree_sizes[n]))
    plt.plot(x, y, linestyle='dashed', marker='s')
    plt.title("Promedio de cantidad de nodos del arbol s/ N")
    # plt.xscale('log')
    plt.show()



  @classmethod
  def varying_sample_size(cls, sample_sizes: list[int]):



  @classmethod
  def varying_dimension(cls, dimensions: list[int]):



  @classmethod
  def varying_C(cls, Cs: list[float]):