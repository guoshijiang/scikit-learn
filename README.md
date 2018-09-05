# 带你一起学习scikit-learn

# 第一部分：快速开始scikit-learn

## 第一章：安装scikit-learn

### 一.安装最新的release版

#### 1.scikit-learn必备条件

* Python (>= 2.7 or >= 3.3),
* NumPy (>= 1.8.2),
* SciPy (>= 0.13.3).

如果你已经在工作环境中安装了numpy和scipy，你可以通过pip或者conda安装scikit-learn

  pip install -U scikit-learn

或者

  conda install scikit-learn

关于Python和Pip工具的安装，请参照其他网络上的教程。


## 第二章：用scikit-learn介绍机器学习

在这个部分，我们通过scikit-learn给出几个简单的例子来介绍机器学习的词汇。

### 一.机器学习：问题设定

通常，学习问题考虑一组n个数据样本，然后尝试预测未知数据的属性。 如果每个样本多于一个数字，例如，多维实体（也称为多变量数据），则称其具有多个属性或特征。

我们可以将几个大类的学习问题分开：

* 监督学习：其中数据带有我们想要预测的其他属性。这个问题可以是：

 * 分类：样本属于两个或更多类，我们希望从已标记的数据中学习如何预测未标记数据的类别。 分类问题的一个例子是手写数字识别示例，其中目的是将每个输入矢量分配给有限数量的离散类别之一。 考虑分类的另一种方式是作为一种离散（相对于连续）形式的监督学习，其中一个类别的数量有限，并且对于所提供的n个样本中的每一个，一个是尝试用正确的类别或类别标记它们。
 
 * 回归：如果所需的输出由一个或多个连续变量组成，则该任务称为回归。 回归问题的一个例子是预测鲑鱼的长度作为其年龄和体重的函数。

* 无监督学习，其中训练数据由一组输入向量x组成，没有任何相应的目标值。 这些问题的目标可能是发现数据中的类似示例组，称为聚类，或确定输入空间内的数据分布，称为密度估计，或从高维投影数据 为了可视化的目的，将空间缩小到两维或三维（单击此处转到Scikit-Learn无监督学习页面）。

训练数据集合测试数据集

机器学习是关于学习数据集的一些属性并将它们应用于新数据。 这就是为什么在机器学习中评估算法的常见做法是将手头的数据分成两组，一组称为训练集，我们在其上学习数据属性，另一组称为测试集，我们在其上测试这些属性。

### 二.加载一个数据集

scikit-learn带有一些标准数据集，例如用于分类的虹膜和数字数据集以及用于回归的波士顿房价数据集。

在下文中，我们从shell启动Python解释器，然后加载iris和digits数据集。 我们的符号约定是$表示shell提示符，而>>>表示Python解释器提示符：

    $ python
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> digits = datasets.load_digits()

数据集是一个类似字典的对象，它包含所有数据和一些有关数据的元数据。 此数据存储在.data成员中，该成员是n_samples，n_features数组。 在监督问题的情况下，一个或多个响应变量存储在.target成员中。

例如，在数字数据集的情况下，digits.data可以访问可用于对数字样本进行分类的功能：

digits.target给出数字数据集的基本事实，即我们试图学习的每个数字图像对应的数字：

以下是执行效果图：

加载数据集图1： 
    ![加载数据集图1： 
](https://github.com/guoshijiang/scikit-learn/blob/master/img/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20180904173143.png "加载数据集图1：")

加载数据集图2： 
    ![加载数据集图2： 
](https://github.com/guoshijiang/scikit-learn/blob/master/img/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20180904173716.png "加载数据集图2：")

* 数据数组的维度

尽管原始数据可能具有不同的形状，但数据始终是2D数组，形状（n_samples，n_features）。 在数字的情况下，每个原始样本是形状（8,8）的图像，并且可以使用以下方式访问：

    >>> digits.images[0]
    array([[  0.,   0.,   5.,  13.,   9.,   1.,   0.,   0.],
           [  0.,   0.,  13.,  15.,  10.,  15.,   5.,   0.],
           [  0.,   3.,  15.,   2.,   0.,  11.,   8.,   0.],
           [  0.,   4.,  12.,   0.,   0.,   8.,   8.,   0.],
           [  0.,   5.,   8.,   0.,   0.,   9.,   8.,   0.],
           [  0.,   4.,  11.,   0.,   1.,  12.,   7.,   0.],
           [  0.,   2.,  14.,   5.,  10.,  12.,   0.,   0.],
           [  0.,   0.,   6.,  13.,  10.,   0.,   0.,   0.]])

这个数据集的简单示例说明了如何从原始问题开始，可以在scikit-learn中对数据进行整形。

### 学习和预测

在数字数据集的情况下，任务是在给定图像的情况下预测它代表的数字。 我们给出了10个可能类别（数字0到9）中的每一个的样本，我们在其上拟合估计器，以便能够预测看不见的样本所属的类。

在scikit-learn中，用于分类的估计器是一个Python对象，它实现了适合（X，y）和预测（T）的方法。

估计器的一个示例是实现支持向量分类的类sklearn.svm.SVC。 估算器的构造函数将模型的参数作为参数，但暂时，我们将估算器视为黑盒子：

* 选择模型的参数

在此示例中，我们手动设置gamma的值。 通过使用网格搜索和交叉验证等工具，可以自动为参数找到合适的值。


我们称之为估算器实例clf，因为它是一个分类器。 它现在必须适合模型，也就是说，它必须从模型中学习。 这是通过将我们的训练集传递给fit方法来完成的。 作为训练集，让我们使用除最后一个之外的数据集的所有图像。 我们使用[：-1] Python语法选择此训练集，该语法生成一个新数组，其中包含除digit.data的最后一个条目以外的所有数组：

    >>> clf.fit(digits.data[:-1], digits.target[:-1])  
    SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

现在您可以预测新值，特别是，我们可以向分类器询问数字数据集中我们最后一个图像的数字，我们还没有用它来训练分类器：

    >>> clf.predict(digits.data[-1:])
    array([8])

相应的图像如下：

训练数据图1： 
    ![训练数据图1： 
](https://github.com/guoshijiang/scikit-learn/blob/master/img/1536116165(1).png "训练数据图1：")

如您所见，这是一项具有挑战性的任务：图像分辨率较差。 你同意分类器吗？

可以使用此分类问题的完整示例作为您可以运行和学习的示例：识别手写数字。

### 模型持久性

可以使用Python的内置持久性模型（即pickle）在scikit中保存模型：

    >>> from sklearn import svm
    >>> from sklearn import datasets
    >>> clf = svm.SVC()
    >>> iris = datasets.load_iris()
    >>> X, Y = iris.data, iris.target
    >>> clf.fit(X, Y)
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
    >>> import pickle
    >>> s = pickle.dumps(clf)
    >>> clf2 = pickle.loads(s)
    >>> clf2.predict(X[0:1])
    array([0])

在scikit的特定情况下，使用joblib替换pickle（joblib.dump和joblib.load）可能更有趣，这对大数据更有效，但只能在磁盘上而不是字符串：

    >>> from sklearn.externals import joblib
    >>> joblib.dump(clf, 'filename.pkl') 

稍后您可以使用以下命令加载pickled模型（可能在另一个Python进程中）：

    >>> clf = joblib.load('filename.pkl') 

注意：joblib.dump和joblib.load函数也接受类文件对象而不是文件名。

### 约定

scikit-learn估算器遵循某些规则，使其行为更具预测性。

#### 类型铸造

除非另有说明，否则输入将转换为float64：

    >>> import numpy as np
    >>> from sklearn import random_projection
    >>>
    >>> rng = np.random.RandomState(0)
    >>> X = rng.rand(10, 2000)
    >>> X = np.array(X, dtype='float32')
    >>> X.dtype
    dtype('float32')

    >>> transformer = random_projection.GaussianRandomProjection()
    >>> X_new = transformer.fit_transform(X)
    >>> X_new.dtype
    dtype('float64')

在这个例子中，X是float32，由fit_transform（X）强制转换为float64。

回归目标转换为float64，维护分类目标：

    >>> from sklearn import datasets
    >>> from sklearn.svm import SVC
    >>> iris = datasets.load_iris()
    >>> clf = SVC()
    >>> clf.fit(iris.data, iris.target)  
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

    >>> list(clf.predict(iris.data[:3]))
    [0, 0, 0]

    >>> clf.fit(iris.data, iris.target_names[iris.target])  
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

    >>> list(clf.predict(iris.data[:3]))  
    ['setosa', 'setosa', 'setosa']

这里，第一个predict（）返回一个整数数组，因为iris.target（一个整数数组）用于fit。 第二个predict（）返回一个字符串数组，因为iris.target_names用于拟合。

### 重新配置和更新参数

通过sklearn.pipeline.Pipeline.set_params方法构建估计器的超参数后，可以更新估计器的超参数。 多次调用fit（）将覆盖之前任何fit（）所学的内容：

    >>> import numpy as np
    >>> from sklearn.svm import SVC

    >>> rng = np.random.RandomState(0)
    >>> X = rng.rand(100, 10)
    >>> y = rng.binomial(1, 0.5, 100)
    >>> X_test = rng.rand(5, 10)

    >>> clf = SVC()
    >>> clf.set_params(kernel='linear').fit(X, y)  
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
    >>> clf.predict(X_test)
    array([1, 0, 1, 1, 0])

    >>> clf.set_params(kernel='rbf').fit(X, y)  
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
    >>> clf.predict(X_test)
    array([0, 0, 0, 1, 0])

这里，默认内核rbf在通过SVC（）构造估计器之后首先变为线性，并且变回rbf以重新构造估计器并进行第二次预测。

### 多类与多标签拟合

使用多类分类器时，执行的学习和预测任务取决于适合的目标数据的格式：

    >>> from sklearn.svm import SVC
    >>> from sklearn.multiclass import OneVsRestClassifier
    >>> from sklearn.preprocessing import LabelBinarizer

    >>> X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
    >>> y = [0, 0, 1, 1, 2]

    >>> classif = OneVsRestClassifier(estimator=SVC(random_state=0))
    >>> classif.fit(X, y).predict(X)
    array([0, 0, 1, 1, 2])

在上面的例子中，分类器适合1d多类标签数组，因此predict（）方法提供相应的多类预测。 它也可以适用于二维标签指示符的二维数组：

    >>> y = LabelBinarizer().fit_transform(y)
    >>> classif.fit(X, y).predict(X)
    array([[1, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 0],
           [0, 0, 0]])
       
这里，分类器使用LabelBinarizer在y的二进制标签表示上拟合（）。 在这种情况下，predict（）返回表示相应多标记预测的二维数组。       
       
请注意，第四个和第五个实例返回全零，表示它们不匹配三个适合的标签。 使用多标签输出，同样可以为实例分配多个标签：    
            
    >> from sklearn.preprocessing import MultiLabelBinarizer
    >> y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
    >> y = MultiLabelBinarizer().fit_transform(y)
    >> classif.fit(X, y).predict(X)
    array([[1, 1, 0, 0, 0],
           [1, 0, 1, 0, 0],
           [0, 1, 0, 1, 0],
           [1, 0, 1, 1, 0],
           [0, 0, 1, 0, 1]])  

 在这种情况下，分类器适合于每个分配了多个标签的实例。 MultiLabelBinarizer用于对多标签的二维数组进行二值化以适应。 因此，predict（）返回一个带有每个实例的多个预测标签的二维数组。      
           
# 第二部分：用户指南
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
