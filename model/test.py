#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# from IPython.display import Image
from sklearn import tree
import pydotplus

# 仍然使用自带的iris数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 总训练集：验证集 = 8:2
X_train, X_test = train_test_split(X, test_size=0.2, random_state=28)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=28)

# 训练模型，限制树的最大深度4
clf = DecisionTreeClassifier(max_depth=2)

#拟合模型
clf = clf.fit(X_train, y_train)
score = clf.score(X_test,y_test)
print(score)   #测试结果

# 混淆矩阵
from sklearn.metrics import confusion_matrix
test_predict = clf.predict(X_test)
cm = confusion_matrix(y_test,test_predict)
print(cm)

# 决策树可视化
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
# 使用ipython的终端jupyter notebook显示。
# Image(graph.create_png())
# 如果没有ipython的jupyter notebook，可以把此图写到pdf文件里，在pdf文件里查看。
graph.write_pdf("tree_deep2.pdf")
