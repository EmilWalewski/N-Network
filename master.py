# import tensorflow as tf

# import numpy as np
# import pandas as pd
# from sklearn import datasets, preprocessing
# import matplotlib.pyplot as plot

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report

# def convert_to_array(array):
    
#     test = np.zeros((150, 3))
    
#     for x in range(array.shape[0]):
#         if array[x] == 0:
#             test[x] = [1, 0, 0]
#         elif array[x] == 1:
#             test[x] = [0, 1, 0]
#         else:
#             test[x] = [0, 0, 1]
    
#     return test

# def read_train_test_data():

#     iris_dataset = datasets.load_iris()

#     x_data = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)

#     x_data.rename(columns={'sepal length (cm)': 'SepalLen',
#                             'sepal width (cm)': 'SepalWth',
#                             'petal length (cm)': 'PetalLen',
#                             'petal width (cm)': 'PetalWth'}, inplace=True)

#     x_data = x_data.apply(lambda x: ( (x-x.min()) / (x.max()-x.min()) ))

#     y_data = pd.DataFrame(iris_dataset.target, columns=['Iris-setosa','Iris-versicolor','Iris-virginica'])

#     X_Train, X_Test, Y_Train, Y_Test = train_test_split(x_data,y_data, test_size=0.3, random_state=101)
    
#     return X_Train, X_Test, Y_Train, Y_Test

# def create_feature_column():
#     feat_SepalLen = tf.feature_column.numeric_column('SepalLen')
#     feat_SepalWth = tf.feature_column.numeric_column('SepalWth')
#     feat_PetalLen = tf.feature_column.numeric_column('PetalLen')
#     feat_PetalWth = tf.feature_column.numeric_column('PetalWth')

#     feature_column = [feat_SepalLen, feat_SepalWth, feat_PetalLen, feat_PetalWth]

#     return feature_column


# X_Train, X_Test, Y_Train, Y_Test = read_train_test_data()

# feature_column = create_feature_column()

# input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_Train, y=Y_Train,batch_size=40,num_epochs=1000, shuffle=True)

# eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_Test, y=Y_Test,batch_size=40,num_epochs=1, shuffle=False)

# predict_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_Test, num_epochs=1, shuffle=False)

# model = tf.estimator.LinearClassifier(feature_columns=feature_column)

# history = model.train(input_fn=input_func, steps=1000)


# above my part----------------------------------------------------------------------



# print(normalized_data)

# print("Targets: "+str(iris.target_names))
# print("Features: "+str(iris.feature_names))
# print(iris.data[0:10:])

# plot.scatter(iris.data[:,1], iris.data[:,2], c=iris.target, cmap=plot.cm.Paired)
# plot.xlabel(iris.feature_names[1])
# plot.ylabel(iris.feature_names[2])
# plot.show()

# plot.scatter(iris.data[:,1], iris.data[:,2], c=iris.target, cmap=plot.cm.Paired)
# plot.xlabel(iris.feature_names[1])
# plot.ylabel(iris.feature_names[2])
# # plot.show()

# plot.scatter(iris.data[:,0], iris.data[:,3], c=iris.target, cmap=plot.cm.Paired)
# plot.xlabel(iris.feature_names[0])
# plot.ylabel(iris.feature_names[3])
# plot.show(

import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn import datasets, preprocessing
import matplotlib.pyplot as plot

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def readTrainTestData():

    path = "iris.csv"
    df = pd.read_csv(path)
        
    columns_norm = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']

    X_Data = df[columns_norm]
    
    print(X_Data)
        
    X_Data.rename(columns={'sepal.length': 'SepalLen',
                            'sepal.width': 'SepalWth',
                            'petal.length': 'PetalLen',
                            'petal.width': 'PetalWth'}, inplace=True)
        
    X_Data = X_Data.apply(lambda x:( (x - x.min()) / (x.max()-x.min())))

    Y_Data = df["variety"]
    Y_Data = df["variety"].map({
            "Setosa":0,
            "Virginica":1,
            "Versicolor":2})
        
    X_Train, X_Test, Y_Train,Y_Test = train_test_split(X_Data,
                                                           Y_Data,
                                                           test_size=0.3,
                                                           random_state=101)

    return X_Train, X_Test , Y_Train, Y_Test

def create_feature_column():
    feat_SepalLen = tf.feature_column.numeric_column('SepalLen')
    feat_SepalWth = tf.feature_column.numeric_column('SepalWth')
    feat_PetalLen = tf.feature_column.numeric_column('PetalLen')
    feat_PetalWth = tf.feature_column.numeric_column('PetalWth')

    feature_column = [feat_SepalLen,feat_SepalWth,feat_PetalLen,feat_PetalWth]  

    return feature_column

X_Train, X_Test, Y_Train, Y_Test = readTrainTestData()
feature_column = create_feature_column()

input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_Train, y=Y_Train,batch_size=40,num_epochs=1000, shuffle=True)

eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_Test, y=Y_Test,batch_size=40,num_epochs=1, shuffle=False)

predict_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_Test, num_epochs=1, shuffle=False)

model = tf.estimator.LinearClassifier(feature_columns=feature_column, n_classes=10)

history = model.train(input_fn=input_func, steps=1000)

results = model.evaluate(eval_input_func)

print(results['accuracy'])

predictions = list(model.predict(input_fn=predict_input_func))
prediction = [p["class_ids"][0] for p in predictions]
data = classification_report(Y_Test, prediction)
conmat = confusion_matrix(Y_Test, prediction)

import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(Y_Test, prediction, figsize=(6, 6), title="Confusion Matrix")
