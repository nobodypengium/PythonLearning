import partIVch2.kt_utils as kt_utils
from partIVch2.network import *

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = kt_utils.load_dataset()

X_train = X_train_orig/255
X_test = X_test_orig/255
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

happy_model = HappyModel(X_train.shape[1:])
happy_model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

#训练模型
happy_model.fit(X_train, Y_train, epochs=40, batch_size=50)

#评估模型
preds = happy_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
print("误差值 = " + str(preds[0]))
print("准确度 = " + str(preds[1]))