from partIVch2_1.network import *
from partIVch2_1.resnets_utils import *

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig/255
X_test = X_test_orig/255

Y_train = convert_to_one_hot(Y_train_orig,6).T
Y_test = convert_to_one_hot(Y_test_orig,6).T

model = ResNet50(input_shape=(64,64,3), classes=6)
model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(X_train,Y_train,epochs=2,batch_size=32)

preds = model.evaluate(X_test,Y_test)

print("误差率：" + str(preds[0]))
print("准确率：" + str(preds[1]))



