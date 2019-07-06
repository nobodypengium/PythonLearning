from partVch2_1.network_v2 import *

max_len=10
model = Emojify_V2((max_len,),word_to_vec_map,word_to_index)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

X_train, Y_train = emo_utils.read_csv('data/train_emoji.csv')
X_test, Y_test = emo_utils.read_csv('data/test.csv')

#训练集
X_train_indices = sentences_to_indices(X_train,word_to_index,max_len)
Y_train_oh=emo_utils.convert_to_one_hot(Y_train,C=5)
model.fit(X_train_indices,Y_train_oh,epochs=50,batch_size=32,shuffle=True)

#测试集
X_test_indices = sentences_to_indices(X_test,word_to_index,max_len=max_len)
Y_test_oh=emo_utils.convert_to_one_hot(Y_test,C=5)
loss, acc = model.evaluate(X_test_indices,Y_test_oh)
print("Test accurcy = ",acc)

# 打印分类错误的
C=5
Y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test,word_to_index,max_len)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num!=Y_test[i]):
        print('正确表情: ' + emo_utils.label_to_emoji(Y_test[i]) + '    预测结果: ' + X_test[i] + emo_utils.label_to_emoji(num).strip())


x_test=np.array(['you are so beautiful'])
X_test_indices = sentences_to_indices(x_test,word_to_index,max_len)
print(x_test[0] +' '+  emo_utils.label_to_emoji(np.argmax(model.predict(X_test_indices))))