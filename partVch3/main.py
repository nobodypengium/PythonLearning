from partVch3.network import *
model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.001)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
s0 = np.zeros((m,n_s))
c0 = np.zeros((m,n_s))
outputs = list(Yoh.swapaxes(0,1))
model.fit([Xoh,s0,c0],outputs,epochs=1,batch_size=100)


