
from tensorflow.keras.layers import Input, Dense, Dropout,LSTM,SimpleRNN
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential
import numpy as np
import matplotlib.pyplot as plt



class LSTM_Model:

    def __init__(self,Y,delta,stp):
        self.Y = Y
        self.delta = delta
        self.stp = stp

    

    def make_model(self,Y,delta,ep=100):

        plt.plot(self.Y)
        plt.savefig("static/learningdata.png")
        plt.close()

        #学習用データ長作成
        N = len(Y)-delta-1
        if N <= delta:
            return 

        x_train = []
        for i in range(N):
            x_train.append(Y[i:i+delta])

        x_train = np.array(x_train)
        x_train = x_train.reshape(N,delta,1)
        

        y_train = np.array(Y[delta:delta+N])
        y_train = y_train.reshape(N,1)
        
        
        model = Sequential()
        model.add(LSTM(units=delta*2,input_shape=(delta,1),return_sequences=False,activity_regularizer=regularizers.l2(0.0),dropout=0.))
        model.add(Dense(units=1,activation='linear'))


        model.compile(optimizer='adam',loss='mean_squared_error')

        model.fit(x_train,y_train,epochs=ep,validation_split=0.2)

        return model, y_train[-delta:].reshape(1,delta,1)

    def model_predict(self):
        
        
        Model, input_series = self.make_model(self.Y,self.delta)
        X = []
        
        for i in range(self.stp):
            point = Model.predict(input_series)
            X.append(point[0][0])
            middle_series = np.delete(input_series[0,:,0],0)
            input_series[0,:,0] = np.append(middle_series,point[0][0])
    
        plt.plot(X)
        plt.savefig('static/plot.png') 
        return



