import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
class ODEsolver(Sequential):
    loss_tracker = keras.metrics.Mean(name="loss")
    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        #minival-5 y maxval5 son los intervalos
        x = tf.random.uniform((batch_size, 1), minval=-5, maxval=5)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                #y_pred son los valores aletorios de "y" los cuales buscan aproximar
                #a la función original
                y_pred = self(x, training=True)
            dy = tape2.gradient(y_pred, x)
            dy2 = tape.gradient(dy, x)
            # Se definen los tensores para las condiciones iniciales en x
            x_0 = tf.zeros((batch_size, 1))
            # Se definien los valores de entrenamiento de "y" con respecto a los valores de x
            y_0 = self(x_0, training=True)
            dy_0 = self(y_0, training=True)
            dy2_0 = self(y_pred, training=True)
            eq = dy2 + y_pred
            #se tienen las siguientes condiciones iniciales que nos sirvan para aproximar la función
            ic = y_0 - 1
            ic2 = dy_0 + 0.5
            ic3 = dy2_0
            # para la obtención de la función de costo utilizamos el error cuadrático medio proporcionada por keras
            # la función de costo está compuesta por el error cuadrático medio entre la ecuacion diferencial y los valores de entrenamiento de "y"
            # además de el error cuadrático medio entre las condiciones iniciales y los valores de entrenamiento generadas para las mismas
            loss = 60*keras.losses.mean_squared_error(0., eq) + 20*keras.losses.mean_squared_error(0., ic) + 0.5*keras.losses.mean_squared_error(0., ic2) + 0.000005*keras.losses.mean_squared_error(0., ic3)
        grads = tape.gradient(loss, self.trainable_variables)
        #Se aplica el optimizador de gradiente y se junta con los valores de entrenamiento
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
        @property
        def metrics(self):
            return [keras.metrics.Mean(name='loss')]

##

model = ODEsolver()

model.add(Dense(10, activation='tanh', input_shape=(1,)))
model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(optimizer=RMSprop(), metrics=['loss'])
tf.keras.layers.Dropout(.25, input_shape=(2,))
x = tf.linspace(-5, 5, 1000)
history = model.fit(x, epochs=1000, verbose=1)
x_testv = tf.linspace(-5, 5, 1000)
y = [(np.cos(x)-0.5*np.sin(x)) for x in x_testv]
a = model.predict(x_testv)
plt.grid()
plt.title('Solución encontrada por la red vs solución analitica')
plt.plot(x_testv, a)
plt.plot(x_testv, y)
plt.show()
model.save('red2.1.h5')
exit()