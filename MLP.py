import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles 
import pandas as pd
import jax.numpy as jnp
import jax 
from sklearn.metrics import confusion_matrix
import jax.numpy as jnp
from jax import grad
from jax import random
from functools import partial
from jax import jit



#Codigo en JAX usando automatic differentiation:

class CNN():
    def __init__(self, topology, act_f, key_seed=0): 
        self.topology = topology
        self.act_f = act_f

        nn= [] #es para crear un vector vacio para ir agregando las capas estos son los parametros  W y b 
        for l, layer in enumerate(topology[:- 1]):
            n_conn= topology[l]
            n_neur= topology[l+1]
            ''' Generate random values for b and W using JAX's random module'''
            key = random.PRNGKey(key_seed+l)
            b = random.uniform(key, (1, n_neur), minval=-1.0, maxval=1.0)
            W = random.uniform(key, (n_conn, n_neur), minval=-1.0, maxval=1.0)
            nn.append((W,b))

        self.nn=nn
        
    def forward_jax(self,X:jnp, params)-> jnp:
        out = [X]    
        for w,b in params[:-1]:
            #w=jnp.array(w,dtype=jnp.float32)
            #b=jnp.array(b,dtype=jnp.float32)
            z=jnp.dot(out[-1],w)+b
            a=jnp.tanh(z)
            out.append(a)
        
        w_final,b_final=params[-1]
        logits = jnp.dot(a,w_final) + b_final
        return jax.nn.softmax(logits)
     
    @partial(jax.jit, static_argnums=(0,)) 
    def loss(self,params, X, Y)-> jnp: 
        last_a =self.forward_jax(X,params) 
        loss = -jnp.sum(Y * jnp.log(last_a)) 
        return loss / X.shape[0]
    
    #@partial(jax.jit, static_argnums=(0,))
    def update(self, params, X, Y, lr):
        grads=[]
        gradientes=grad(self.loss)(params, X, Y)
        grads.append(gradientes)
        for l, layer in enumerate(self.nn):
            new_w=self.nn[l][0] -grads[0][l][0]* lr
            new_b=self.nn[l][1] -grads[0][l][1]* lr
            self.nn[l]=(new_w,new_b)

        last_a=self.forward_jax(X, params)
        return last_a

    def train_jax(self,X,Y,lr=0.38):
        self.forward_jax(X,self.nn)
        last_a=self.update(self.nn ,X, Y, lr)
        return last_a
    
    #---------------------------metrics-----------------------------
    @staticmethod
    @jit
    def get_predictions(A2):
        return jnp.argmax(A2,axis=1)

    @staticmethod
    @jit
    def get_accuracy(predictions, Y_sinhot):
        #print(predictions, Y_sinhot)
        return jnp.sum(predictions == Y_sinhot) / Y_sinhot.size

    @staticmethod
    @jit
    def recall_2clases(y_sinhot, y_hat):
        TP=0
        FN=0
        for i in range(len(y_sinhot)):
            if (y_sinhot[i]>0 and y_hat[i]>0):
                TP += 1
            if (y_sinhot[i]<=0 and y_hat[i]>0):
                FN +=1
        recall = (TP/(TP+FN))
        return float(recall)

    @staticmethod
    @jit
    def recall(y_sinhot, y_hat,return_TP_FN=False):
        TP=0
        FN=0
        unique=jnp.unique(y_hat)
        recalls=[]
        for clase in range(len(unique)):
            for muestra in range(len(y_sinhot)):
                if (y_sinhot[muestra]==clase and y_hat[muestra]==clase):
                    TP += 1
                if (y_sinhot[muestra]==clase and y_hat[muestra]!=clase):
                    FN +=1
            recalls.append(TP/(TP+FN))

        mean_recall=jnp.mean(recalls) 
        if return_TP_FN:
            return mean_recall,TP,FN
        else:
            return mean_recall
            
    @staticmethod
    @jit
    def precision(y_sinhot, y_hat,return_TP_FP=False):
        TP=0
        FP=0
        unique=jnp.unique(y_hat)
        precisions=[]
        for clase in range(len(unique)):
            for muestra in range(len(y_sinhot)):
                if (y_sinhot[muestra]==clase and y_hat[muestra]==clase):
                    TP += 1
                if (y_sinhot[muestra]!=clase and y_hat[muestra]==clase):
                    FP +=1
            #para evitar divisiones entre 0
            if (TP+FP)==0:
                precisions.append(0)
            else:
                precisions.append(TP/(TP+FP))
        
        mean_precision=jnp.mean(precisions)
        if return_TP_FP:
            return mean_precision,TP,FP
        else:
            return mean_precision
        

#----------funciones de activacion----------------
#funciones de activacion sigmoide y tanh (#funcion de activacion y derivada)
sigm=(lambda x: 1/(1+ np.e**(-x)),lambda x: x*(1-x))
# por defecto se usara la funcion de activacion tanh
tanh=(lambda x: np.tanh(x),lambda x: 1-np.tanh(x)**2) 

#--------------- FUNCIONES PARA METRICAS ----------------
def get_predictions(A2):
    return np.argmax(A2,axis=1)

def get_accuracy(predictions, Y_sinhot):
    #print(predictions, Y_sinhot)
    return np.sum(predictions == Y_sinhot) / Y_sinhot.size

def recall(y_sinhot, y_hat,return_TP_FN=False):
    """
    recall
    args:
        y_sinhot: Real Labels
        y_hat: estimated labels
    return TP/(TP+FN)
    """
    TP=0
    FN=0
    unique=np.unique(y_hat)
    recalls=[]
    for clase in range(len(unique)):
        for muestra in range(len(y_sinhot)):
            if (y_sinhot[muestra]==clase and y_hat[muestra]==clase):
                TP += 1
            if (y_sinhot[muestra]==clase and y_hat[muestra]!=clase):
                FN +=1
        recalls.append(TP/(TP+FN))

    mean_recall=np.mean(recalls) 
    #print("Recall: ", mean_recall)
    if return_TP_FN:
        return mean_recall,TP,FN
    else:
        return mean_recall
        

def precision( y_sinhot, y_hat,return_TP_FP=False):
    """
    precision                           
    args:
        y_sinhot: Real Labels
        y_hat: estimated labels
    return TP/(TP+FP)
    """
    TP=0
    FP=0
    #return float(precision)
    unique=np.unique(y_hat)
    precisions=[]
    for clase in range(len(unique)):
        for muestra in range(len(y_sinhot)):
            if (y_sinhot[muestra]==clase and y_hat[muestra]==clase):
                TP += 1
            if (y_sinhot[muestra]!=clase and y_hat[muestra]==clase):
                FP +=1
        #para evitar divisiones entre 0
        if (TP+FP)==0:
            precisions.append(0)
        else:
            precisions.append(TP/(TP+FP))
    
    mean_precision=np.mean(precisions)
    if return_TP_FP:
        return mean_precision,TP,FP
    else:
        return mean_precision
    
#--------One hot encoding----------------
def one_hot(x, k, dtype=jnp.float32):
        """Create a one-hot encoding of x of size k."""
        return jnp.array(x[:, None] == jnp.arange(k), dtype)




#Codigo para entrenar usando JAX code
#aqui pasara todo foward pass, backward pass y decenso del gradiente
def trainig_jax_model(train_images,Y_sinhot,topology,steps=500,lr=0.1,threshold=0.008, precision_recall_steps=100): 
  '''
  args:
    train_images: imagenes de entrenamiento shape (datos,features)
    Y_sinhot: etiquetas de entrenamiento sin hot encoding
    topology: topologia de la red neuronal ej: [784,32,10]
    steps: numero de pasos para el entrenamiento default=2000
    lr: learning rate para el decenso del gradiente default=0.28
    threshold: umbral para detener el entrenamiento default=0.000008
  return:
    red_neuronal: red neuronal entrenada
    loss: lista con el valor de la funcion de perdida en cada paso
    acuracies: lista con el valor de la acuracia en cada paso
    precisions: lista con el valor de la precision en cada paso
    recalls: lista con el valor de la recall en cada paso
  '''
  num_labels = len(np.unique(Y_sinhot))
  train_labels = one_hot(Y_sinhot, num_labels)
  precisions=[]
  recalls=[]
  loss=[]
  acuracies=[]
  red_neuronal=CNN(topology, tanh)

  for i in range (steps):
    salida_ultima_capa=red_neuronal.train_jax(train_images,train_labels,lr)
    loss.append(red_neuronal.loss(red_neuronal.nn,train_images,train_labels))
    '''print cada 10 pasos para ahorrar tiempo de computo'''
    if i%10==0:
      prediccion=get_predictions(salida_ultima_capa)   
      prediccion=jnp.reshape(prediccion, Y_sinhot.shape)
      acuracies.append(get_accuracy(prediccion,Y_sinhot))
      print('training ------> step=',i,'lost: {:.3f}, acuracy: {:.3f}'.format(loss[-1], acuracies[-1]), end='\r')
      print('training ------> step=',i,'lost: {:.3f}'.format(loss[-1]), end='\r')


    '''print cada 100 pasos para ahorrar tiempo de computo, estas metricas son demasiado costosas'''
    if i% precision_recall_steps==0 and i!=0:
      prediccion=get_predictions(salida_ultima_capa)   
      prediccion=jnp.reshape(prediccion, Y_sinhot.shape)
      acuracies.append(get_accuracy(prediccion,Y_sinhot))
      precisions.append(precision(Y_sinhot[:500],prediccion[:500]))
      recalls.append(recall(Y_sinhot[:500],prediccion[:500]))
      print('\r' + ' ' * 100, end='\r')
      print('training ------> step=',i,'lost: {:.3f}, acuracy: {:.3f}, recall: {:.3f},precisions: {:.3f}'.format(loss[-1],acuracies[-1], recalls[-1], precisions[-1]), end='\r')

    #si la perdida no cambia mucho en 2 pasos, se detiene el entrenamiento
    if i>1:
      if jnp.abs(loss[-1]-loss[-2])< threshold:
        '''se realizan los calculos de precision y recall para los primeros 
            3000 datos (pueden ser mas, pero requiere mas tiempo de computo)'''
        
        print('\n Stop training, loss not change more than threshold, calculating model metrics...(wait)')
        prediccion=get_predictions(salida_ultima_capa)    
        prediccion=jnp.reshape(prediccion, Y_sinhot.shape)
        acuracies.append(get_accuracy(prediccion,Y_sinhot))
        precisions.append(precision(Y_sinhot[:3000],prediccion[:3000]))
        recalls.append(recall(Y_sinhot[:3000],prediccion[:3000]))
        print('\n -------------------------final metrics-----------------------------')
        print('step=',i,'lost: {:.3f}, acuracy: {:.3f}, recall: {:.3f},precisions: {:.3f}'.format(loss[-1],acuracies[-1], recalls[-1], precisions[-1]))
        
        break
  return recalls,precisions,loss,prediccion


#----------------------funciones para graficar----------------------
import seaborn as sns
def precision_recall_plot(recalls,precisions):
  plt.style.use('rose-pine')
  plt.plot(recalls,precisions)
  plt.xlabel('recall')
  plt.ylabel('precision')
  #titulo
  plt.title('Precision-Recall Curve')
  plt.savefig('precision_recall_curve.png')
  plt.show()

def loss_plot(loss):
  plt.plot(loss)
  plt.xlabel('steps')
  plt.ylabel('loss')
  #titulo
  plt.title('Loss Curve')
  plt.savefig('loss_curve.png')
  plt.show()

def confusion_matrix_plot(Y_sinhot, last_prediccion):
  #confusion matrix
  cm=confusion_matrix(Y_sinhot, last_prediccion)
  #print(cm)
  #title
  #plt.figure(figsize=(3,2))
  sns.heatmap(cm, annot=True)
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.title('Confusion Matrix')
  plt.savefig('confusion_matrix.png')
  plt.show()


if __name__ == '__main__':
    recalls,precisions,loss,last_prediccion=trainig_jax_model(train_data,train_labels_sinhot,
                                                              topology,steps,lr,threshold,precision_recall_steps)
    
    precision_recall_plot(recalls,precisions)
    loss_plot(loss)
    confusion_matrix_plot(Y_sinhot, last_prediccion)