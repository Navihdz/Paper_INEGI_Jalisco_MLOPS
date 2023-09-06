import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
from jax import random
import os
import matplotlib.pyplot as plt
import mlflow
# Switch off the cache 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'




class Linear_Model():
    """
    Basic Linear Regression with Ridge Regression
    """
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.key = random.PRNGKey(0)
        self.cpus = jax.devices("cpu")
    
    # The classic one
    def generate_canonical_estimator(self, X: jnp, y:jnp) -> jnp:
        """
        Cannonical LSE error solution for the Linearly separable classes 
        args:
            X: Data array at the GPU or CPU
            y: Label array at the GPU 
        returns:
            w: Weight array at the GPU or CPU
        """
        return  jax.numpy.linalg.inv(jax.numpy.transpose(X)@X)@jax.numpy.transpose(X)@y

    def generate_ridge_estimator(self, X: jnp, y:jnp, lamda:float) -> jnp:
        """
        Cannonical LSE error solution for the Linearly separable classes 
        args:
            X: Data array at the GPU or CPU
            y: Label array at the GPU 
        returns:
            w: Weight array at the GPU or CPU
        """
        
        XX=jax.numpy.transpose(X)@X
        
        Identity=jax.numpy.identity(jax.numpy.shape(XX)[0])
        w_ridge=jax.numpy.linalg.inv(XX + lamda * Identity )@jax.numpy.transpose(X)@y
        return  w_ridge
    
    @staticmethod
    @jit
    def linear_model(X: jnp, theta: jnp) -> jnp:
        """
        Classic Linear Model. Jit has been used to accelerate the loops after the first one
        for the Gradient Descent part
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
        returns:
            f(x): the escalar estimation on vector x or the array of estimations
        """
        w = theta[:-1]
        b = theta[-1]
        return jax.numpy.matmul(X, w) + b
    
    @partial(jit, static_argnums=(0,))
    def LSE(self, theta: jnp, X: jnp, y: jnp)-> jnp:
        """
        LSE in matrix form. We also use Jit por froze info at self to follow 
        the idea of functional programming on Jit for no side effects
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
            y: array of labels
        returns:
            the Loss function LSE under data X, labels y and theta initial estimation
            
        """
        error=(jax.numpy.transpose(y - self.linear_model(X, theta))@(y - self.linear_model(X, theta)))[0,0]
        return error
    
    @partial(jit, static_argnums=(0,))
    def update(self, theta: jnp, X: jnp, y: jnp, lr):
        """
        Update makes use of the autograd at Jax to calculate the gradient descent.
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
            y: array of labels
            lr: Learning rate for Gradient Descent
        returns:
            the step update w(n+1) = w(n)-δ(t)𝜵L(w(n))   
                 
        """
        error=jax.grad(self.LSE)(theta, X, y)
        theta_2=theta - lr * error
        
        return theta_2, error
    
    def generate_theta(self):
        """
        Use the random generator at Jax to generate a random generator to instanciate
        the augmented values
        """
        keys = random.split(self.key, 1)
        return jax.numpy.vstack([random.normal(keys[0], (self.dim,1)), jax.numpy.array(0)])
        #return jax.numpy.vstack([random.normal(keys), jax.numpy.array(0)])
    
    @partial(jit, static_argnums=(0,))
    def estimate_grsl(self, X, theta):
        """
        Estimation for the Gradient Descent version
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
        return:
            Estimation of data X under linear model
        """
        w = theta[:-1]
        b = theta[-1]
        return X@w+b
    
    @staticmethod
    def estimate_cannonical(X: jnp, w: jnp)->jnp:
        """
        Estimation for the Gradient Descent version
        args:
            X: Data array at the GPU or CPU
            w: Parameter w under extended space
        return:
            Estimation of data X under cannonical solution
        """
        return X@w
    
    def recall(self, y, y_hat):
        """
        recall
        args:
            y: Real Labels
            y_hat: estimated labels
        return TP/(TP+FN)
        """
        TP=0
        FN=0
        print(y)
        print(y_hat)
        for i in range(len(y)):
            if (y[i]>0 and y_hat[i]>0):
                TP += 1
            if (y[i]>0 and y_hat[i]<0):
                FN +=1
        #evitar division por cero
        print(TP, FN)
        if (TP+FN)==0:
            return 0
        else:
            precision_cpu = jax.jit(lambda x: x, device=self.cpus[0])(TP/(TP+FN))
            return float(precision_cpu)
        
    def recall_jax(self,y, y_hat):
        """
            recall
            args:
                y: Real Labels
                y_hat: estimated labels
            return TP/(TP+FN)
        """
        TP = jnp.sum((y > 0) & (y_hat > 0))
        FN = jnp.sum((y > 0) & (y_hat <= 0))

        #evitar division por cero
        recall_cpu = jax.lax.cond(
            TP + FN == 0,
            lambda _: 0.0,
            lambda _: TP / (TP + FN),
            operand=None,
        )

        return float(recall_cpu)
    
    def precision_jax(self, y, y_hat):
        """
        precision
        args:
            y: Real Labels
            y_hat: estimated labels
        return TP/(TP+FP)
        """
        TP = jnp.sum((y > 0) & (y_hat > 0))
        FP = jnp.sum((y <= 0) & (y_hat > 0))

        #evitar division por cero
        precision_cpu = jax.lax.cond(
            TP + FP == 0,
            lambda _: 0.0,
            lambda _: TP / (TP + FP),
            operand=None,
        )

        return float(precision_cpu)


    def accuracy(self, y, y_hat):
        """
        accuracy
        args:
            y: Real Labels
            y_hat: estimated labels
        return  TP +TN/ TP +FP +FN+TN
        """
        TP=0
        FP=0
        FN=0
        TN=0
        for i in range(len(y)):
            if (y[i]>0 and y_hat[i]>0):
                TP += 1
            if (y[i]<0 and y_hat[i]<0):
                TN +=1
            if (y[i]<0 and y_hat[i]>0):
                FP +=1
            if (y[i]>0 and y_hat[i]<0):
                FN +=1  
        #evitar division por cero         
        if (TP+FP+TN+FN)==0:
            return 0
        else:
            precision_cpu = jax.jit(lambda x: x, device=self.cpus[0])((TP+TN)/(TP+FP+TN+FN))
            return float(precision_cpu)
        
    def accuracy_jax(self,y, y_hat):
        """
            accuracy
            args:
                y: Real Labels
                y_hat: estimated labels
            return  TP +TN/ TP +FP +FN+TN
        """
        TP = jnp.sum((y > 0) & (y_hat > 0))
        FP = jnp.sum((y <= 0) & (y_hat > 0))
        FN = jnp.sum((y > 0) & (y_hat <= 0))
        TN = jnp.sum((y <= 0) & (y_hat <= 0))
        
        #evitar division por cero         
        if (TP+FP+TN+FN)==0:
            return 0
        else:
            accuracy_cpu = jit(lambda x: x, device=self.cpus[0])((TP+TN)/(TP+FP+TN+FN))
            return float(accuracy_cpu)
    
    def gradient_descent(self, theta: jnp,  X: jnp, y: jnp, n_steps: int, lr):
        """
        Gradient Descent Loop for the LSE Linear Model
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
            y: array of labels
            n_steps: number steps for the Gradient Loop
            lr: Learning rate for Gradient Descent   
        return:
            Updated Theta
        """
        for i in range(n_steps):
            theta,error = self.update(theta, X, y, lr)
            #print('este es el paso',i, 'error:', self.LSE(theta, X, y))
            if i%100==0:
                print('step: {}, error: {:.3f}'.format(i, self.LSE(theta, X, y)))

            
        return theta
    

    def ridge_regression(self,X,y,lamda):
        '''
        args:
            X: Data  with shape (n_samples, n_features) ej: (100, 2)
            y: array of labels with shape (n_samples, 1)
            k_clases: number of classes default 2
            lamda: regularization parameter
        return:

            w_rigdge: Updated Theta
            y_hat_ridge: estimated labels
            recall: recall
            accuracy: accuracy
        '''
        #creamos X aumentado con y
        X_aug = jnp.hstack([X, y])

        #model = Linear_Model(k_clases=k_clases)
        w_rigdge = self.generate_ridge_estimator(X_aug, y, lamda)  #calcula w  usando Ridge regression, si usamos lamda=0 recuperamos canonico
        y_hat_ridge = self.estimate_cannonical(X_aug, w_rigdge)  #probamos que funcione igual que canonico
        recall=self.recall_jax(y, y_hat_ridge)
        accuracy=self.accuracy_jax(y, y_hat_ridge)
        precision=self.precision_jax(y, y_hat_ridge)

        return w_rigdge, y_hat_ridge, precision,recall, accuracy
    

def descenso_gradiente(X,y,steps,lr): 
    '''
    args:
        X: Data  with shape (n_samples, n_features)
        y: array of labels (-1 y 1) with shape (n_samples, 1)
        steps: number steps for the Gradient Loop
        lr: Learning rate for Gradient Descent
    return:
        theta: Updated Theta
        y_hat: estimated labels
        recall: recall
        accuracy: accuracy
    '''  
    #------------------------------------MLFLOW------------------------------------
    with mlflow.start_run(run_name="descenso_gradiente") as run:
        mlflow.log_param("steps", steps)
        mlflow.log_param("lr", lr)
        mlflow.log_param("n_features", len(X[1]))   #numero de features
    #------------------------------------------------------------------------------

        y=jnp.array(y)
        X=jnp.array(X)
        y=jnp.reshape(y, (len(y),1))
        k_clases=len(X[1])  #numero de clases es igual al numero de columnas o features de X 

        model = Linear_Model(k_clases)
        
        theta = model.generate_theta()  #generamos theta con valores inicales aleatorios
        theta = model.gradient_descent(theta, X, y, steps, lr)
        y_hat = model.estimate_grsl(X, theta)
        precision=model.precision_jax(y, y_hat)
        #recall=model.recall(y, y_hat)
        recall=model.recall_jax(y, y_hat)
        #accuracy=model.accuracy(y, y_hat)
        accuracy=model.accuracy_jax(y, y_hat)


        #------------------------------------MLFLOW------------------------------------
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", accuracy)
        #------------------------------------------------------------------------------

        return theta, y_hat, precision, recall, accuracy




def RidgeRegresion(X, y, lamda):
    '''
    args:
        X: Data  with shape (n_samples, n_features)
        y: array of labels (0 y 1) with shape (n_samples, 1) 
        lamda: regularization parameter
    return:
    '''
    #------------------------------------MLFLOW------------------------------------
    with mlflow.start_run(run_name="grid_search_Ridge") as run:
        mlflow.log_param("lamda_grid", lamda)
        #features
        mlflow.log_param("n_features", len(X[1]))
    #------------------------------------------------------------------------------
    #revisa cuantas clases hay en label y
        y=jnp.array(y)
        X=jnp.array(X)
        y=jnp.reshape(y, (len(y),1))

        k_clases=len(X[1])  #numero de clases es igual al numero de columnas o features de X 
        model = Linear_Model(k_clases)       
        w_rigdge, y_hat_ridge, precision,recall, acuracy = model.ridge_regression(X,y,lamda)

        #------------------------------------MLFLOW------------------------------------
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", acuracy)
        #------------------------------------------------------------------------------

        return w_rigdge, y_hat_ridge,precision, recall, acuracy



if __name__ == '__main__':

    theta, y_hat, precision, recall, accuracy= descenso_gradiente(X,y,steps,lr)
    w_rigdge, y_hat_ridge,precision, recall, acuracy= RidgeRegresion(X, y, lamda)