
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import jax.numpy as jnp
import jax
from jax import jit
import pandas as pd
#Compute the Gamma function for the E-Step

def gamma_fun(TotalClasses,mus,sigmas,pis):
    '''  
    This function calculates the gamma probability and N_k (the sum of gammas for each class) 
    args: 
        TotalClasses: List of NDarrays, each array is a class with the data of the class 
        mus: Tuple of NDarrays, each array have the mu of each class of shape (NumberOfFeatures,1)
        sigmas: Tuple of NDarrays, each array have the sigma of each class of shape (NumberOfFeatures,NumberOfFeatures)
        pis: Tuple of floats, each float is the pi of each class (you can start with => (Number of samples in each class/total samples)

    return: 
        gamma_k: array of gamma probability of shape (samples,number of classes ) 
        N_k: NDarray of floats, each float is the sum of gamma for each class
    ''' 

    NumberOfFeatures=len(TotalClasses[0][:,0])  #numero de features ex.2
    ShapeVectores=max([len(TotalClasses[0][0,:]),len(TotalClasses[1][0,:])])
    NumberOfClasses=len(TotalClasses)   #numero de clases ex.2

    #------Is to compute the denominator of gamma probability ->  sum_k(Pi_k * N(x|mu_k,sigma_k)) ---------------
    denominador_gamma_array=np.zeros(shape=(ShapeVectores,NumberOfClasses))
    for each_clase in range(len(TotalClasses)):     #is for the sum of each x \times with the normal of each class
        denominador_gamma_array_k=np.zeros(shape=(ShapeVectores,2))  #clean the array for each class
        p_classe=pis[each_clase] 
        sigma=sigmas[each_clase]
        sigma += np.eye(NumberOfFeatures) * 0.01
        mu=np.reshape(mus[each_clase],(NumberOfFeatures,1))     #reshape para evitar tener (NumberOfFeatures,)
        determinante_sigma=np.linalg.det(sigma)
        inv_sigma=np.linalg.inv(sigma)
        for clase in range(len(TotalClasses)):  #is for the x who lives in each X_class
            TamañoClase=len(TotalClasses[clase][0,:])
            for element in range(TamañoClase):  #element inside X_class (ex. X_class1 or X_class2)
                current_x=np.reshape(TotalClasses[clase][:,element],(NumberOfFeatures,1))
                #anteslen(TotalClasses)
                #print(1/(np.sqrt((2*np.pi**NumberOfFeatures)*determinante_sigma)))
                #print(np.exp(-1/2*(np.transpose(current_x-mu)@(inv_sigma)@(current_x-mu))))
                #print(determinante_sigma)
                gaussian=(1/(np.sqrt(2*np.pi**(NumberOfFeatures)*determinante_sigma)))*np.exp(-1/2*(np.transpose(current_x-mu)@(inv_sigma)@(current_x-mu)))
                denominador_gamma_array_k[element,clase]=p_classe*gaussian
            denominador_gamma_array=denominador_gamma_array+denominador_gamma_array_k  
    #guardo  denominador_gamma_array en data frame y luego en csv
    #print(denominador_gamma_array.shape)
    #print(denominador_gamma_array)

    df=pd.DataFrame(denominador_gamma_array)
    df.to_csv('denominador_gamma_array.csv')

    #------Compute Gamma probabilitie and N_k  ->    gamma_ik= pi*N(x|mu_k,sigma_k) / sum_k(Pi_k * N(x|mu_k,sigma_k))  and N_k=sum(Kamma_k) -----------------
    gamma_k=np.zeros(shape=(ShapeVectores,NumberOfClasses))#---------------antes  gamma_k=np.zeros(shape=(ShapeVectores,NumberOfClasses))
    N_k=np.zeros(shape=(len(TotalClasses)))
    #gaussianas=np.zeros(shape=(ShapeVectores,NumberOfClasses))
    for clase in range(len(TotalClasses)):
        TamañoClase=len(TotalClasses[clase][0,:])
        N_k[clase]=0
        p_classe=pis[clase]
        sigma=sigmas[clase]
        sigma += np.eye(NumberOfFeatures) * 0.01
        mu=np.reshape(mus[clase],(NumberOfFeatures,1))     #reshape para evitar tener (2,)
        determinante_sigma=np.linalg.det(sigma)
        inv_sigma=np.linalg.inv(sigma)
        for element in range(TamañoClase):
            current_x=np.reshape(TotalClasses[clase][:,element],(NumberOfFeatures,1))
            gaussian=(1/(np.sqrt(2*np.pi)**(NumberOfFeatures)*determinante_sigma))*np.exp(-1/2*(np.transpose(current_x-mu)@(inv_sigma)@(current_x-mu)))
            #gaussianas[element,clase]=gaussian

            ##evitar division por cero
            #if denominador_gamma_array[element,:].sum()==0:#----------------antes denominador_gamma_array[element,clase]
            #    gamma_probaility=0
            #else:
            gamma_probaility=p_classe* gaussian/denominador_gamma_array[element,clase]
            gamma_k[element,clase]=gamma_probaility
            N_k[clase]= N_k[clase] + gamma_probaility

    return gamma_k,N_k




def M_step(TotalClasses,N_k,gamma_k):
    '''
    This function compute de M step of the EM algorithm 
    args: 
        TotalClasses: List of NDarrays, each array is a class with the data of the class 
        N_k: NDarray of floats, each float is the sum of gamma for each class
        gamma_k: Array of gamma probability of shape (samples,number of classes )
    return: 
        mus: Tuple of NDarrays, each array have the mu of each class of shape (NumberOfFeatures,1)
        sigmas: Tuple of NDarrays, each array have the sigma of each class of shape (NumberOfFeatures,NumberOfFeatures)
        pis: Tuple of floats, each float is the pi of each class (you can start with => (Number of samples in each class/total samples)
     '''

    NumberOfFeatures=len(TotalClasses[0][:,0])  #numero de features ex.2
    #Compute the news mu_k y pi_km  ->  mu_k=(1/N_k)* sum_i (Gamma_ik *x_i)   and pi= N_k/ Total num of data  ------------------------------
    mus=[]
    pis=[]
    for clase in range(len(TotalClasses)):
        TamañoClase=len(TotalClasses[clase][0,:])
        pis.append(N_k[clase]/([len(TotalClasses[0][0,:])+len(TotalClasses[1][0,:])]))

        mu_k=np.zeros(shape=(NumberOfFeatures,1))
        for element in range(TamañoClase):   
            current_x=np.reshape(TotalClasses[clase][:,element],(NumberOfFeatures,1))
            gamma_probaility=gamma_k[element,clase]
            #sumo los gamma_probaility sobre los elementos de la clase
            new_mu=(1/N_k[clase])*gamma_probaility*current_x
            mu_k=mu_k+new_mu
        mu_k=np.reshape(mu_k,(NumberOfFeatures,))   #lo regresamos a su forma original ya que gamma function requiere este formato
        mus.append(mu_k)

    # compute the new sigmas ->  sigma_k=(1/N_k)* sum_i (Gamma_ik * (x_i-mu_k) * (x_i-mu_k)^T)   --------------------------
    sigmas=[]
    for clase in range(len(TotalClasses)):
        sigma_k=np.zeros(shape=(NumberOfFeatures,NumberOfFeatures)) 
        TamañoClase=len(TotalClasses[clase][0,:])     
        for element in range(TamañoClase):
            current_x=np.reshape(TotalClasses[clase][:,element],(NumberOfFeatures,1))
            gamma_probaility=gamma_k[element,clase]         
            new_sigma=(1/N_k[clase])*gamma_probaility*(current_x-mus[clase])@np.transpose(current_x-mus[clase])
            sigma_k=sigma_k+new_sigma
        sigmas.append(sigma_k)

    return mus,sigmas,pis


def M_step_optimized(TotalClasses, N_k, gamma_k):
    '''
    This function compute de M step of the EM algorithm 
    args: 
        TotalClasses: List of NDarrays, each array is a class with the data of the class 
        N_k: NDarray of floats, each float is the sum of gamma for each class
        gamma_k: Array of gamma probability of shape (samples,number of classes )
    return: 
        mus: Tuple of NDarrays, each array have the mu of each class of shape (NumberOfFeatures,1)
        sigmas: Tuple of NDarrays, each array have the sigma of each class of shape (NumberOfFeatures,NumberOfFeatures)
        pis: Tuple of floats, each float is the pi of each class (you can start with => (Number of samples in each class/total samples)
    '''
    n_samples, n_classes = gamma_k.shape
    n_features = TotalClasses[0].shape[0]

    # Compute the new mu_k and pi_km
    pis = N_k / n_samples
    denominador = sum(TotalClass.shape[1] for TotalClass in TotalClasses)
    pis /= denominador

    mus = []
    sigmas = []
    for clase in range(n_classes):
        gamma_probaility = gamma_k[:, clase]
        suma_gamma = gamma_probaility.sum()

        # Compute mu_k
        X = TotalClasses[clase]
        mu_k = np.dot(X, gamma_probaility) / suma_gamma
        mus.append(mu_k)

        # Precompute (X - mu_k) and its transpose
        X_mu = X - mu_k.reshape(-1, 1)
        X_mu_T = X_mu.T

        # Compute sigma_k
        sigma_k = np.dot(X_mu, X_mu_T * gamma_probaility.reshape(-1, 1)) / suma_gamma
        sigmas.append(sigma_k)

    return mus, sigmas, pis








#------ EM algorithm----------------
def GaussianMixtureModel(mus,sigmas,pis,TotalClasses,NumberOfSteps):
    '''  
    This function compute de M step of the EM algorithm 
    args: 
        mus: Tuple of NDarrays, each array have the mu of each class of shape (NumberOfFeatures,1)
        sigmas: Tuple of NDarrays, each array have the sigma of each class of shape (NumberOfFeatures,NumberOfFeatures)
        pis: Tuple of floats, each float is the pi of each class (you can start with => (Number of samples in each class/total samples)
        TotalClasses: List of NDarrays, each array is a class with the data of the class
        NumberOfSteps: Number of steps to run the EM algorithm
    return: 
        mus: New Mu's -Tuple of NDarrays, each array have the mu of each class of shape (NumberOfFeatures,1)
        sigmas: New Sigmas's - Tuple of NDarrays, each array have the sigma of each class of shape (NumberOfFeatures,NumberOfFeatures)
        pis: New Pi's -Tuple of floats, each float is the pi of each class (you can start with => (Number of samples in each class/total samples)
    ''' 
    contador=0
    while contador<NumberOfSteps:
        gamma_k,N_k=gamma_fun(TotalClasses,mus,sigmas,pis)

        #si los datos estan balanceados usar la funcion M_step_optimized
        if len(TotalClasses[0][0,:])==len(TotalClasses[1][0,:]):
            mus,sigmas,pis=M_step_optimized(TotalClasses,N_k,gamma_k)
        else:
            mus,sigmas,pis=M_step(TotalClasses,N_k,gamma_k)

        print('calculando step:',contador)
        contador+=1
    return mus,sigmas,pis

#------Inicial values for two classes----------------
def inicial_values(X_Clase_1,X_Clase_2):    #shape of X_Clase_1 and X_Clase_2 is (NumberOfFeatures,NumberOfSamples)
    TotalClasses=[X_Clase_1,X_Clase_2]
    numberfeatures=X_Clase_1.shape[0]
    
    np.random.seed(0)
    rand_mu1=np.random.rand(numberfeatures) #shape of rand_num is (NumberOfFeatures)
    mu_1=rand_mu1
    #mu_1=np.array([(0.1),(0.2)])
    np.random.seed(0)
    rand_mu2=np.random.rand(numberfeatures) #shape of rand_num is (NumberOfFeatures)
    mu_2=rand_mu2
    #mu_2=np.array([(0.3),(0.4)])
    sigmas_1=np.identity(numberfeatures)
    sigmas_2=np.identity(numberfeatures)

    mus=(mu_1, mu_2)
    sigmas=(sigmas_1,sigmas_2)
    pis=(0.3, 0.7)
    return mus,sigmas,pis,TotalClasses


#------Metricas----------------
def precision_jax(y, y_hat):
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


def recall_jax(y, y_hat):
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

    
def accuracy_jax(y, y_hat):
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
        accuracy_cpu = jit(lambda x: x, device=jax.devices("cpu")[0])((TP+TN)/(TP+FP+TN+FN))
        return float(accuracy_cpu)                                              




def MixtureOfGaussians(train_data,y_label, NumberOfSteps=2):
    '''
    This function is the main function of the program
    args:
        train_data: NDarray of shape (NumberOfFeatures,NumberOfSamples)
        y_label: labels sin hot (0,-1 or -1,-1) of the data
        NumberOfSteps: Number of steps to run the EM algorithm default=2
        return:
        mus: New Mu's -Tuple of NDarrays, each array have the mu of each class of shape (NumberOfFeatures,1)
        sigmas: New Sigmas's - Tuple of NDarrays, each array have the sigma of each class of shape (NumberOfFeatures,NumberOfFeatures)
        pis: New Pi's -Tuple of floats, each float is the pi of each class (you can start with => (Number of samples in each class/total samples)
    '''
    #------------------------------------MLFLOW------------------------------------
    with mlflow.start_run(run_name="MixtureOfGaussians") as run:
        mlflow.log_param("NumberOfSteps", NumberOfSteps)
    #--------------------------------------------------------------------------------
        #verificamos los numeros de label de la clase 1 y 2 para separar los datos
        label_1=np.unique(y_label)[0]
        label_2=np.unique(y_label)[1]
        #separamos los datos en dos clases ya que se requiere de dos clases para el algoritmo
        X_Clase_1 = train_data[y_label==label_1]
        X_Clase_2 = train_data[y_label==label_2]

        #transponemos
        X_Clase_1=X_Clase_1.T
        X_Clase_2=X_Clase_2.T

        mus,sigmas,pis,TotalClasses=inicial_values(X_Clase_1,X_Clase_2)
        mus,sigmas,pis=GaussianMixtureModel(mus,sigmas,pis,TotalClasses,NumberOfSteps=NumberOfSteps)

        prediction=predict(train_data, mus, sigmas, pis)
        #metricas


        #converttimos a jax array para calcular metricas
        y_label=jnp.array(y_label)
        prediction=jnp.array(prediction)
        print('y_label:',y_label)
        print('prediction:',prediction)
        print(np.bincount(y_label))
        print(np.bincount(prediction))

        precision=precision_jax(y_label, prediction)
        recall=recall_jax(y_label, prediction)
        accuracy=accuracy_jax(y_label, prediction)

        print('precision:',precision)
        print('recall:',recall)
        print('accuracy:',accuracy)

        #------------------------------------MLFLOW------------------------------------
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", accuracy)
        #--------------------------------------------------------------------------------

        return mus,sigmas,pis,precision,recall,accuracy


#------Plotting the results solo casos de 2D----------------
def gaussianPltFunction(train_data,y_label,mus,sigmas, ):

    #verificamos los numeros de label de la clase 1 y 2 para separar los datos
    label_1=np.unique(y_label)[0]
    label_2=np.unique(y_label)[1]
    #separamos los datos en dos clases ya que se requiere de dos clases para el algoritmo
    X_Clase_1 = train_data[y_label==label_1]
    X_Clase_2 = train_data[y_label==label_2]

    #transponemos
    X_1=X_Clase_1.T
    X_2=X_Clase_2.T




    #-----plot of the samples----------------
    plt.plot(X_1[0,:], X_1[1,:], 'ro') 
    plt.plot(X_2[0,:], X_2[1,:], 'bo') 

    mu_gaussian_1=mus[0]
    mu_gaussian_2=mus[1]
    sigma_gaussian_1=sigmas[0]*2
    sigma_gaussian_2=sigmas[1]*2
    plt.figure(1)
    # Plotting first Gaussian
    m = np.array(mu_gaussian_1)  # defining the mean of the Gaussian 
    cov = np.array(sigma_gaussian_1)   # defining the covariance matrix
    cov_inv = np.linalg.inv(cov)  # inverse of covariance matrix
    cov_det = np.linalg.det(cov)  # determinant of covariance matrix

    x = np.linspace(-4, 4) # defining the x axis from 0 to 1 (because we have normalized the data)
    y = np.linspace(-4, 4)
    X,Y = np.meshgrid(x,y)
    coe = 1.0 / ((2 * np.pi)**2 * cov_det)**0.5
    Z = coe * np.e ** (-0.5 * (cov_inv[0,0]*(X-m[0])**2 + (cov_inv[0,1] + cov_inv[1,0])*(X-m[0])*(Y-m[1]) + cov_inv[1,1]*(Y-m[1])**2))
    plt.contour(X,Y,Z)

    # Plotting second Gaussian
    m = np.array(mu_gaussian_2)  # defining the mean of the Gaussian (mX = 0.2, mY=0.6)
    cov = np.array(sigma_gaussian_2)   # defining the covariance matrix
    cov_inv = np.linalg.inv(cov)  # inverse of covariance matrix
    cov_det = np.linalg.det(cov)  # determinant of covariance matrix

    x = np.linspace(-4, 4)
    y = np.linspace(-4, 4)
    X,Y = np.meshgrid(x,y)
    coe = 1.0 / ((2 * np.pi)**2 * cov_det)**0.5
    Z = coe * np.e ** (-0.5 * (cov_inv[0,0]*(X-m[0])**2 + (cov_inv[0,1] + cov_inv[1,0])*(X-m[0])*(Y-m[1]) + cov_inv[1,1]*(Y-m[1])**2))
    plt.contour(X,Y,Z)

    





def predict(winner_group_class_1, mus, sigmas, pis):
    #number of features
    number_features= winner_group_class_1.shape[1]
    gaussian_values=[]
    prediction=[]
    for i in range(2):
        mu=mus[i]
        sigma=sigmas[i]
        sigma += np.eye(number_features) * 0.01
        determinante_sigma=np.linalg.det(sigma)
        inv_sigma=np.linalg.inv(sigma)
        current_x=winner_group_class_1
        gaussian=(1 / np.sqrt((2 * np.pi) ** number_features * determinante_sigma)) * np.exp(
            -0.5 * np.einsum('ij,ji->i', current_x - mu, np.matmul(inv_sigma, (current_x - mu).T)))
        if i==0:
            gaussian=gaussian
        else:
            gaussian=gaussian
        gaussian_values.append(gaussian)
    print('gaussian_values:',gaussian_values[0])
    print('gaussian_values:',gaussian_values[1])
    prob1=gaussian_values[0]/ (gaussian_values[0]+gaussian_values[1])
    prob2=gaussian_values[1]/ (gaussian_values[0]+gaussian_values[1])

    #creamos vector de prediccion
    for i in range(len(prob1)):
        if prob1[i]>=prob2[i]:
            prediction.append(0)
        else:
            prediction.append(1)
    prediction=np.array(prediction)
    return prediction




if __name__ == "__main__":
    mus,sigmas,pis,precision,recall,accuracy=MixtureOfGaussians(train_data,y_label,NumberOfSteps)
    prediction=predict(winner_group_class_1_trans, mus, sigmas,pis)
    gaussianPltFunction(X_1,X_2,mus,sigmas)