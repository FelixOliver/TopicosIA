import numpy as np
import random
import math

def create_layer(n_neuro, n_entradas):#n_entradas = pesos
	layer = []
	for i in range(n_neuro):
		layer.append(np.random.rand(n_entradas))
	return layer

def sigmodea(x):
	return 1/float(1 + pow(math.e, x))

def h_sigmodea(x):
	rpta = []
	for e in x:
		#eps = 1+ pow(sys.float_info.epsilon,-prod(theta, e))
		eps = 1 + pow(math.e,(-1)*e)
		rpta.append(1/float(eps))# 1/1+e**-theta.x
	return rpta

def error(sd,so):
	return np.sum(pow(e1-e2,2) for e1,e2 in zip(sd,so)) / float(2)

def backpropagation(w1,w2,sh,so,sd,x):
	#print w1
	#print w2
	f_w1, c_w1 = w1.shape
	f_w2, c_w2 = w2.shape
	print f_w2,c_w2
	
	print f_w1, c_w1
	print len(sh)

	print w2[1,3]

	err = error(sd,so)
	print "error:",err

	#a_gamma =[]
	while err>0.01:
		sum_delta = []
		for i in range(0,c_w2):
			deltas = []
			for j in range(0,f_w2):
				#print i
				delta = (((-1)*(sd[j]-so[j]))*((1-so[j])*so[j]))
				deltas.append(w2[j][i]*delta)
				#a_gamma.append(errork*w2[j][i])	
				w2[j][i] = w2[j][i] - (0.01 * delta *sh[i])
			sum_delta.append(np.sum(deltas))
		#print "sum delta:  ",sum_delta
		#sum_wgamma = np.sum(a_gamma)
		#print "delta len: ", len(sum_delta)
		for m in range(0,c_w1):
			for n in range(0,f_w1):
				#w_deltas = []
				#for k in range(0,len(so)):
				#	deltajk =  (((-1)*(sd[k]-so[k]))*((1-so[k])*so[k]))
				#	w_deltas.append((w2[k][n+1])* deltajk)
				#sum_wdelta = np.sum(w_deltas)
				w1[n][m] = w1[n][m] - (0.01 * ((sum_delta[n+1]) * ((1-sh[n])*sh[n]) )*x[m])

		sh =  np.array([1]+h_sigmodea(np.inner(x,w1)))
		so =  np.array(h_sigmodea(np.inner(sh,w2)))
		err = error(sd,so)
	print "error - : ",err
	return w1, w2

def backpropagation_solo(w1,w2,sh,so,sd,x):
	#print w1
	#print w2
	f_w1, c_w1 = w1.shape
	f_w2, c_w2 = w2.shape
	
	err = error(sd,so)
	#print "error:",err

	sum_delta = []
	for i in range(0,c_w2):
		deltas = []
		for j in range(0,f_w2):
			#print i
			delta = (((-1)*(sd[j]-so[j]))*((1-so[j])*so[j]))
			deltas.append(w2[j][i]*delta)
			#a_gamma.append(errork*w2[j][i])	
			w2[j][i] = w2[j][i] - (0.5 * delta *sh[i])
		sum_delta.append(np.sum(deltas))
	
	for m in range(0,c_w1):
		for n in range(0,f_w1):
			w1[n][m] = w1[n][m] - (0.5 * ((sum_delta[n+1]) * ((1-sh[n])*sh[n]) )*x[m])
	sh =  np.array([1]+h_sigmodea(np.inner(x,w1)))
	so =  np.array(h_sigmodea(np.inner(sh,w2)))
	err = error(sd,so)
	
	#print "error - : ",err
	return w1, w2,err

def normalizar_vec(y):
	temp=[]
	temp=sorted(y)
	min=temp[0]
	max=temp[len(y)-1]
	for i in range(len(y)):
		y[i]=(y[i]-min)/(max-min)

	return y

def train_iris_solo(entradas):
	n_neuronas = 8
	n_pesos = 5
	w1 = np.array(create_layer(n_neuronas, n_pesos))
	#Sh = np.array([1.000]+h_sigmodea(np.inner(Iris,w1)))
	w2 = np.array(create_layer( 3, 9))
	#So = np.array(h_sigmodea(np.inner(Sh,w2)))
	#error = 10
	umbral = 0.01
	
	sum_error = 10
	while sum_error>umbral:
		vec_error=[]
		for ent in entradas:
			
			#print "ent1 : ",ent
			Sd = [0.000,0.000,0.000]
			clase = int(ent[len(ent)-1])
			#print ent[len(ent)-1]
			#print "clase: ",clase
			ent.remove(ent[len(ent)-1])
			#print ent
			back = ent
			#ent = [1.000]+normalizar_vec(ent)
			#print ent
					
			#Sd[clase] = 0.999
			if(clase==0):Sd = [0.0001,0.0001,0.0001]
			if(clase==1):Sd = [0.5,0.5,0.5]
			if(clase==2):Sd = [0.9999,0.9999,0.9999]
			#print "Sd: ",Sd
			Sh = np.array([1.000]+h_sigmodea(np.inner([1.000]+normalizar_vec(ent) ,w1)))
			#print "Sh: ",Sh
			So = np.array(h_sigmodea(np.inner(Sh,w2)))
			#print "So: ",So
			w1,w2 ,error= backpropagation_solo(w1,w2,Sh,So,Sd,[1.000]+normalizar_vec(ent))

			ent = ent.append(clase)
			#print "ent: ",ent
			

			vec_error.append(error)
		sum_error = np.sum(vec_error)/len(vec_error)
		print "erro sum: ",sum_error
	return w1, w2

def train_iris(entradas):
	n_neuronas = 8
	n_pesos = 5
	w1 = np.array(create_layer(n_neuronas, n_pesos))
	#Sh = np.array([1.000]+h_sigmodea(np.inner(Iris,w1)))
	w2 = np.array(create_layer( 3, 9))
	#So = np.array(h_sigmodea(np.inner(Sh,w2)))

	for ent in entradas:
		print ent
		Sd = [0.0,0.0,0.0]
		clase = ent[len(ent)-1]
		print ent[len(ent)-1]
		print clase
		ent.remove(ent[len(ent)-1])
		print ent
		ent = [1.000]+normalizar_vec(ent)
		print ent

		
		Sd[clase] = 0.999

		Sh = np.array([1.000]+h_sigmodea(np.inner(ent,w1)))
		So = np.array(h_sigmodea(np.inner(Sh,w2)))

		w1,w2 = backpropagation(w1,w2,Sh,So,Sd,ent)
	return w1, w2

def test_clase(entrada_c,w1,w2):
	#E = [8.1,1.2,5.5,5.6] #datos lejanos de prueba
	Eres = [1.000]+normalizar_vec(entrada_c)
	print Eres
	_sh = np.array([1.000]+h_sigmodea(np.inner(Eres,w1)))

	_so = np.array(h_sigmodea(np.inner(_sh,w2)))
	#print "datos que se usaron en el entrenamiento  ", I
	print "datos de prueba E = ",Eres
	print "salida obtenida : ",_so
	#print "salida deseada: ", Sd



##########ejecucion####################
I = [[5.1,3.5,1.4,0.2,0],[4.9,3.0,1.4,0.2,0],[4.7,3.2,1.3,0.2,0],[4.6,3.1,1.5,0.2,0],[5.0,3.6,1.4,0.2,0],
	[7.0,3.2,4.7,1.4,1],[6.4,3.2,4.5,1.5,1],[6.9,3.1,4.9,1.5,1],[5.5,2.3,4.0,1.3,1],[6.5,2.8,4.6,1.5,1],
	[6.3,3.3,6.0,2.5,2],[5.8,2.7,5.1,1.9,2],[7.1,3.0,5.9,2.1,2],[6.3,2.9,5.6,1.8,2],[6.5,3.0,5.8,2.2,2]]


n_neuronas = 8
n_pesos = 5

#I = [1.,2.,4.,3.,6.]
#I = [5.1,3.5,1.4,0.2]
#Iris = [1.000]+normalizar_vec(I)

#print I
#w1 = np.array(create_layer(n_neuronas, n_pesos))
#print w1
#Sh = np.array([1.000]+h_sigmodea(np.inner(Iris,w1)))

#print Sh
#print len(Sh)

#w2 = np.array(create_layer( 3, 9))
#print w2

#So = np.array(h_sigmodea(np.inner(Sh,w2)))
#print So

#Sd = np.array([0.9,0.2,0.3])
#print "w1: ",w1
#w1,w2 = backpropagation(w1,w2,Sh,So,Sd,Iris)
#print "w1 result: ",w1
#####################prueba####################33
#E = [5.2,3.4,1.2,0.1] #datos cercanos a mi IRIS CON EL QUE SE CALCULO
#E = [8.1,1.2,5.5,5.6] #datos lejanos de prueba
#Eres = [1.000]+normalizar_vec(E)
#print Eres
#_sh = np.array([1.000]+h_sigmodea(np.inner(Eres,w1)))

#_so = np.array(h_sigmodea(np.inner(_sh,w2)))
#print "datos que se usaron en el entrenamiento  ", I
#print "datos de prueba E = ",E
#print "error con pesos modificados: ", error(Sd,_so)
#print "salida obtenida : ",_so
#print "salida deseada: ", Sd

#print Eres
#print np.delete(Eres,len(Eres)-1,0)
###########PRUEBA IRIS##################
##  4.9,3.0,1.4,0.2		clase1
##  6.9,3.1,4.9,1.5     clase 2
##  6.3,2.9,5.6,1.8    clase 3
w1,w2 = train_iris_solo(I)
#test_clase([5.1,3.5,1.4,0.2],w1,w2)
test_clase([6.3,2.9,5.6,1.8],w1,w2)
