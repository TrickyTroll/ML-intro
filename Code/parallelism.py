import time
import numpy as np

# Création de nos matrices
A = np.random.rand(1000,1000)
B = np.random.rand(1000,1000)
C = np.zeros((1000,1000))

# Test de la première implémentation

start_time = time.time()

for i in range(len(A)):
	for j in range(len(B[0])):
		for k in range(len(B)):
		# 	print(A[i][k])
# 			print(B[k][j])
# 			print(i,j)
			C[i][j] += A[i][k] * B[k][j]
			
end_time = time.time()

time_1 = end_time - start_time

# Test de l'implémentation avec Numpy
start_time = time.time()

C = A*B

end_time = time.time()

time_2 = end_time - start_time

print("Run time 1 = {} seconds".format(time_1))
print("Run time numpy = {} seconds".format(time_2))