import time
import numpy as np

# Création du tableau et initiation de la somme.
tableau = np.random.rand(1000000000)
somme = 0

# Test de la première implémentation.

start_time = time.time()

# Pour chaque chiffre dans le tableau.
for chiffre in tableau:
    # Sommation de l'ancienne somme avec le nouveau chiffre.
    somme += chiffre
			
end_time = time.time()

time_1 = end_time - start_time

# Test de l'implémentation avec Numpy.

# Réinitialisation de la somme.
somme = 0

start_time = time.time()

somme += np.sum(tableau)

end_time = time.time()

time_2 = end_time - start_time

print("Run time 1 = {} seconds".format(time_1))
print("Run time numpy = {} seconds".format(time_2))