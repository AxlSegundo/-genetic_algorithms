import numpy as np
import sympy as sp
from math import sin,cos, exp
import random

# Definir la función simbólica
x1 = sp.Symbol('x1')
funcion_simb = (x1**3/8) - 2*x1**2 + 3
funcion = sp.lambdify(x1, funcion_simb)

# Parámetros del algoritmo
intervalo_x1 = [-1.5, 1]
num_individuos = 10
num_generaciones = 1500
porcentaje_mutacion = 0.3

# Generar población inicial
def generar_poblacion(num_individuos, intervalo):
    return np.random.uniform(intervalo[0], intervalo[1], num_individuos)

# Evaluar la función para todos los individuos
def evaluar_poblacion(poblacion):
    return np.array([funcion(ind) for ind in poblacion])

# Selección por ruleta
def seleccion_ruleta(poblacion, fitness):
    total_fitness = np.sum(fitness)
    seleccionados = []
    for _ in range(len(poblacion)):
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fit in enumerate(fitness):
            current += fit
            if current > pick:
                seleccionados.append(poblacion[i])
                break
    return np.array(seleccionados)

# Cruzamiento de dos padres
def cruzar(padre1, padre2):
    return (padre1 + padre2) / 2

# Mutación de un individuo
def mutar(individuo, intervalo, porcentaje_mutacion):
    if random.random() < porcentaje_mutacion:
        mutacion = np.random.uniform(-0.1, 0.1)  # pequeña variación
        individuo += mutacion
        # Asegurarse que la mutación mantenga al individuo dentro del intervalo
        individuo = np.clip(individuo, intervalo[0], intervalo[1])
    return individuo

# Algoritmo genético
def algoritmo_genetico():
    poblacion = generar_poblacion(num_individuos, intervalo_x1)

    for generacion in range(num_generaciones):
        fitness = evaluar_poblacion(poblacion)

        # Selección de padres
        padres = seleccion_ruleta(poblacion, fitness)

        # Cruzamiento y creación de nueva población
        nueva_poblacion = []
        for i in range(0, len(padres), 2):
            if i+1 < len(padres):
                hijo1 = cruzar(padres[i], padres[i+1])
                hijo2 = cruzar(padres[i+1], padres[i])
                nueva_poblacion.extend([hijo1, hijo2])
        
        # Mutación
        nueva_poblacion = [mutar(ind, intervalo_x1, porcentaje_mutacion) for ind in nueva_poblacion]

        # Actualizar población
        poblacion = np.array(nueva_poblacion)

        # Imprimir mejores resultados de la generación
        mejor_fitness = np.max(fitness)
        mejor_individuo = poblacion[np.argmax(fitness)]
        print(f"Generación {generacion}: Mejor Fitness: {mejor_fitness}, Mejor Individuo: {mejor_individuo}")

    # Devolver mejor solución encontrada
    fitness_final = evaluar_poblacion(poblacion)
    mejor_individuo_final = poblacion[np.argmax(fitness_final)]
    return mejor_individuo_final, np.max(fitness_final)

# Ejecutar el algoritmo genético
mejor_individuo, mejor_fitness = algoritmo_genetico()
print(f"Mejor Individuo: {mejor_individuo}, Mejor Fitness: {mejor_fitness}")
