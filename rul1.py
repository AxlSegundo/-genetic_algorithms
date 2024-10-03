import numpy as np
import sympy as sp
import random

# Definir el algoritmo genético para cualquier función y múltiples variables
def algoritmo_genetico(funcion_simb, variables, intervalos, num_individuos=10, num_generaciones=1500, porcentaje_mutacion=0.3):
    # Crear la función lambda a partir de la función simbólica
    funcion = sp.lambdify(variables, funcion_simb)

    # Generar población inicial para múltiples variables
    def generar_poblacion(num_individuos, intervalos):
        poblacion = []
        for intervalo in intervalos:
            poblacion.append(np.random.uniform(intervalo[0], intervalo[1], num_individuos))
        return np.column_stack(poblacion)

    # Evaluar la función para toda la población
    def evaluar_poblacion(poblacion):
        return np.array([funcion(*ind) for ind in poblacion])

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
    def mutar(individuo, intervalos, porcentaje_mutacion):
        for i in range(len(individuo)):
            if random.random() < porcentaje_mutacion:
                mutacion = np.random.uniform(-0.1, 0.1)  # pequeña variación
                individuo[i] += mutacion
                # Asegurarse que la mutación mantenga al individuo dentro del intervalo
                individuo[i] = np.clip(individuo[i], intervalos[i][0], intervalos[i][1])
        return individuo

    # Iniciar la población
    poblacion = generar_poblacion(num_individuos, intervalos)

    # Ciclo de generaciones
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
        nueva_poblacion = [mutar(ind, intervalos, porcentaje_mutacion) for ind in nueva_poblacion]

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

# Ejemplos de uso con diferentes funciones y variables

# 1. Función 3*x1^3 , x1[0.5, 0.8]
x1 = sp.Symbol('x1')
funcion1 = 3*x1**3
intervalos1 = [[0.5, 0.8]]

# 2. Función x1^3/8 - 2x1^2 + 3 , x1[-1.5, 1]
funcion2 = (x1**3 / 8) - 2*x1**2 + 3
intervalos2 = [[-1.5, 1]]

# 3. Función cos(x1 - 1) , x1[-7, -3]
funcion3 = sp.cos(x1 - 1)
intervalos3 = [[-7, -3]]

# 4. Función e^(sin(x1 + x2)) , x1[-3,5], x2[0,3]
x2 = sp.Symbol('x2')
funcion4 = sp.exp(sp.sin(x1 + x2))
intervalos4 = [[-3, 5], [0, 3]]

# Ejecutar el algoritmo con cada función
print("Ejemplo 1:")
mejor_individuo1, mejor_fitness1 = algoritmo_genetico(funcion1, [x1], intervalos1)
print(f"Mejor Individuo: {mejor_individuo1}, Mejor Fitness: {mejor_fitness1}\n")

print("Ejemplo 2:")
mejor_individuo2, mejor_fitness2 = algoritmo_genetico(funcion2, [x1], intervalos2)
print(f"Mejor Individuo: {mejor_individuo2}, Mejor Fitness: {mejor_fitness2}\n")

print("Ejemplo 3:")
mejor_individuo3, mejor_fitness3 = algoritmo_genetico(funcion3, [x1], intervalos3)
print(f"Mejor Individuo: {mejor_individuo3}, Mejor Fitness: {mejor_fitness3}\n")

print("Ejemplo 4:")
mejor_individuo4, mejor_fitness4 = algoritmo_genetico(funcion4, [x1, x2], intervalos4)
print(f"Mejor Individuo: {mejor_individuo4}, Mejor Fitness: {mejor_fitness4}\n")
