import numpy as np
import sympy as sp
import random

# Definir el algoritmo genético
def algoritmo_genetico(funcion_simb, variables, intervalos, nombre_funcion, num_individuos=10, num_generaciones=1000, porcentaje_mutacion=0.1):
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

    # Almacenar resultados de cada generación
    resultados_generacion = []

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

        # Obtener mejores resultados de la generación
        mejor_fitness = np.max(fitness)
        mejor_individuo = poblacion[np.argmax(fitness)]
        resultados_generacion.append(f"Generación {generacion}: Mejor Fitness: {mejor_fitness}, Mejor Individuo: {mejor_individuo}")

    # Devolver mejor solución encontrada
    fitness_final = evaluar_poblacion(poblacion)
    mejor_individuo_final = poblacion[np.argmax(fitness_final)]

    # Guardar resultados en archivo de texto
    with open(f'{nombre_funcion}_resultados.txt', 'w') as f:
        for resultado in resultados_generacion:
            f.write(f"{resultado}\n")
        f.write(f"\nMejor Individuo Final: {mejor_individuo_final}, Mejor Fitness Final: {np.max(fitness_final)}\n")

    return mejor_individuo_final, np.max(fitness_final)

# Listas con funciones, variables y sus intervalos
funciones_simb = [
    3 * sp.Symbol('x1')**3,  # Función 1: 3*x1^3
    (sp.Symbol('x1')**3 / 8) - 2*sp.Symbol('x1')**2 + 3,  # Función 2: x1^3/8 - 2x1^2 + 3
    sp.cos(sp.Symbol('x1') - 1),  # Función 3: cos(x1 - 1)
    sp.exp(sp.sin(sp.Symbol('x1') + sp.Symbol('x2')))  # Función 4: e^(sin(x1 + x2))
]

variables = [
    [sp.Symbol('x1')],  # Variables para la función 1
    [sp.Symbol('x1')],  # Variables para la función 2
    [sp.Symbol('x1')],  # Variables para la función 3
    [sp.Symbol('x1'), sp.Symbol('x2')]  # Variables para la función 4
]

intervalos = [
    [[0.5, 0.8]],  # Intervalos para la función 1
    [[-1.5, 1]],  # Intervalos para la función 2
    [[-7, -3]],  # Intervalos para la función 3
    [[-3, 5], [0, 3]]  # Intervalos para la función 4
]

# Ejecutar el algoritmo genético para cada función
for i, (funcion, vars, inter) in enumerate(zip(funciones_simb, variables, intervalos)):
    print(f"Ejecutando el algoritmo para la función {i + 1}")
    mejor_individuo, mejor_fitness = algoritmo_genetico(funcion, vars, inter, f'funcion_{i+1}')
    print(f"Mejor Individuo: {mejor_individuo}, Mejor Fitness: {mejor_fitness}\n")
