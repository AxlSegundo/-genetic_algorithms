import numpy as np
import sympy as sp
import random

# Definir el algoritmo genético con selección elitista
def algoritmo_genetico_elitista(funcion_simb, variables, intervalos, nombre_funcion, restricciones=None, num_individuos=10, num_generaciones=1500, porcentaje_mutacion=0.1, porcentaje_elitismo=0.2):
    # Crear la función lambda a partir de la función simbólica
    funcion = sp.lambdify(variables, funcion_simb)

    # Generar población inicial para múltiples variables
    def generar_poblacion(num_individuos, intervalos):
        poblacion = []
        for intervalo in intervalos:
            poblacion.append(np.random.uniform(intervalo[0], intervalo[1], num_individuos))
        return np.column_stack(poblacion)

    # Evaluar la función y verificar restricciones
    def evaluar_poblacion(poblacion):
        fitness = []
        for individuo in poblacion:
            valor_funcion = funcion(*individuo)
            # Verificar si se cumple la restricción (penalizar si no se cumple)
            if restricciones:
                cumple_restricciones = True
                for restriccion in restricciones:
                    restriccion_func = sp.lambdify(variables, restriccion)
                    if not restriccion_func(*individuo):
                        cumple_restricciones = False
                        break
                if not cumple_restricciones:
                    valor_funcion -= 100  # Penalización si no cumple la restricción
            fitness.append(valor_funcion)
        return np.array(fitness)

    # Selección elitista
    def seleccion_elitista(poblacion, fitness, porcentaje_elitismo):
        num_elite = int(len(poblacion) * porcentaje_elitismo)
        elite_indices = np.argsort(fitness)[-num_elite:]  # Selecciona los mejores
        return poblacion[elite_indices]

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

        # Validar que haya valores en fitness
        if np.all(fitness <= 0):  # Si todos los valores son negativos o cero
            print(f"Advertencia: Todos los individuos están penalizados en la generación {generacion}.")
            mejor_individuo = poblacion[np.random.randint(len(poblacion))]  # Seleccionar aleatoriamente uno
            mejor_fitness = np.min(fitness)  # Tomar el mínimo (o cualquier valor de fitness)
            break

        # Selección elitista de los mejores padres
        padres = seleccion_elitista(poblacion, fitness, porcentaje_elitismo)

        # Cruzamiento y creación de nueva población
        nueva_poblacion = []
        while len(nueva_poblacion) < len(poblacion):
            padre1, padre2 = random.sample(list(padres), 2)
            hijo1 = cruzar(padre1, padre2)
            hijo2 = cruzar(padre2, padre1)
            nueva_poblacion.extend([hijo1, hijo2])
        
        # Truncar si excede el tamaño original
        nueva_poblacion = nueva_poblacion[:num_individuos]

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
    with open(f'MayorFit/{nombre_funcion}_resultados.txt', 'w') as f:
        for resultado in resultados_generacion:
            f.write(f"{resultado}\n")
        f.write(f"\nMejor Individuo Final: {mejor_individuo_final}, Mejor Fitness Final: {np.max(fitness_final)}\n")

    return mejor_individuo_final, np.max(fitness_final)
# Listas con funciones, variables, intervalos y restricciones
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

restricciones = [
    [sp.Symbol('x1') > 0],  # Restricción para la función 1
    [sp.Symbol('x1') > 0],  # Sin restricción para la función 2
    [sp.Symbol('x1') < 0],  # Restricción para la función 3
    [sp.Symbol('x1') + sp.Symbol('x2') > 0]  # Restricción para la función 4
]

# Ejecutar el algoritmo genético para cada función
for i, (funcion, vars, inter, restr) in enumerate(zip(funciones_simb, variables, intervalos, restricciones)):
    print(f"Ejecutando el algoritmo para la función {i + 1}")
    mejor_individuo, mejor_fitness = algoritmo_genetico_elitista(funcion, vars, inter, f'funcion_{i+1}', restricciones=restr)
    print(f"Mejor Individuo: {mejor_individuo}, Mejor Fitness: {mejor_fitness}\n")