# Task 1.2 — Prompt utilizado (texto exacto)

Desarrolla en Python un simulador de eventos discretos para un hospital con dos variantes de atención: M/M/c, donde los pacientes esperan si todas las camas están ocupadas, y M/M/c/c, donde los pacientes son desviados si no hay camas disponibles.

El simulador debe utilizar una lista de eventos de llegada y salida, con tiempos entre llegadas exponenciales de tasa λ y estancias exponenciales de tasa μ.

Debe aceptar como parámetros λ, μ, c, horizonte de simulación, período de calentamiento y semilla aleatoria.

Para ambas variantes debe reportar: número promedio de pacientes en el sistema (L), número promedio en cola (Lq), tiempo promedio de espera (Wq), utilización de las camas, probabilidad de esperar y probabilidad de bloqueo.

Utiliza los parámetros del Grupo 4: λ = 3.6 pacientes por día, μ = 0.25 por día y c = 17 camas.

Entrega el código completo, comentado y ejecutable, utilizando Python y una lista de eventos ordenada cronológicamente. Incluye un ejemplo de ejecución de ambas variantes.
