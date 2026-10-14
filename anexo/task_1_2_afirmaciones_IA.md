# Task 1.2(b) — Afirmaciones de la IA (copiadas literalmente)

## Explicación que acompañó al código

Utilizaremos heapq para gestionar la lista de eventos, random para generar los tiempos exponenciales y deque para representar la fila de espera. No necesitamos SimPy para esta primera implementación.
El horizonte y el calentamiento serán configurables. Para el ejemplo inicial usaremos 1,000 días de simulación y 200 días de calentamiento; estos valores son provisionales y deberán justificarse posteriormente mediante las pruebas del Task 1.4.

## Resultados de la ejecución inicial

Ejecuté el código con la semilla 42 y estos fueron los resultados obtenidos:

| Métrica | Con espera | Sin espera |
|---|---|---|
| L (pacientes) | 14.9454 | 12.8676 |
| Lq (pacientes) | 1.1164 | 0.0000 |
| Wq (horas) | 7.6035 | 0.0000 |
| Utilización | 81.35 % | 75.69 % |
| Probabilidad de esperar | 29.09 % | 0 % |
| Probabilidad de bloqueo | 0 % | 8.96 % |

Estos resultados corresponden a una sola realización, no a una validación estadística.

## Afirmaciones del agente de IA sobre el simulador original

1. El código se ejecutó sin errores con los parámetros del Grupo 4, utilizando Python.
2. Ambas variantes generaron resultados numéricos para las métricas solicitadas.
3. En la ejecución inicial, la variante M/M/17/17 reportó una probabilidad de bloqueo de 8.96 %, mientras que la variante M/M/17 reportó una probabilidad de espera de 29.09 %.
4. No se ha demostrado todavía que los resultados coincidan con la teoría de Erlang ni que el simulador cumpla los siete criterios de auditoría del Task 1.3.
