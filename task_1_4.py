"""Task 1.4 – Validación del simulador corregido contra la teoría del Task 1.1 (grupo 4).

(a) Justificación del calentamiento y del horizonte (promedio móvil de la fila, procedimiento de
    Welch) y 30 réplicas independientes por variante con IC de 95 % frente a la teoría.
(b) Ley de Little con las cantidades de la SIMULACIÓN (independiente de la teoría).
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from task_1_3_corregido import simulate_hospital, simulate_replicas

# ---------------- Parámetros del grupo 4 (Task 1.1) ----------------
g = 4
mu = 0.25
lam = 3.0 + 0.2 * ((g - 1) % 5)        # λ = 3.6
c = 17 + ((g - 1) % 3)                 # c = 17
a = lam / mu                           # a = λ/μ = 14.4
rho = a / c


def erlang_b(c, a):
    B = 1.0                            # B(0)=1 ; B(k) = a B(k-1) / (k + a B(k-1))
    for k in range(1, c + 1):
        B = a * B / (k + a * B)
    return B


def teoria():
    """Valores teóricos del Task 1.1 (fórmulas de Erlang B y C)."""
    B = erlang_b(c, a)
    C = B / (1 - rho * (1 - B))                    # Erlang C a partir de Erlang B
    Lq = C * rho / (1 - rho)
    Wq = Lq / lam                                  # días
    W = Wq + 1 / mu
    con = {'L': Lq + a, 'Lq': Lq, 'Wq_hours': 24 * Wq, 'W_hours': 24 * W,
           'utilization': rho, 'prob_wait': C, 'prob_block': 0.0, 'lambda_ef': lam}
    sin = {'L': a * (1 - B), 'Lq': 0.0, 'Wq_hours': 0.0, 'W_hours': 24 / mu,
           'utilization': a * (1 - B) / c, 'prob_wait': 0.0, 'prob_block': B,
           'lambda_ef': lam * (1 - B)}
    return con, sin


# ---------------- (a) Calentamiento: promedio móvil (Welch) ----------------
def welch(waiting, n_rep=2000, largo=600, ventana=50, semilla0=90000):
    """Trayectoria muestreada cada día desde un sistema VACÍO, promediada entre n_rep réplicas (2000: la fila con ρ = 0.85 está muy autocorrelacionada y con pocas réplicas la curva es puro ruido)
    (para quitar el ruido) y suavizada con un promedio móvil de `ventana` días."""
    corridas = []
    for i in range(n_rep):
        tr = simulate_hospital(lam, mu, c, largo, 0, semilla0 + i, waiting, sample_every=1.0)['trayectoria']
        # con espera: longitud de la fila Nq ; sin espera no hay fila -> camas ocupadas N
        corridas.append([(nq if waiting else ns) for (_, nq, ns) in tr][:largo])
    media = np.mean(corridas, axis=0)
    movil = np.convolve(media, np.ones(ventana) / ventana, mode='valid')
    return media, movil


def dia_estable(movil, ventana, tolerancia=0.10):
    """Día (centro de la ventana) desde el cual el promedio móvil se queda dentro de
    ±tolerancia del valor de meseta (media de la segunda mitad de la curva)."""
    meseta = movil[len(movil) // 2:].mean()
    fuera = np.where(np.abs(movil - meseta) > tolerancia * meseta)[0]
    return ((fuera.max() + 1 if len(fuera) else 0) + ventana // 2), meseta


def grafico_calentamiento(ruta, warmup):
    con, sin = teoria()
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    info = {}
    for ax, waiting, titulo, teo, ylab in (
            (ejes[0], True, 'Variante con espera (M/M/17)', con['Lq'], 'Pacientes en la fila (Lq)'),
            (ejes[1], False, 'Variante sin espera (M/M/17/17)', sin['L'], 'Camas ocupadas (no hay fila)')):
        media, movil = welch(waiting)
        ventana = len(media) - len(movil) + 1
        dia, meseta = dia_estable(movil, ventana)
        ax.plot(np.arange(len(media)), media, color='0.8', lw=0.8, label='promedio de 2000 réplicas (por día)')
        ax.plot(np.arange(len(movil)) + ventana / 2, movil, color='C0', lw=2, label=f'promedio móvil ({ventana} d)')
        ax.axhline(teo, color='C3', ls='--', lw=1, label=f'teoría = {teo:.2f}')
        ax.axvspan(0, warmup, color='C1', alpha=0.15, label=f'calentamiento = {warmup} d')
        ax.set_title(titulo); ax.set_xlabel('día'); ax.set_ylabel(ylab)
        ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)
        info['con' if waiting else 'sin'] = {'dia_estable': int(dia), 'meseta': float(meseta)}
    fig.tight_layout()
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return info


# ---------------- (a) 30 réplicas vs teoría y (b) Little ----------------
def validar(waiting, n, horizon, warmup, master_seed):
    resumen, corridas = simulate_replicas(n, master_seed, lam=lam, mu=mu, c=c, horizon=horizon,
                                          warmup=warmup, waiting=waiting)
    # (b) Little con las cantidades simuladas: L = λ_ef·W y (con espera) Lq = λ_ef·Wq,
    # por réplica; error relativo = |L - λ_ef·W| / L.
    # Wq y W en días: Wq_hours / 24.
    err_L = [abs(r['L'] - r['lambda_ef'] * r['W_days']) / r['L'] for r in corridas]
    err_Lq = ([abs(r['Lq'] - r['lambda_ef'] * r['Wq_days']) / r['Lq'] for r in corridas]
              if waiting else None)
    L_m, lamef_m, W_m = (np.mean([r['L'] for r in corridas]), np.mean([r['lambda_ef'] for r in corridas]),
                         np.mean([r['W_days'] for r in corridas]))
    little = {'L_sim': L_m, 'lambda_ef_sim': lamef_m, 'W_dias_sim': W_m,
              'L_por_little': lamef_m * W_m,
              'error_rel_medias': abs(L_m - lamef_m * W_m) / L_m,
              'error_rel_replica_media': float(np.mean(err_L)),
              'error_rel_replica_max': float(np.max(err_L))}
    if waiting:
        Lq_m, Wq_m = np.mean([r['Lq'] for r in corridas]), np.mean([r['Wq_days'] for r in corridas])
        little.update({'Lq_sim': Lq_m, 'Wq_dias_sim': Wq_m, 'Lq_por_little': lamef_m * Wq_m,
                       'error_rel_medias_Lq': abs(Lq_m - lamef_m * Wq_m) / Lq_m,
                       'error_rel_replica_media_Lq': float(np.mean(err_Lq)),
                       'error_rel_replica_max_Lq': float(np.max(err_Lq))})
    return resumen, little


if __name__ == '__main__':
    WARMUP, HORIZON, N = 100, 3000, 30
    con, sin = teoria()
    info = grafico_calentamiento('figuras/task_1_4_calentamiento.png', WARMUP)
    print('Día en que el promedio móvil queda dentro de ±10 % de su meseta:', info)

    salida = {'parametros': dict(lam=lam, mu=mu, c=c, horizon=HORIZON, warmup=WARMUP, n=N), 'welch': info}
    for nombre, waiting, teo in (('con_espera', True, con), ('sin_espera', False, sin)):
        resumen, little = validar(waiting, N, HORIZON, WARMUP, master_seed=2026)
        print('\n=== ' + nombre + f'  ({N} réplicas, horizonte {HORIZON} d, calentamiento {WARMUP} d) ===')
        filas = {}
        for m, (media, h, s) in resumen.items():
            dentro = abs(media - teo[m]) <= h
            filas[m] = {'media': media, 'semiancho': h, 'teoria': teo[m], 'dentro': bool(dentro)}
            print(f'  {m:12s} sim = {media:9.4f} ± {h:7.4f}   teoría = {teo[m]:9.4f}   '
                  f'{"dentro" if dentro else "FUERA"}')
        print('  Little:', {k: round(v, 5) for k, v in little.items()})
        salida[nombre] = {'medidas': filas, 'little': little}

    with open('resultados_task_1_4.json', 'w') as f:
        json.dump(salida, f, indent=2)
