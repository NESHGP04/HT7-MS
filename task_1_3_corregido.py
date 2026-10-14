"""Simulador de eventos discretos del hospital (Task 1.3 – versión corregida).

Parte del código original de la IA (task_1_2_ia.py, que NO se modifica). El motor de eventos
ya era correcto (pasó las pruebas de respuesta conocida); lo que se corrige es la capa
estadística. Cada cambio está marcado con  [CORRECCIÓN (n)]  donde n es el punto de la
lista de auditoría (i)–(vii) del Task 1.3.
"""
import heapq
import math
import random
from collections import deque

from scipy import stats


def simulate_hospital(lam, mu, c, horizon, warmup, seed, waiting=True, sample_every=None):
    """Una réplica. Devuelve las métricas de la ventana de observación [warmup, horizon].

    sample_every: si se da (en días), también devuelve la trayectoria (t, en_cola, en_sistema)
    muestreada desde t = 0, para justificar el calentamiento con un promedio móvil.
    """
    if lam <= 0 or mu <= 0 or c < 1 or not (0 <= warmup < horizon):
        raise ValueError('Parametros invalidos')

    rng = random.Random(seed)
    events = []          # lista de eventos futuros (heap ordenado por tiempo)
    counter = 0          # desempata eventos con el mismo tiempo

    def schedule(t, kind, info=None):
        # info = (t_llegada, medido) del paciente que ocupa la cama; sirve para medir W en la salida
        nonlocal counter
        heapq.heappush(events, (t, counter, kind, info))
        counter += 1

    schedule(rng.expovariate(lam), 'arrival')
    t_prev = 0.0
    busy = 0                       # camas ocupadas
    queue = deque()                # (t_llegada, medido) de quienes esperan
    area_system = 0.0              # integral de N(t) dt   (N = pacientes en el sistema)
    area_queue = 0.0               # integral de Nq(t) dt  (Nq = pacientes en cola)
    area_busy = 0.0                # integral de camas_ocupadas(t) dt
    arrivals = 0
    blocked = 0
    waited = 0
    admitted = 0
    wait_sum = 0.0
    wait_count = 0
    sojourn_sum = 0.0              # suma de tiempos en el sistema (llegada -> salida) de los medidos
    sojourn_count = 0
    trayectoria = []
    next_sample = 0.0

    while events:
        t, _, kind, info = heapq.heappop(events)
        end = min(t, horizon)
        left = max(t_prev, warmup)
        # (i) Promedio PONDERADO POR TIEMPO: L = (1/T) ∫ N(t) dt. El estado vigente se
        # multiplica por la duración del tramo [left, end]; solo cuenta la ventana medida.
        if end > left:
            duration = end - left
            area_system += (busy + len(queue)) * duration
            area_queue += len(queue) * duration
            area_busy += busy * duration
        # Muestreo de la trayectoria (incluye el calentamiento) para el promedio móvil.
        if sample_every:
            while next_sample <= end:
                trayectoria.append((next_sample, len(queue), busy + len(queue)))
                next_sample += sample_every
        t_prev = end

        # [CORRECCIÓN (vi)] El original hacía `break` al pasar el horizonte y descartaba a los
        # pacientes que seguían en cola: sus esperas largas nunca entraban en Wq (censura) y
        # P(esperar) y Wq tenían denominadores distintos. Ahora no se aceptan llegadas nuevas
        # después del horizonte, pero se deja terminar a los que ya llegaron, de modo que
        # TODO paciente medido tiene su espera completa. Las áreas siguen cortadas en `horizon`.
        if kind == 'arrival' and t > horizon:
            continue

        if kind == 'arrival':
            schedule(t + rng.expovariate(lam), 'arrival')
            measured = t >= warmup          # se mide por hora de LLEGADA (después del calentamiento)
            if measured:
                arrivals += 1
            if busy < c:
                busy += 1
                if measured:
                    admitted += 1
                    wait_count += 1         # espera 0 (entra directo): también cuenta para Wq
                schedule(t + rng.expovariate(mu), 'departure', (t, measured))
            elif waiting:
                queue.append((t, measured))
                if measured:
                    waited += 1             # P(esperar) = fracción que llega y no halla cama
                    admitted += 1
            else:
                if measured:
                    blocked += 1            # M/M/c/c: se pierde (B(c,a) en teoría)
        else:
            # W = tiempo en el sistema (espera + estancia); se mide en la salida, desde la LLEGADA.
            if info[1]:
                sojourn_sum += t - info[0]
                sojourn_count += 1
            if queue:
                # (vii) Wq se mide desde la LLEGADA (t_llegada), no desde el inicio del servicio.
                t_llegada, measured = queue.popleft()
                if measured:
                    wait_count += 1
                    wait_sum += t - t_llegada
                schedule(t + rng.expovariate(mu), 'departure', (t_llegada, measured))
            else:
                busy -= 1

    observed = horizon - warmup
    nan = float('nan')
    resultado = {
        'L': area_system / observed,
        'Lq': area_queue / observed,
        # [CORRECCIÓN] sin datos se devuelve NaN, no 0.0 (un 0.0 inventado se vería como válido).
        'Wq_days': wait_sum / wait_count if wait_count else nan,
        'Wq_hours': 24 * wait_sum / wait_count if wait_count else nan,
        # W = tiempo medio en el sistema de los pacientes medidos (todos terminan: no hay censura)
        'W_days': sojourn_sum / sojourn_count if sojourn_count else nan,
        'W_hours': 24 * sojourn_sum / sojourn_count if sojourn_count else nan,
        'utilization': area_busy / (c * observed),
        'prob_wait': waited / arrivals if arrivals else nan,
        'prob_block': blocked / arrivals if arrivals else nan,
        # [CORRECCIÓN (v)] tasa EFECTIVA de ingresos: λ_ef = λ(1 - B). Es la que va en la
        # ley de Little (L = λ_ef W); usar λ sobreestimaría L en la variante sin espera.
        'lambda_ef': admitted / observed,
        'arrivals': arrivals,
        'admitted': admitted,
        'blocked': blocked,
    }
    if sample_every:
        resultado['trayectoria'] = trayectoria
    return resultado


METRICAS = ('L', 'Lq', 'Wq_hours', 'W_hours', 'utilization', 'prob_wait', 'prob_block', 'lambda_ef')


def simulate_replicas(n, master_seed, confidence=0.95, **params):
    """[CORRECCIÓN (ii) y (iv)] n réplicas independientes y IC por réplica.

    (iv) Las semillas se derivan de UN generador maestro (getrandbits), así son distintas y
         reproducibles; no se reutiliza ni se encadena la misma semilla.
    (ii) La unidad de muestreo es la RÉPLICA: cada réplica aporta un solo número por métrica
         (el promedio de toda su corrida). Los pacientes dentro de una réplica están
         correlacionados, por lo que NO se usan como muestra independiente.
         IC:  x̄ ± t_{1-α/2, n-1} · s/√n
    """
    maestro = random.Random(master_seed)
    seeds = []
    while len(seeds) < n:
        s = maestro.getrandbits(63)
        if s not in seeds:
            seeds.append(s)
    corridas = [simulate_hospital(seed=s, **params) for s in seeds]

    tcrit = stats.t.ppf(1 - (1 - confidence) / 2, n - 1)
    resumen = {}
    for m in METRICAS:
        xs = [r[m] for r in corridas if not math.isnan(r[m])]
        if len(xs) < 2:
            resumen[m] = (float('nan'), float('nan'), float('nan'))
            continue
        media = sum(xs) / len(xs)
        s = math.sqrt(sum((x - media) ** 2 for x in xs) / (len(xs) - 1))
        resumen[m] = (media, tcrit * s / math.sqrt(len(xs)), s)   # (media, semi-ancho, desv. est.)
    return resumen, corridas


if __name__ == '__main__':
    params = dict(lam=3.6, mu=0.25, c=17, horizon=1000, warmup=200)
    for name, waiting in [('M/M/17', True), ('M/M/17/17', False)]:
        resumen, _ = simulate_replicas(30, master_seed=2026, waiting=waiting, **params)
        print('\n' + name + '  (30 réplicas, IC 95 %)')
        for m, (media, h, _) in resumen.items():
            print(f'  {m:12s} {media:10.4f} ± {h:.4f}')
