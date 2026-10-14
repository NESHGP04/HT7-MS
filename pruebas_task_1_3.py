"""Pruebas de auditoría del Task 1.3: se corren contra el código ORIGINAL de la IA (sin tocarlo)
y contra la versión corregida. Cada prueba tiene una respuesta conocida calculable a mano."""
import math
import statistics as st
from unittest import mock

import task_1_2_ia as original
import task_1_3_corregido as corregido


class RngDeterminista:
    """Sustituye a random.Random: expovariate(tasa) = 1/tasa exactamente (sin azar).
    Convierte el simulador en un sistema D/D/c con resultados calculables a mano."""
    def __init__(self, seed): pass
    def expovariate(self, tasa): return 1.0 / tasa


def correr_det(modulo, **kw):
    with mock.patch.object(modulo.random, 'Random', RngDeterminista):
        return modulo.simulate_hospital(**kw)


def ok(cond): return 'OK ' if cond else 'FALLA'


print('=' * 78)
print('PRUEBA (i) D/D/1: llegada cada 3 d, estancia 1 d, 1 cama  ->  L = 1/3 (a mano)')
print('  Un promedio sobre eventos daría 0.5 (estados 1,0,1,0,...); el ponderado por tiempo, 1/3.')
for nombre, mod in (('original', original), ('corregido', corregido)):
    r = correr_det(mod, lam=1/3, mu=1.0, c=1, horizon=300, warmup=30, seed=0)
    print(f"  {nombre:10s} L={r['L']:.6f} util={r['utilization']:.6f}  esperado 0.333333  [{ok(abs(r['L']-1/3)<1e-9)}]")

print('=' * 78)
print('PRUEBAS (vi) y (vii) D/D/1 sobrecargado: llegada cada 1 d, estancia 1.5 d, 1 cama, T = 10 d')
print('  Espera del paciente n (desde su LLEGADA) = 0.5(n-1) d  ->  los 10 pacientes: 0, 0.5, ..., 4.5')
print('  Wq verdadero de los 10 = 2.25 d; los primeros 7 que alcanzan a entrar a servicio: 1.5 d.')
for nombre, mod in (('original', original), ('corregido', corregido)):
    r = correr_det(mod, lam=1.0, mu=1/1.5, c=1, horizon=10, warmup=0, seed=0)
    print(f"  {nombre:10s} llegadas={r['arrivals']} atendidos={r['admitted']} "
          f"P(esperar)={r['prob_wait']:.2f} Wq={r['Wq_days']:.4f} d")
print('  -> original: Wq=1.5 promedia solo 7 de 10 (descarta a los 3 que siguen en cola); '
      'P(esperar)=0.9 sí los cuenta.')
print('     Que 1.5 = promedio(0..3) confirma que mide desde la LLEGADA  [punto (vii) cumple].')

print('=' * 78)
print('PRUEBA (iv) reproducibilidad: misma semilla vs semilla distinta (original)')
p = dict(lam=3.6, mu=0.25, c=17, horizon=1000, warmup=200)
a = original.simulate_hospital(seed=42, **p); b = original.simulate_hospital(seed=42, **p)
d = original.simulate_hospital(seed=43, **p)
print(f"  misma semilla idéntica: {a == b}   semilla distinta idéntica: {a == d}")
print('  -> si se reusa la semilla, las "réplicas" serían copias y el IC tendría ancho 0.')

print('=' * 78)
print('PRUEBA (ii) variabilidad entre réplicas del ORIGINAL (30 semillas, grupo 4, con espera)')
lqs = [original.simulate_hospital(seed=1000 + i, waiting=True, **p)['Lq'] for i in range(30)]
print(f"  Lq por corrida: mín={min(lqs):.3f} máx={max(lqs):.3f} media={st.mean(lqs):.3f}  "
      f"(la corrida de la IA con semilla 42 dio {a['Lq']:.3f}; teoría 2.281)")
print('  -> una sola corrida no dice si 1.12 es error o azar; el original no entrega ningún IC.')

print('=' * 78)
print('PRUEBA (v) M/M/1/1 con lam=1, mu=1 (a=1): B=a/(1+a)=0.5, lambda_ef=0.5, L=0.5')
r = original.simulate_hospital(lam=1, mu=1, c=1, horizon=20000, warmup=200, seed=7, waiting=False)
lam_ef = r['admitted'] / (20000 - 200)
print(f"  original: L={r['L']:.4f}  lambda_ef(admitted/T)={lam_ef:.4f}  "
      f"claves devueltas incluyen 'lambda_ef': {'lambda_ef' in r}")
print(f"  Little con lambda_ef: lam_ef*W = {lam_ef:.4f}*1 = {lam_ef:.4f} ≈ L   |  con lambda: 1*1 = 1.0000 ≠ L")

print('=' * 78)
print('PRUEBA (iii) calentamiento: trayectoria media del número en cola desde un sistema VACÍO')
print('  (corregido, 100 réplicas, grupo 4 con espera; promedio de Nq por ventanas de 25 días)')
tray = [corregido.simulate_hospital(3.6, 0.25, 17, 400, 0, 5000 + i, True, sample_every=1.0)['trayectoria']
        for i in range(100)]
for ini in range(0, 400, 25):
    vals = [nq for t in tray for (tt, nq, _) in t if ini <= tt < ini + 25 for nq in [nq]]
    print(f"  días {ini:3d}-{ini+25:3d}: Lq medio = {st.mean(vals):.3f}")
print('  (teoría estacionaria: Lq = 2.281)')

print('=' * 78)
print('VALIDACIÓN del corregido con respuestas conocidas (30 réplicas, IC 95 %)')
res, _ = corregido.simulate_replicas(30, 11, lam=1, mu=1, c=1, horizon=5000, warmup=100, waiting=False)
m, h, _ = res['prob_block']; print(f"  M/M/1/1 a=1  P(bloqueo) = {m:.4f} ± {h:.4f}  teoría 0.5  [{ok(abs(m-0.5)<=h)}]")
m, h, _ = res['L'];          print(f"  M/M/1/1 a=1  L          = {m:.4f} ± {h:.4f}  teoría 0.5  [{ok(abs(m-0.5)<=h)}]")
res, _ = corregido.simulate_replicas(30, 12, lam=0.5, mu=1, c=1, horizon=5000, warmup=100, waiting=True)
for k, t in (('L', 1.0), ('Lq', 0.5), ('Wq_hours', 24.0), ('prob_wait', 0.5), ('utilization', 0.5)):
    m, h, _ = res[k]
    print(f"  M/M/1 ρ=0.5  {k:11s}= {m:.4f} ± {h:.4f}  teoría {t}  [{ok(abs(m-t)<=h)}]")
