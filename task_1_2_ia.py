import heapq
import random
from collections import deque


def simulate_hospital(lam, mu, c, horizon, warmup, seed, waiting=True):
    """Simula M/M/c (waiting=True) o M/M/c/c (waiting=False)."""
    if lam <= 0 or mu <= 0 or c < 1 or not (0 <= warmup < horizon):
        raise ValueError('Parametros invalidos')

    rng = random.Random(seed)
    events = []
    counter = 0

    def schedule(t, kind, arrival_time=None):
        nonlocal counter
        heapq.heappush(events, (t, counter, kind, arrival_time))
        counter += 1

    schedule(rng.expovariate(lam), 'arrival')
    t_prev = 0.0
    busy = 0
    queue = deque()
    area_system = 0.0
    area_queue = 0.0
    area_busy = 0.0
    arrivals = 0
    blocked = 0
    waited = 0
    admitted = 0
    wait_sum = 0.0
    wait_count = 0

    while events:
        t, _, kind, arrival_time = heapq.heappop(events)
        end = min(t, horizon)
        left = max(t_prev, warmup)
        if end > left:
            duration = end - left
            area_system += (busy + len(queue)) * duration
            area_queue += len(queue) * duration
            area_busy += busy * duration
        t_prev = end
        if t > horizon:
            break

        if kind == 'arrival':
            schedule(t + rng.expovariate(lam), 'arrival')
            measured = t >= warmup
            if measured:
                arrivals += 1
            if busy < c:
                busy += 1
                if measured:
                    admitted += 1
                    wait_count += 1
                schedule(t + rng.expovariate(mu), 'departure')
            elif waiting:
                queue.append((t, measured))
                if measured:
                    waited += 1
            else:
                if measured:
                    blocked += 1
        else:
            if queue:
                start, measured = queue.popleft()
                if measured:
                    admitted += 1
                    wait_count += 1
                    wait_sum += t - start
                schedule(t + rng.expovariate(mu), 'departure')
            else:
                busy -= 1

    observed = horizon - warmup
    return {
        'L': area_system / observed,
        'Lq': area_queue / observed,
        'Wq_days': wait_sum / wait_count if wait_count else 0.0,
        'Wq_hours': 24 * wait_sum / wait_count if wait_count else 0.0,
        'utilization': area_busy / (c * observed),
        'prob_wait': waited / arrivals if arrivals else 0.0,
        'prob_block': blocked / arrivals if arrivals else 0.0,
        'arrivals': arrivals,
        'admitted': admitted,
        'blocked': blocked,
    }


if __name__ == '__main__':
    params = dict(lam=3.6, mu=0.25, c=17, horizon=1000, warmup=200, seed=42)
    for name, waiting in [('M/M/17', True), ('M/M/17/17', False)]:
        print('\n' + name)
        for metric, value in simulate_hospital(**params, waiting=waiting).items():
            print(f'{metric}: {value:.5f}' if isinstance(value, float) else f'{metric}: {value}')
