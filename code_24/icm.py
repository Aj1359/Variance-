import random

def simulate_ic_timed_fast(adj, seeds, alpha, n, T):
    """Fast IC simulation returning final informed set."""
    informed = set(seeds)
    frontier = list(seeds)
    for t in range(1, T + 1):
        next_frontier = []
        for v in frontier:
            for u in adj[v]:
                if u not in informed and random.random() < alpha:
                    informed.add(u)
                    next_frontier.append(u)
        frontier = next_frontier
        if not frontier:
            break
    return informed

def simulate_ic_timed(adj, seeds, alpha, n, T):
    """IC simulation tracking activation time per node."""
    informed = set(seeds)
    time_of_inf = [-1] * n
    for s in seeds:
        if 0 <= s < n:
            time_of_inf[s] = 0

    frontier = list(seeds)

    for t in range(1, T + 1):
        next_frontier = []
        for v in frontier:
            for u in adj[v]:
                if u not in informed and random.random() < alpha:
                    informed.add(u)
                    time_of_inf[u] = t
                    next_frontier.append(u)
        frontier = next_frontier
        if not frontier:
            break

    return informed, time_of_inf

def prob_est_timed(adj, seeds, alpha, n, T, R=200):
    """
    Estimate IC activation probabilities and return both probabilities
    and the average variance over individual runs.
    """
    hits = [0] * n
    run_variances = []
    for _ in range(R):
        informed = simulate_ic_timed_fast(adj, seeds, alpha, n, T)
        for v in informed:
            hits[v] += 1
        mu = len(informed) / n
        run_variance = mu * (1.0 - mu)
        run_variances.append(run_variance)
    probs = [hits[v] / R for v in range(n)]
    mean_variance = sum(run_variances) / R
    return probs, mean_variance

def simulate_timeseries(adj, seeds, alpha, n, T, R=200):
    """
    Simulate step-by-step influence cascade from t=0 to t=T.
    Returns average cumulative number of informed nodes at each time step.
    """
    all_ts = []
    for _ in range(R):
        informed, time_of_inf = simulate_ic_timed(adj, seeds, alpha, n, T)
        ts = [0] * (T + 1)
        for v in range(n):
            toi = time_of_inf[v]
            if toi >= 0:
                for step in range(toi, T + 1):
                    ts[step] += 1
        all_ts.append(ts)
    avg_ts = [sum(run[t] for run in all_ts) / R for t in range(T + 1)]
    return avg_ts
