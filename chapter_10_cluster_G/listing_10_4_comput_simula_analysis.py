import pandas as pd
import numpy as np

# Load validated simulation parameters
# (output of Listing 10.3)
params = pd.read_csv(
    "out_G_comput_simula_preprocessed.csv",
    encoding='utf-8').iloc[0]
N_AGENTS      = int(params['N_Agents'])
N_STEPS       = int(params['N_Steps'])
INIT_FREQ_A   = float(params['Init_Freq_A'])
PRESTIGE_BIAS = float(params['Prestige_Bias'])
NET_DENSITY   = float(params['Network_Density'])
N_SIMS        = int(params['N_Simulations'])
print(f"Agents: {N_AGENTS}, Steps: {N_STEPS}, "
      f"Simulations: {N_SIMS}")
print(f"Init freq A: {INIT_FREQ_A}, "
      f"Prestige bias: {PRESTIGE_BIAS}, "
      f"Network density: {NET_DENSITY}")

##### 1. Build random network adjacency list
rng = np.random.default_rng(seed=42)

def build_network(n, density, rng):
    """Return dict of neighbor lists for n agents."""
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                adj[i].append(j)
                adj[j].append(i)
    return adj

##### 2. Single simulation run
def run_simulation(n, steps, init_freq,
                   bias, adj, rng):
    """Run one replication; return freq_A at
    each step."""
    # Initialise agent states:
    # 1 = variant A, 0 = variant B
    states = (rng.random(n) < init_freq).astype(int)
    freq_trajectory = [states.mean()]
    for _ in range(steps):
        # Each agent interacts with one random neighbor
        new_states = states.copy()
        for agent in range(n):
            neighbors = adj[agent]
            if not neighbors:
                continue
            neighbor = rng.choice(neighbors)
            # Adoption probability with prestige bias
            adopt_prob = (
                0.5 + (0.5 * bias
                       if states[neighbor] == 1
                       else -0.5 * bias))
            adopt_prob = np.clip(adopt_prob, 0.0, 1.0)
            if rng.random() < adopt_prob:
                new_states[agent] = states[neighbor]
        states = new_states
        freq_trajectory.append(states.mean())
    return freq_trajectory

##### 3. Monte Carlo ensemble
print(f"\nRunning {N_SIMS} simulations...")
adj = build_network(N_AGENTS, NET_DENSITY, rng)
all_trajectories = []
for sim in range(N_SIMS):
    sim_rng = np.random.default_rng(seed=sim)
    traj = run_simulation(
        N_AGENTS, N_STEPS, INIT_FREQ_A,
        PRESTIGE_BIAS, adj, sim_rng)
    all_trajectories.append(traj)

traj_array = np.array(all_trajectories)
mean_traj  = traj_array.mean(axis=0)
lower_traj = np.percentile(traj_array, 5, axis=0)
upper_traj = np.percentile(traj_array, 95, axis=0)

final_freqs      = traj_array[:, -1]
prop_fixation_a  = (final_freqs > 0.95).mean()
prop_fixation_b  = (final_freqs < 0.05).mean()
prop_coexistence = 1 - prop_fixation_a - prop_fixation_b
print(f"Fixation A (>95%): {prop_fixation_a:.3f}")
print(f"Fixation B (<5%):  {prop_fixation_b:.3f}")
print(f"Coexistence:       {prop_coexistence:.3f}")

trajectory_df = pd.DataFrame({
    'Step':      range(N_STEPS + 1),
    'Mean_Freq': mean_traj.round(4),
    'CI_Lower':  lower_traj.round(4),
    'CI_Upper':  upper_traj.round(4),
})

##### 4. Sensitivity analysis: vary prestige bias
# and network density
print("\nRunning sensitivity analysis...")
bias_grid    = [-0.4, -0.2, 0.0, 0.2, 0.4]
density_grid = [0.1, 0.3, 0.5, 0.8]
sensitivity_records = []
N_SENS_SIMS = max(20, N_SIMS // 5)
for bias in bias_grid:
    for density in density_grid:
        adj_s = build_network(
            N_AGENTS, density, rng)
        finals = []
        for sim in range(N_SENS_SIMS):
            sim_rng = np.random.default_rng(
                seed=sim + 1000)
            traj = run_simulation(
                N_AGENTS, N_STEPS, INIT_FREQ_A,
                bias, adj_s, sim_rng)
            finals.append(traj[-1])
        finals = np.array(finals)
        sensitivity_records.append({
            'Prestige_Bias':    bias,
            'Network_Density':  density,
            'Mean_Final_Freq':  round(
                finals.mean(), 4),
            'SD_Final_Freq':    round(
                finals.std(), 4),
            'Prop_Fixation_A':  round(
                (finals > 0.95).mean(), 3),
            'Prop_Fixation_B':  round(
                (finals < 0.05).mean(), 3),
        })
sensitivity_df = pd.DataFrame(sensitivity_records)
print("Sensitivity analysis complete.")
print(sensitivity_df.to_string(index=False))

##### 5. Export results
trajectory_df.to_csv(
    "out_G_comput_simula_trajectories.csv",
    index=False)
pd.DataFrame({
    'Simulation': range(N_SIMS),
    'Final_Freq_A': final_freqs.round(4)
}).to_csv(
    "out_G_comput_simula_final_freqs.csv",
    index=False)
sensitivity_df.to_csv(
    "out_G_comput_simula_sensitivity.csv",
    index=False)
print("\nComputational simulation analysis completed.")
print("Output files:")
print("  out_G_comput_simula_trajectories.csv")
print("  out_G_comput_simula_final_freqs.csv")
print("  out_G_comput_simula_sensitivity.csv")
