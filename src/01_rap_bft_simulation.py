import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_bridging_score

# --- CONFIGURATION ---
N_AGENTS = 1000
STEPS = 100
MALICIOUS_RATIO = 0.40  # 40% Bots (Beyond standard BFT threshold) 
EPSILON = 0.2           # Bounded Confidence

class RAPSimulation:
    def __init__(self):
        # Init Beliefs (Random 0-1)
        self.beliefs = np.random.rand(N_AGENTS)
        
        # Malicious Nodes Setup
        n_malicious = int(N_AGENTS * MALICIOUS_RATIO)
        self.malicious_mask = np.zeros(N_AGENTS, dtype=bool)
        self.malicious_mask[:n_malicious] = True
        # Bots are polarized extremes
        self.beliefs[:n_malicious // 2] = 0.0
        self.beliefs[n_malicious // 2 : n_malicious] = 1.0
        
    def step(self, bridging_active=True):
        new_beliefs = self.beliefs.copy()
        
        for i in range(N_AGENTS):
            if self.malicious_mask[i]: continue # Bots don't update
            
            # Identify neighbors within confidence bound (epsilon)
            distances = np.abs(self.beliefs - self.beliefs[i])
            neighbors = distances < EPSILON
            
            if bridging_active:
                # Bridging Logic: Penalize influence from polarized extremes
                # Reward neighbors closer to the "bridging" center (0.5)
                weights = 1.0 - np.abs(self.beliefs[neighbors] - 0.5)
            else:
                # Standard Engagement: All loud voices have equal weight
                weights = np.ones(np.sum(neighbors))
                
            # Update belief
            if np.sum(weights) > 0:
                new_beliefs[i] = np.average(self.beliefs[neighbors], weights=weights)
                
        self.beliefs = new_beliefs

# --- EXECUTION ---
print("Running R.A.P. Byzantine Fault Tolerance Test...")
sim = RAPSimulation()
history = []

for t in range(STEPS):
    sim.step(bridging_active=True)
    history.append(sim.beliefs.copy())

# --- VISUALIZATION ---
plt.figure(figsize=(10, 6))
plt.imshow(np.array(history).T, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
plt.colorbar(label='Belief (0=True, 1=False)')
plt.title(f'R.A.P. Resilience: {N_AGENTS} Agents vs {MALICIOUS_RATIO*100}% Bots')
plt.xlabel('Time Steps')
plt.ylabel('Agent ID')
plt.savefig('../output/fig1_resilience.png')
plt.show()