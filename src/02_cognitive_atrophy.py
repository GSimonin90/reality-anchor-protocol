import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
CYCLES = 200
SOCRATIC_FRICTION_RATE = 0.40  # The Breakpoint found in T22 Test 

class UserAgent:
    def __init__(self):
        self.skill = 0.8  # Initial Critical Thinking Skill ($S$)
        self.reputation_score = 50

class SocraticFrictionEngine:
    """
    Layer 2.5: Adaptive logic with Behavioral safeguards.
    Ref: R.A.P. Addendum, Appendix C logic.
    """
    def __init__(self, user):
        self.user = user
        # 'supervision_need' (0.0-1.0): How much system checks this user
        self.supervision_need = 0.5 

    def calculate_trigger_probability(self, content_ambiguity=0.5):
        # 1. Base Risk dependent on supervision need 
        p = 0.1 + (self.supervision_need * 0.6)
        
        # 2. Context Modifiers (Ambiguity increases friction) 
        if content_ambiguity > 0.6: p += 0.3
        
        # 3. Anti-Reactance: Trust Discount for high rep users (>80) 
        if self.user.reputation_score > 80:
            p *= 0.5
            
        # 4. Baseline Vigilance (Champion's Workout) 
        p = max(0.05, p)
        return min(0.95, p)

    def interact(self):
        p_trigger = self.calculate_trigger_probability()
        
        # Does the system challenge the user?
        if np.random.rand() < p_trigger:
            # === CHALLENGE TRIGGERED ===
            # User uses Skill ($S$) to verify
            if np.random.rand() < self.user.skill:
                # SUCCESS: Reputation up, Supervision down 
                self.supervision_need = max(0.0, self.supervision_need - 0.1)
                self.user.reputation_score += 2
                # Skill Growth (Mental Workout)
                self.user.skill = min(1.0, self.user.skill + 0.01)
            else:
                # FAILURE
                self.supervision_need = min(1.0, self.supervision_need + 0.15)
                self.user.reputation_score -= 1
        else:
            # === FREE FLOW (Automation) ===
            # No challenge -> Cognitive Atrophy sets in 
            self.user.skill = max(0.0, self.user.skill - 0.005)

# --- EXECUTION ---
print("Running Cognitive Atrophy Simulation...")

# Scenario A: No Friction (Pure Automation - T21)
agent_lazy = UserAgent()
history_lazy = []
for _ in range(CYCLES):
    # Pure atrophy decay
    agent_lazy.skill = max(0.0, agent_lazy.skill - 0.005)
    history_lazy.append(agent_lazy.skill)

# Scenario B: Smart Socratic Engine (T22)
agent_smart = UserAgent()
engine = SocraticFrictionEngine(agent_smart)
history_smart = []
for _ in range(CYCLES):
    engine.interact()
    history_smart.append(agent_smart.skill)

# --- VISUALIZATION ---
plt.figure(figsize=(10, 6))
plt.plot(history_lazy, label='100% Automation (Atrophy T21)', color='red', linestyle='--')
plt.plot(history_smart, label='Smart Socratic Engine (Resilience T22)', color='blue')
plt.axhline(y=0.1, color='gray', linestyle=':', label='Critical Failure Zone (<0.10)')
plt.title('The Paradox of Delegated Truth: Cognitive Atrophy')
plt.ylabel('Human Critical Skill ($S$)')
plt.xlabel('Interactions')
plt.legend()
plt.savefig('../output/fig2_socratic_atrophy.png')
plt.show()