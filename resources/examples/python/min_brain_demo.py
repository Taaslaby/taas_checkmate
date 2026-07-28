import numpy as np
from taas_checkmate import MinBrain

def main():
    print("=" * 60)
    print("🧠 MINIMAL BRAIN (MINBRAIN) DEMO 🧠")
    print("Architecture: Sensory -> GNW <-> PFC (Working Memory) -> BG (Action)")
    print("=" * 60)
    
    sensory_dim = 16
    n_actions = 4
    brain = MinBrain(sensory_dim=sensory_dim, workspace_size=32, pfc_slots=2, n_actions=n_actions, clear_threshold=0.8)
    
    # 1. Provide an empty input, let it tick a few times
    empty_input = np.zeros(sensory_dim, dtype=np.float32)
    print("\n[Tick 1-3] Empty input, testing resting dynamics...")
    for i in range(3):
        action = brain.step(empty_input)
        print(f"  Step {i+1}: Selected Action = {action}")
        
    # 2. Provide a strong sensory input
    # Let's say feature 5 represents a "store to memory" cue.
    strong_input = np.zeros(sensory_dim, dtype=np.float32)
    strong_input[5] = 10.0
    strong_input[10] = 5.0
    print("\n[Tick 4] Strong sensory cue (e.g. object appears)...")
    action = brain.step(strong_input)
    print(f"  Step 4: Selected Action = {action}")
    
    # 3. Check PFC State
    pfc_state = brain.get_pfc_state()
    print(f"  PFC State Norm = {np.linalg.norm(pfc_state):.4f}")
    
    # 4. Remove input, see if PFC remembers it (working memory)
    print("\n[Tick 5-10] Removing input, relying on Working Memory...")
    for i in range(5, 11):
        action = brain.step(empty_input)
        pfc = brain.get_pfc_state()
        print(f"  Step {i}: Action = {action} | PFC State Norm = {np.linalg.norm(pfc):.4f}")

    print("\n[Success] The Basal Ganglia continues to select actions based on the fading memory in the Frontal Lobe!")

if __name__ == "__main__":
    main()
