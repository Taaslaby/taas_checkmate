import numpy as np
import matplotlib.pyplot as plt
import taas_checkmate

def simulate_human_arm():
    # Движок требует размерности кратной 8
    # 8 мышц = 4 пары агонист/антагонист (суставов)
    n_muscles = 8
    n_joints = 4

    # Массивы состояния
    ca = np.zeros(n_muscles, dtype=np.float32)
    tension = np.zeros(n_muscles, dtype=np.float32)
    
    # 1. Локоть (Бицепс vs Трицепс)
    # 2. Плечо (Передняя дельта vs Задняя дельта)
    # 3. Кисть (Сгибатель vs Разгибатель)
    # 4. Пальцы (Сгибатель vs Разгибатель)
    
    # Физиология человека (длинные времена релаксации по сравнению с насекомыми)
    d_ca = np.full(n_muscles, 0.5, dtype=np.float32)
    tau_pump = np.full(n_muscles, 50.0, dtype=np.float32) # Долгая откачка кальция
    eta_reset = np.full(n_muscles, 0.8, dtype=np.float32)
    
    # Максимальная сила мышц в Ньютонах
    # Бицепс ~ 1000Н, Трицепс ~ 700Н
    t_max = np.array([1000.0, 700.0, 800.0, 800.0, 300.0, 200.0, 100.0, 100.0], dtype=np.float32)
    kd = np.full(n_muscles, 0.3, dtype=np.float32)
    
    # Динамика суставов
    joint_delta_f = np.zeros(n_joints, dtype=np.float32)
    joint_stiffness = np.zeros(n_joints, dtype=np.float32)
    joint_damping = np.zeros(n_joints, dtype=np.float32)
    
    # K0 (Пассивная жесткость), Beta (Коэффициент активной жесткости при коконтракции)
    k0 = np.array([10.0, 15.0, 5.0, 2.0], dtype=np.float32)
    beta = np.array([50.0, 60.0, 20.0, 10.0], dtype=np.float32)
    b0 = np.array([2.0, 3.0, 1.0, 0.5], dtype=np.float32)
    alpha = np.array([5.0, 6.0, 2.0, 1.0], dtype=np.float32)

    dt = 1.0 # 1 мс шаг
    steps = 1000 # 1 секунда симуляции

    # Сбор данных
    history_tension_biceps = []
    history_tension_triceps = []
    history_joint_torque = []
    history_stiffness = []

    for t in range(steps):
        spikes = np.zeros(n_muscles, dtype=np.int32)
        resets = np.zeros(n_muscles, dtype=np.int32)
        
        # Фаза 1: Сгибание (работает только Бицепс) t=100..400
        if 100 <= t < 400:
            if t % 10 == 0:  # 100 Hz firing rate
                spikes[0] = 1
                
        # Фаза 2: Разгибание (работает только Трицепс) t=500..800
        if 500 <= t < 800:
            if t % 10 == 0:  
                spikes[1] = 1

        # Фаза 3: Коконтракция (Бицепс и Трицепс напрягаются одновременно для стабилизации сустава) t=850..950
        if 850 <= t < 950:
            if t % 10 == 0:
                spikes[0] = 1
                spikes[1] = 1
        
        # Симуляция 1 шага
        taas_checkmate.neurons.step_atlasov_neuromotor(
            ca, tension, spikes, resets, dt,
            d_ca, tau_pump, eta_reset, t_max, kd,
            joint_delta_f, joint_stiffness, joint_damping,
            k0, beta, b0, alpha
        )
        
        history_tension_biceps.append(tension[0])
        history_tension_triceps.append(tension[1])
        history_joint_torque.append(joint_delta_f[0])
        history_stiffness.append(joint_stiffness[0])

    # Отрисовка
    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.title("Сила мышц (Tension) в Ньютонах")
    plt.plot(history_tension_biceps, label="Бицепс (Сгибатель)", color='red')
    plt.plot(history_tension_triceps, label="Трицепс (Разгибатель)", color='blue')
    plt.ylabel("Сила (Н)")
    plt.legend(loc="upper left")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.title("Тяга Локтевого Сустава (Трицепс - Бицепс)")
    plt.plot(history_joint_torque, color='purple', label="Крутящий момент (Delta F)")
    plt.ylabel("Разность сил (Н)")
    plt.legend(loc="upper left")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.title("Динамическая жесткость локтевого сустава (Stiffness)")
    plt.plot(history_stiffness, color='green', label="Жесткость сустава")
    plt.xlabel("Время (мс)")
    plt.ylabel("Коэфф. жесткости")
    plt.legend(loc="upper left")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("human_arm_simulation.png")
    print("Симуляция завершена! Графики сохранены в human_arm_simulation.png")

if __name__ == "__main__":
    simulate_human_arm()
