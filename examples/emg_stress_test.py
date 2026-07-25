import numpy as np
import matplotlib.pyplot as plt
import taas_checkmate
import time

def generate_muap_template(duration_ms=15, dt=1.0):
    # Аппроксимация потенциала действия моторной единицы (MUAP) через вейвлет Рикера (Мексиканская шляпа)
    points = int(duration_ms / dt)
    t = np.linspace(-3, 3, points) # Нормализованное время от -3 до 3 сигма
    # Уравнение вейвлета Рикера
    return (1.0 - t**2) * np.exp(-0.5 * t**2)

def run_stress_test():
    # 4000 моторных единиц (максимальный стресс-тест)
    # Это масштаб огромной мышцы (например, квадрицепса)
    n_muscles = 4000 
    n_joints = n_muscles // 2 
    
    ca = np.zeros(n_muscles, dtype=np.float32)
    tension = np.zeros(n_muscles, dtype=np.float32)
    
    d_ca = np.full(n_muscles, 0.5, dtype=np.float32)
    eta_reset = np.full(n_muscles, 0.8, dtype=np.float32)
    kd = np.full(n_muscles, 0.3, dtype=np.float32)
    
    tau_pump = np.zeros(n_muscles, dtype=np.float32)
    t_max = np.zeros(n_muscles, dtype=np.float32)
    
    # Пул моторных единиц (Распределение Хеннемана)
    for f in range(n_muscles):
        # От мелких к очень крупным (экспоненциальный рост размера моторной единицы)
        unit_size = np.exp((f / n_muscles) * 4.0) 
        t_max[f] = 0.5 * unit_size # Максимальная сила волокна
        tau_pump[f] = 100.0 - (f / n_muscles) * 80.0 # Крупные волокна откачивают кальций быстрее
        
    joint_delta_f = np.zeros(n_joints, dtype=np.float32)
    joint_stiffness = np.zeros(n_joints, dtype=np.float32)
    joint_damping = np.zeros(n_joints, dtype=np.float32)
    k0 = np.ones(n_joints, dtype=np.float32)
    beta = np.ones(n_joints, dtype=np.float32)
    b0 = np.ones(n_joints, dtype=np.float32)
    alpha = np.ones(n_joints, dtype=np.float32)

    dt = 1.0 
    steps = 4000 # 4 секунды

    history_force = []
    
    # Массив для сбора электрической активности всех волокон
    emg_spiketrain = np.zeros(steps, dtype=np.float32)

    start_time = time.time()
    print("="*60)
    print(f"🚀 СТРЕСС-ТЕСТ: Запуск симуляции {n_muscles} независимых волокон...")
    print("="*60)

    for t in range(steps):
        spikes = np.zeros(n_muscles, dtype=np.int32)
        resets = np.zeros(n_muscles, dtype=np.int32)
        
        # Моторная команда: плавное напряжение и расслабление
        # t=500..2000 (напряжение), t=2000..3500 (расслабление)
        cmd = 0.0
        if 500 <= t < 2000:
            cmd = (t - 500) / 1500.0
        elif 2000 <= t < 3500:
            cmd = 1.0 - (t - 2000) / 1500.0
            
        # Рекрутинг 4000 волокон
        active_fibers = int(cmd * n_muscles)
        for f in range(active_fibers):
            drive = cmd - (f / n_muscles)
            freq = 10 + drive * 120 # Частота от 10 до 130 Гц
            period = max(1, int(1000 / freq))
            
            # Если волокно спайкует
            if (t + f * 13) % period == 0:
                spikes[f] = 1
                # Электрический потенциал больше у крупных волокон.
                # (np.random.rand() - 0.5) задает случайную поляризацию (биполярный электрод)
                voltage = np.exp((f / n_muscles) * 4.0) * (np.random.rand() - 0.5)
                emg_spiketrain[t] += voltage

        # SIMD ядро Rust
        taas_checkmate.neurons.step_atlasov_neuromotor(
            ca, tension, spikes, resets, dt,
            d_ca, tau_pump, eta_reset, t_max, kd,
            joint_delta_f, joint_stiffness, joint_damping,
            k0, beta, b0, alpha
        )
        
        history_force.append(np.sum(tension))
        
        if t % 1000 == 0:
            print(f"[t={t:04d} ms] Сила: {np.sum(tension):7.1f} N | Работает волокон: {active_fibers}")

    sim_time = time.time() - start_time
    print(f"\n[OK] Симуляция 4,000 волокон завершена за {sim_time:.3f} сек!")

    # 4. Генерация реалистичной ЭМГ (Surface EMG)
    print("[INFO] Конволюция интерференционного паттерна (генерация клинической ЭМГ)...")
    muap = generate_muap_template(duration_ms=15, dt=dt)
    raw_emg = np.convolve(emg_spiketrain, muap, mode='same')
    
    # Добавляем базовый тепловой шум сенсора (как на настоящем осциллографе)
    raw_emg += np.random.normal(0, 1.0, size=steps)

    # Отрисовка в медицинском стиле
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#ffffff')
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_facecolor('#fdfdfd')
    # Черная тонкая линия как на ЭКГ/ЭМГ мониторах
    ax1.plot(raw_emg, color='#000000', linewidth=0.7)
    ax1.set_title(f"Сырая Поверхностная Электромиограмма (sEMG) - Агрегация {n_muscles} моторных единиц", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Амплитуда (mV)")
    ax1.set_xlim(0, 4000)
    ax1.grid(True, color='#eeeeee', linestyle='--')
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_facecolor('#fdfdfd')
    ax2.plot(history_force, color='#d62728', linewidth=2.5)
    ax2.set_title(f"Суммарная макроскопическая сила мышцы | Пик: {max(history_force):.0f} Ньютонов", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Время (мс)")
    ax2.set_ylabel("Сила (Ньютоны)")
    ax2.set_xlim(0, 4000)
    ax2.grid(True, color='#eeeeee', linestyle='--')
    
    plt.tight_layout()
    plt.savefig("clinical_emg_stress_test.png", dpi=200)
    print("[OK] Графики сохранены в clinical_emg_stress_test.png")

if __name__ == "__main__":
    run_stress_test()
