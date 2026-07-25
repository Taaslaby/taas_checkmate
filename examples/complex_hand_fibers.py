import numpy as np
import matplotlib.pyplot as plt
import taas_checkmate

def simulate_fibers():
    # 4 мышечные группы, каждая состоит из 40 моторных единиц (волокон)
    # Итого 160 каналов. Это кратно 8, идеально для движка.
    n_fibers_per_muscle = 40
    n_muscles = 4 * n_fibers_per_muscle # 160
    n_joints = n_muscles // 2 
    
    # Массивы состояния
    ca = np.zeros(n_muscles, dtype=np.float32)
    tension = np.zeros(n_muscles, dtype=np.float32)
    
    # Распределение мышечных волокон:
    # 0..39: Бицепс (Сгибатель локтя)
    # 40..79: Трицепс (Разгибатель локтя)
    # 80..119: Сгибатели пальцев (FDS/FDP - Хват)
    # 120..159: Разгибатели пальцев (EDC - Разжатие)
    
    d_ca = np.full(n_muscles, 0.5, dtype=np.float32)
    eta_reset = np.full(n_muscles, 0.8, dtype=np.float32)
    kd = np.full(n_muscles, 0.3, dtype=np.float32)
    
    tau_pump = np.zeros(n_muscles, dtype=np.float32)
    t_max = np.zeros(n_muscles, dtype=np.float32)
    
    # Принцип размера Хеннемана:
    # Малые мотонейроны иннервируют медленные волокна (слабая сила, долгая релаксация)
    # Большие мотонейроны иннервируют быстрые волокна (огромная сила, быстрая релаксация)
    for m in range(4):
        offset = m * n_fibers_per_muscle
        for f in range(n_fibers_per_muscle):
            # Экспоненциальный рост силы от медленных к быстрым волокнам
            unit_size = np.exp(f / 10.0) 
            t_max[offset + f] = 0.5 * unit_size # Ньютоны на 1 моторную единицу
            
            # Быстрые волокна быстрее откачивают кальций
            tau_pump[offset + f] = 60.0 - f * 1.0 # от 60мс до 21мс
            
    # Заглушки для суставов (нам здесь важна агрегированная сила)
    joint_delta_f = np.zeros(n_joints, dtype=np.float32)
    joint_stiffness = np.zeros(n_joints, dtype=np.float32)
    joint_damping = np.zeros(n_joints, dtype=np.float32)
    k0 = np.ones(n_joints, dtype=np.float32)
    beta = np.ones(n_joints, dtype=np.float32)
    b0 = np.ones(n_joints, dtype=np.float32)
    alpha = np.ones(n_joints, dtype=np.float32)

    dt = 1.0 
    steps = 2000 # 2 секунды

    history_biceps_force = []
    history_grip_force = []
    history_spikes = []

    # Функция рекрутирования волокон
    def get_recruitment(command, n_fibers):
        return int(command * n_fibers)

    for t in range(steps):
        spikes = np.zeros(n_muscles, dtype=np.int32)
        resets = np.zeros(n_muscles, dtype=np.int32)
        
        # Сценарий симуляции:
        # t=200..600: Плавно сжимаем пальцы (берем кружку)
        # t=600..1200: Поднимаем кружку (Бицепс активируется)
        # t=1500..1800: Разжимаем пальцы (ставим кружку, экстензоры пальцев вкл)
        
        grip_cmd = 0.0
        biceps_cmd = 0.0
        extensor_cmd = 0.0
        
        if 200 <= t < 600:
            grip_cmd = (t - 200) / 400.0 # Наращиваем хват
        elif 600 <= t < 1500:
            grip_cmd = 1.0 # Держим
            
        if 600 <= t < 800:
            biceps_cmd = (t - 600) / 200.0 # Плавно поднимаем
        elif 800 <= t < 1200:
            biceps_cmd = 1.0 # Удерживаем вес
        elif 1200 <= t < 1500:
            biceps_cmd = 1.0 - (t - 1200) / 300.0 # Опускаем руку
            
        if 1500 <= t < 1800:
            extensor_cmd = 1.0 # Раскрываем пальцы
            
        # Генерация спайков (ЭМГ) для каждой группы мышц
        for m, cmd in enumerate([biceps_cmd, 0.0, grip_cmd, extensor_cmd]):
            active_fibers = get_recruitment(cmd, n_fibers_per_muscle)
            offset = m * n_fibers_per_muscle
            for f in range(active_fibers):
                # Чем сильнее команда, тем выше частота (от 10 до 100 Гц)
                drive = cmd - (f / n_fibers_per_muscle)
                freq = 10 + drive * 90 
                period = max(1, int(1000 / freq))
                
                # Добавим фазовый сдвиг чтобы волокна не стреляли идеально синхронно (как в реальности)
                if (t + f * 7) % period == 0:
                    spikes[offset + f] = 1
                    history_spikes.append((t, offset + f))
                    
        # Вызов Rust SIMD движка
        taas_checkmate.neurons.step_atlasov_neuromotor(
            ca, tension, spikes, resets, dt,
            d_ca, tau_pump, eta_reset, t_max, kd,
            joint_delta_f, joint_stiffness, joint_damping,
            k0, beta, b0, alpha
        )
        
        # Агрегируем силу по всем волокнам
        biceps_force = np.sum(tension[0:40])
        grip_force = np.sum(tension[80:120])
        history_biceps_force.append(biceps_force)
        history_grip_force.append(grip_force)
        
        # Выводим телеметрию в реальном времени (понты)
        if t % 40 == 0:
            active_f_biceps = np.sum(spikes[0:40])
            active_f_grip = np.sum(spikes[80:120])
            print(f"[SIM t={t:04d} ms] | BICEPS: {biceps_force:7.2f} N (Fibers Active: {active_f_biceps:2d}) | GRIP (FDS/FDP): {grip_force:7.2f} N (Fibers Active: {active_f_grip:2d}) | J_STIFFNESS: {joint_stiffness[0]:6.2f}")

    # Отрисовка
    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor('#111111')
    
    # 1. Электромиограмма (Raster plot)
    ax1 = plt.subplot(3, 1, 1)
    ax1.set_facecolor('#1e1e1e')
    spk_t = [s[0] for s in history_spikes]
    spk_id = [s[1] for s in history_spikes]
    ax1.scatter(spk_t, spk_id, s=1.5, color='#00ffcc', alpha=0.7)
    ax1.axhline(40, color='gray', linestyle='-', alpha=0.5)
    ax1.axhline(80, color='gray', linestyle='-', alpha=0.5)
    ax1.axhline(120, color='gray', linestyle='-', alpha=0.5)
    ax1.set_yticks([20, 60, 100, 140])
    ax1.set_yticklabels(['Бицепс', 'Трицепс', 'Хват (Пальцы)', 'Разжим (Пальцы)'], color='white')
    ax1.set_title("Сырая Электромиограмма (EMG) — 160 Моторных Единиц", color='white')
    ax1.set_xlim(0, 2000)
    ax1.tick_params(colors='white')
    
    # 2. Сила хвата
    ax2 = plt.subplot(3, 1, 2)
    ax2.set_facecolor('#1e1e1e')
    ax2.plot(history_grip_force, color='#ff0055', linewidth=2)
    ax2.set_title("Сила хвата кисти (Агрегация 40 волокон FDS/FDP)", color='white')
    ax2.set_ylabel("Сила (Ньютоны)", color='white')
    ax2.set_xlim(0, 2000)
    ax2.grid(True, color='#333333')
    ax2.tick_params(colors='white')
    
    # 3. Сила бицепса
    ax3 = plt.subplot(3, 1, 3)
    ax3.set_facecolor('#1e1e1e')
    ax3.plot(history_biceps_force, color='#00aaff', linewidth=2)
    ax3.set_title("Сила Бицепса (Агрегация 40 волокон)", color='white')
    ax3.set_xlabel("Время симуляции (мс)", color='white')
    ax3.set_ylabel("Сила (Ньютоны)", color='white')
    ax3.set_xlim(0, 2000)
    ax3.grid(True, color='#333333')
    ax3.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig("complex_hand_fibers.png", dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"\n[OK] Симуляция 160 волокон завершена. Графики сохранены в complex_hand_fibers.png")

if __name__ == "__main__":
    simulate_fibers()
