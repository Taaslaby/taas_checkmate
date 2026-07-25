#!/usr/bin/env python3
"""
🏊 C. elegans DepthWave Swimming Simulation

Пайплайн: C. elegans connectome (279n) → DepthWave ψ-wave → Body 2D → Video

Сравнение с VNC DepthWave (муха):
  VNC: 422 mm/s, 25K нейронов, 8 ног
  C. elegans: ? mm/s, 279 нейронов, 0 ног (змеевидное тело)
"""

import sys, os, time, math
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from celegans_depthwave import CelegansDepthWave, CelegansBody2D

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("⚠️ matplotlib not installed — will save CSV instead of video")

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("⚠️ imageio not installed — will save PNG frames instead of WebP")


class CelegansSwimSim:
    """C. elegans swimming simulation with DepthWave control."""

    def __init__(self,
                 graphml_path: str = None,
                 n_segments: int = 8,
                 dt_psi: float = 0.001,
                 dt_body: float = 0.005,
                 mod_range: float = 2.0):
        self.dt_psi = dt_psi
        self.dt_body = dt_body
        self.n_segments = n_segments
        self.psi_steps_per_body = max(1, int(dt_body / dt_psi))

        # Initialize DepthWave
        print("🧠 Инициализация C. elegans DepthWave...")
        self.dw = CelegansDepthWave(
            graphml_path=graphml_path,
            n_segments=n_segments,
            dt=dt_psi,
            mod_range=mod_range
        )

        # Initialize body
        print("🐛 Инициализация 2D тела...")
        self.body = CelegansBody2D(
            n_segments=n_segments,
            dt_body=dt_body,
        )

        # Sensors: последние N позиций сегментов
        self.sensor_buffer = np.zeros((self.n_segments, 2), dtype=np.float32)

        self.step_count = 0
        self.history = {
            'head_x': [], 'head_y': [], 'speed': [],
            'coherence': [], 'muscle_activation': [],
            'body_shape': [],
        }

    def step(self, record: bool = True) -> dict:
        """Один шаг симуляции (body dt)."""
        # Multiple ψ-wave steps per body step
        for _ in range(self.psi_steps_per_body):
            # Sensory feedback: clamped body position
            sensory = np.zeros(self.dw.n_neurons, dtype=np.float32)
            n_in = min(self.n_segments * 2, self.dw.n_neurons)
            buf = self.sensor_buffer.flatten()[:n_in]
            # Clamp to prevent ψ-wave blowup from large positional values
            buf = np.clip(buf * 0.1, -0.1, 0.1)
            sensory[:n_in] = buf

            out = self.dw.step(sensory_input=sensory)

        # Body step with motor drive
        body_out = self.body.step(out['segment_dorsal'], out['segment_ventral'])

        # Update sensor buffer
        self.sensor_buffer = np.roll(self.sensor_buffer, 1, axis=0)
        self.sensor_buffer[0] = body_out['head_x'], body_out['head_y']

        self.step_count += 1

        if record:
            self.history['head_x'].append(body_out['head_x'])
            self.history['head_y'].append(body_out['head_y'])
            self.history['speed'].append(body_out['speed'])
            self.history['coherence'].append(out['coherence'])
            self.history['muscle_activation'].append({
                'dorsal': out['segment_dorsal'].copy(),
                'ventral': out['segment_ventral'].copy(),
            })
            self.history['body_shape'].append({
                'x': body_out['x'].copy(),
                'y': body_out['y'].copy(),
                'theta': body_out['theta'].copy(),
            })

        return body_out

    def run(self, n_steps: int, record: bool = True, verbose: bool = True):
        """Запуск симуляции на N шагов."""
        print(f"\n🏊 Симуляция плавания ({n_steps} шагов = {n_steps * self.dt_body:.1f}s)...")
        t0 = time.time()

        for step in range(n_steps):
            body_out = self.step(record=record)

            if verbose and step % max(1, n_steps // 10) == 0:
                print(f"   Step {step:4d}: head=({body_out['head_x']*1000:.1f}, {body_out['head_y']*1000:.1f}) mm "
                      f"speed={body_out['speed']*1000:.2f} mm/s "
                      f"coh={self.history['coherence'][-1]:.4f}")

        sim_time = time.time() - t0
        total_dist = self.body.head_pos[0] * 1000  # в мм
        avg_speed = total_dist / (n_steps * self.dt_body)

        print(f"\n{'='*60}")
        print(f"🏁 РЕЗУЛЬТАТЫ ПЛАВАНИЯ C. ELEGANS")
        print(f"{'='*60}")
        print(f"   Дистанция:   {total_dist:.1f} mm")
        print(f"   Время:       {n_steps * self.dt_body:.2f}s")
        print(f"   Средняя скорость: {avg_speed:.1f} mm/s "
              f"({avg_speed/10:.2f} body_lengths/s)")
        print(f"   Макс скорость:    {max(self.history['speed'])*1000:.1f} mm/s")
        print(f"   Средняя coherence: {np.mean(self.history['coherence']):.4f}")
        print(f"   Compute:      {sim_time:.1f}s ({n_steps/sim_time:.0f} steps/s)")
        print(f"{'='*60}")

        return {
            'distance_mm': total_dist,
            'avg_speed_mm_s': avg_speed,
            'max_speed_mm_s': max(self.history['speed']) * 1000,
            'mean_coherence': float(np.mean(self.history['coherence'])),
            'sim_time_s': n_steps * self.dt_body,
            'compute_time_s': sim_time,
        }

    def render_frame(self, ax, step: int, title: str = None):
        """Отрисовка одного кадра."""
        shape = self.history['body_shape'][step]
        x, y = shape['x'], shape['y']

        ax.clear()
        ax.plot(x * 1000, y * 1000, 'b-o', linewidth=3, markersize=6, label='Body')
        ax.plot(x[0] * 1000, y[0] * 1000, 'ro', markersize=8, label='Head')
        ax.plot(x[-1] * 1000, y[-1] * 1000, 'gs', markersize=6, label='Tail')

        # Muscle activation heatmap
        muscle = self.history['muscle_activation'][step]
        for i in range(len(x) - 1):
            mx = (x[i] + x[i+1]) / 2 * 1000
            my = (y[i] + y[i+1]) / 2 * 1000
            act = muscle['dorsal'][i] - muscle['ventral'][i]
            color = 'red' if act > 0 else 'blue'
            ax.plot(mx, my, 'o', color=color, alpha=min(abs(act) + 0.2, 1.0), markersize=5)

        # Head trajectory
        if step > 10:
            trail = max(0, step - 100)
            hx = self.history['head_x'][trail:step + 1]
            hy = self.history['head_y'][trail:step + 1]
            ax.plot(np.array(hx) * 1000, np.array(hy) * 1000,
                    'gray', alpha=0.3, linewidth=1)

        # Dynamic limits
        all_x = np.array(x) * 1000
        valid = np.isfinite(all_x)
        if valid.any():
            x_max = max(abs(all_x[valid])) + 5
            ax.set_xlim(-max(5, x_max/4), max(30, x_max + 5))
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(title)
        else:
            speed = self.history['speed'][step] if np.isfinite(self.history['speed'][step]) else 0
            coh = self.history['coherence'][step] if np.isfinite(self.history['coherence'][step]) else 0
            ax.set_title(f"C. elegans Swimming — Step {step}\n"
                        f"Speed: {speed*1000:.1f} mm/s  Coherence: {coh:.4f}")

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.legend(loc='upper right', fontsize=8)

    def save_video(self, output_path: str, n_frames: int = None,
                   fps: int = 30, dpi: int = 100):
        """Сохранение видео плавания."""
        if not HAS_MPL:
            print("⏭ Skipping video — matplotlib not available")
            return

        if n_frames is None:
            n_frames = len(self.history['head_x'])

        n_frames = min(n_frames, len(self.history['head_x']))
        print(f"\n🎬 Рендеринг видео ({n_frames} frames, {fps} fps)...")

        fig, ax = plt.subplots(figsize=(8, 4))
        canvas = FigureCanvas(fig)

        frames = []
        t0 = time.time()

        for step in range(0, n_frames, max(1, n_frames // 300)):  # max 300 frames
            self.render_frame(ax, step)
            canvas.draw()

            w, h = canvas.get_width_height()
            buf = canvas.tostring_argb()
            img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            frames.append(img[:, :, :3])  # drop alpha

            if step % 50 == 0:
                print(f"   Frame {step}/{n_frames} ({len(frames)} total)")

        render_time = time.time() - t0
        print(f"   Рендеринг: {render_time:.1f}s")

        # Save
        if output_path.endswith('.webp') and HAS_IMAGEIO:
            imageio.mimsave(output_path, frames, fps=fps, codec='libwebp')
            print(f"   💾 Сохранено: {output_path} ({len(frames)} frames)")
        elif output_path.endswith('.gif') and HAS_IMAGEIO:
            imageio.mimsave(output_path, frames, fps=fps)
            print(f"   💾 Сохранено: {output_path} ({len(frames)} frames)")
        else:
            # Save as individual PNGs
            out_dir = Path(output_path).with_suffix('')
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                plt.imsave(str(out_dir / f"frame_{i:04d}.png"), frame)
            print(f"   💾 Сохранено {len(frames)} PNG в {out_dir}/")

        plt.close(fig)


def main():
    print("=" * 60)
    print("🏊 C. ELEGANS DEPTHWAVE SWIMMING SIMULATION")
    print("=" * 60)

    # Parameters
    N_SEGMENTS = 8
    N_STEPS = 1000    # body steps = 1000 * 0.005 = 5s
    DT_PSI = 0.001
    DT_BODY = 0.005

    # Find graphml
    graphml = os.path.join(
        os.path.dirname(__file__),
        '../../../../sardaanaCM/experiments/datasets',
        'c.elegans.herm_pharynx_1.graphml'
    )

    # Run simulation
    sim = CelegansSwimSim(
        graphml_path=graphml if os.path.exists(graphml) else None,
        n_segments=N_SEGMENTS,
        dt_psi=DT_PSI,
        dt_body=DT_BODY,
    )

    results = sim.run(N_STEPS)

    # Save video
    video_path = os.path.join(
        os.path.dirname(__file__),
        'celegans_swim.webp'
    )
    sim.save_video(video_path, fps=30, dpi=100)

    # Save metrics CSV
    csv_path = video_path.replace('.webp', '.csv')
    with open(csv_path, 'w') as f:
        f.write('step,head_x_mm,head_y_mm,speed_mm_s,coherence\n')
        for i in range(len(sim.history['head_x'])):
            f.write(f"{i},"
                    f"{sim.history['head_x'][i]*1000:.3f},"
                    f"{sim.history['head_y'][i]*1000:.3f},"
                    f"{sim.history['speed'][i]*1000:.3f},"
                    f"{sim.history['coherence'][i]:.6f}\n")
    print(f"   💾 Метрики: {csv_path}")

    # Bonus: render final frame as PNG
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 5))
        sim.render_frame(ax, len(sim.history['head_x']) - 1,
                        title=f"C. elegans Swimming — Final Frame\n"
                              f"Avg Speed: {results['avg_speed_mm_s']:.1f} mm/s")
        plt.tight_layout()
        png_path = video_path.replace('.webp', '_final.png')
        plt.savefig(png_path, dpi=150)
        print(f"   💾 Финальный кадр: {png_path}")
        plt.close()

    print("\n✅ Симуляция завершена!")
    return results


if __name__ == '__main__':
    main()
