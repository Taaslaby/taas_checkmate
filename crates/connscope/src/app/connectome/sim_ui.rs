//! # Запуск симуляций (`app::sim_ui`)
//!
//! Тонкий слой между UI и движком `crate::sim`:
//!
//! * `sim_nodes` — набор нейронов для локальной симуляции: драйв + ego-соседи
//!   (лимит `MAX_SIM_NODES`, чтобы не висло);
//! * `run_sim` — кнопка ▶: либо **локальный** контур (подграф вокруг драйва),
//!   либо **полный VNC** (вся сеть, 102K нейронов — см. `run_full_sim`);
//!   результат кладётся в `sim_result` → нижние графики и 3D-трасса;
//! * `run_disin` — детектор растормаживания `тормоз →⊣ тормоз →⊣ нейрон`
//!   для выбранного тормозного нейрона.

use eframe::egui;

use crate::app::connectome::ConnectomeMode;
use crate::sim::models;
use crate::sim::{find_disinhibition, run_full_sim_stream, run_simulation, SimOutcome, StreamMsg};

impl ConnectomeMode {
    /// Узлы локальной симуляции: драйв + ego-соседи (без дублей, с лимитом).
    pub(crate) fn sim_nodes(&self) -> Vec<usize> {
        let mut nodes: Vec<usize> = Vec::new();
        if let Some(sel) = self.sel.selected {
            nodes.push(sel);
            for &(t, _, _) in &self.sel.ego {
                if nodes.len() >= crate::app::MAX_SIM_NODES {
                    break;
                }
                if !nodes.contains(&t) {
                    nodes.push(t);
                }
            }
        }
        nodes
    }

    /// Запуск симуляции (кнопка ▶ в инспекторе). Драйв = выбранный нейрон.
    /// Сбрасывает `sim_result` и кеш 3D-текстуры (трасса перепекается).
    pub(crate) fn run_sim(&mut self) {
        self.sim.result = None;
        self.camera.atlas3d = None; // перепекаем 3D (возможно, с трассой)
        let Some(sel) = self.sel.selected else {
            self.sim.status = "сначала кликни по нейрону (он будет драйвом)".into();
            return;
        };
        let Some(d) = self.data.as_ref() else {
            self.sim.status = "нет данных".into();
            return;
        };
        if self.sim.full {
            // Полный VNC — всегда живым потоком: UI не замирает, вспышки
            // сигнала летят по 3D-атласу (см. `start_live`).
            self.start_live();
            return;
        }
        let nodes = self.sim_nodes();
        match run_simulation(
            d,
            &nodes,
            sel,
            self.sim.model,
            self.sim.bias,
            self.sim.steps,
            self.sim.pulse,
            0.5,
            self.sim.gh,
        ) {
            Ok(r) => {
                let total: i32 = r.raster.iter().sum();
                let active = r.rates.iter().filter(|&&x| x > 0.0).count();
                self.sim.status = format!(
                    "{}: {} нейронов, {} спайков, активных {}/{}",
                    r.model,
                    r.idx.len(),
                    total,
                    active,
                    r.idx.len()
                );
                self.sim.result = Some(SimOutcome::Local(r));
                self.plots.enable_kind_if_none(true); // плоты ego-контура
            }
            Err(e) => self.sim.status = format!("⚠ {e}"),
        }
    }

    // ── живой сигнал: потоковая симуляция ─────────────────────

    /// Запуск «живого сигнала»: потоковая симуляция всей сети, где каждый
    /// шаг превращается в вспышку на 3D-атласе (см. `glow` + шейдер сплэтов).
    /// Драйв = выбранный нейрон; скорость крутится `live_step_delay_ms`.
    pub(crate) fn start_live(&mut self) {
        self.stop_live(); // гасим прошлый поток до заимствований
        let Some(sel) = self.sel.selected else {
            self.live.status = "сначала кликни по нейрону (он будет драйвом)".into();
            return;
        };
        let Some(d) = self.data.as_ref() else {
            self.live.status = "нет данных".into();
            return;
        };
        self.sim.result = None;
        self.camera.atlas3d = None; // трасса перепекается
        self.camera.view3d = true; // вспышки живут в 3D-сплэтах — включаем вид
        self.live.glow = vec![0.0; d.n()];
        match run_full_sim_stream(
            d,
            sel,
            self.sim.model,
            self.sim.bias,
            self.sim.steps,
            self.sim.pulse,
            0.5,
            self.sim.gh,
        ) {
            Ok(s) => {
                self.live.steps_done = 0;
                self.live.status = format!(
                    "живой поток · {} · драйв #{sel} · {} шагов · {} шагов/с",
                    models::MODELS[self.sim.model].name(),
                    self.sim.steps,
                    self.live.speed as i32
                );
                self.live.stream = Some(s);
            }
            Err(e) => self.live.status = format!("⚠ {e}"),
        }
    }

    /// Останавливает живой поток (если есть) и гасит свечение.
    pub(crate) fn stop_live(&mut self) {
        if let Some(mut s) = self.live.stream.take() {
            s.stop();
        }
        if !self.live.glow.is_empty() {
            self.live.glow.fill(0.0);
        }
    }

    /// Один кадр живого сигнала: гасим хвосты свечения (экспонента от
    /// реального времени кадра), потребляем события потока (список
    /// сработавших нейронов → glow = 1.0); на `Done` — кладём результат в
    /// `sim_result` (для трассы/графиков) и закрываем поток.
    pub(crate) fn tick_live(&mut self, ctx: &egui::Context) {
        if !self.live.glow.is_empty() {
            let dt = ctx.input(|i| i.stable_dt).min(0.1);
            let dec = (-dt * 12.0).exp(); // ~пол-секунды до полного затухания
            for g in &mut self.live.glow {
                *g *= dec;
            }
        }
        let Some(s) = self.live.stream.as_ref() else {
            return;
        };
        let dt = ctx.input(|i| i.stable_dt).clamp(0.001, 0.1);
        // Дозированное потребление: показываем не быстрее `live_speed` шагов/с,
        // чтобы волна шла плавно, а не схлопывалась во вспышку. Остальное
        // копится в канале (порядок событий сохраняется).
        let mut budget = (self.live.speed * dt).ceil().max(1.0) as usize;
        let mut finished = false;
        while budget > 0 {
            match s.rx.try_recv() {
                Ok(StreamMsg::Fired { step, fired }) => {
                    budget -= 1;
                    self.live.steps_done = step + 1;
                    for i in fired {
                        if i < self.live.glow.len() {
                            self.live.glow[i] = 1.0;
                        }
                    }
                }
                Ok(StreamMsg::Done(res)) => {
                    let r = *res;
                    let n = self.data.as_ref().map(|d| d.n()).unwrap_or(0);
                    self.live.status = format!(
                        "полный прогон · {}: {} спайков, активных {}/{} · {} мс",
                        r.model,
                        r.total_spikes,
                        r.active_neurons,
                        n,
                        r.steps as f32 * r.dt_ms
                    );
                    self.sim.result = Some(SimOutcome::Full(r));
                    self.plots.enable_kind_if_none(false); // научные плоты
                    finished = true;
                    break;
                }
                Err(_) => break, // канал пуст — ждём следующий кадр
            }
        }
        if finished {
            self.live.stream = None;
        } else {
            ctx.request_repaint(); // поток жив — не даём кадру уснуть
        }
    }

    /// Поиск цепей растормаживания для выбранного тормозного нейрона:
    /// `источник →⊣ тормозной M →⊣ цель Y` — торможение тормоза = плюс.
    pub(crate) fn run_disin(&mut self) {
        self.disin.chains = Vec::new();
        let Some(sel) = self.sel.selected else {
            self.disin.status = "сначала выбери тормозной нейрон".into();
            return;
        };
        let Some(d) = self.data.as_ref() else {
            self.disin.status = "нет данных".into();
            return;
        };
        if !d.nt(sel).inhibitory() {
            self.disin.status = format!("{} — возбуждающий, растормаживания нет", d.nt(sel).name());
            return;
        }
        self.disin.chains = find_disinhibition(d, sel, 30);
        self.disin.status = format!("{} цепей тормоз→⊣тормоз→⊣нейрон", self.disin.chains.len());
    }
}
