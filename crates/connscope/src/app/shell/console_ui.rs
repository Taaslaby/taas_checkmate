//! # Встроенная консоль-комбайн (`app::console_ui`)
//!
//! Тот же синтаксис, что у CLI-комбайна (`connscope` из терминала), но
//! прямо в GUI: `stats`, `search`, `ego`, `sim`, `fullsim`, `disin`,
//! `select`, `model`, `trace`, `camera`, `clear`, `help`.
//!
//! * `console_view` — вкладка нижнего дока: вывод + строка ввода с историей
//!   (↑/↓), кликабельные `#индексы` выбирают нейрон (F12 — открыть);
//! * `console_exec` — диспетчер команд (парсит как CLI);
//! * `console_sim` — `sim`/`fullsim` из консоли: результат ложится в те же
//!   поля, что и кнопка ▶ (нижние графики + 3D-трасса).

use eframe::egui::{self, Key};

use crate::app::connectome::ConnectomeMode;
use crate::cli;
use crate::sim::models;
use crate::sim::{run_full_sim, SimOutcome};

const CONSOLE_HELP: &str = "\
Команды (как в CLI-комбайне):
  stats                       статистика графа
  search <запрос>             поиск по типу/классу/НТ/ноге/body_id
  ego <idx> [--k N]           ego-граф нейрона
  select <idx> | focus <idx>  выбрать нейрон и навести камеру
  sim [--drive idx] [--model lif|izh|hh|haap] [--steps N] [--bias F] [--pulse F] [--gh F] [--drug имя] [--conc µM] [--half-life шаги]
  fullsim [--drive idx] [...] полный прогон всей сети (+ 3D-трасса при trace on) [+ --drug имя] [+ --half-life шаги]
  disin <idx> [--top N]       растормаживание: тормоз→⊣тормоз→⊣нейрон
  model <lif|izh|hh|haap>     модель симуляции по умолчанию
  trace on|off                трасса сигнала в 3D
  camera <yaw> <pitch>        ракурс 3D (пример: camera 0 -1.5)
  clear                       очистить вывод
Клик по #индексу в выводе выбирает нейрон.";

impl ConnectomeMode {
    /// Консоль (вкладка нижнего дока): ScrollArea с выводимыми строками
    /// (кликабельные `#N` — выбирают нейрон) + строка ввода с историей.
    pub(crate) fn console_view(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .max_height(ui.available_height() - 40.0)
            .show(ui, |ui| {
                let mut clicked = None;
                for l in &self.console.lines {
                    if let Some(n) = l.neuron {
                        // Строка с #индексом — кликабельная.
                        let resp = ui.add(
                            egui::Label::new(egui::RichText::new(&l.text).monospace())
                                .sense(egui::Sense::click()),
                        );
                        if resp.on_hover_text("выбрать нейрон #").clicked() {
                            clicked = Some(n);
                        }
                    } else {
                        ui.monospace(&l.text);
                    }
                }
                if let Some(n) = clicked {
                    self.console_select(n);
                }
            });
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("❯");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.console.input)
                    .desired_width(ui.available_width() - 8.0)
                    .font(egui::TextStyle::Monospace)
                    .hint_text("stats | search gaba | ego 158 | fullsim --drive 17 --model haap --gh 0.8 | select 158 | help"),
            );
            // История ↑/↓.
            if resp.has_focus() {
                if ui.input(|i| i.key_pressed(Key::ArrowUp)) {
                    self.console.hist_prev();
                }
                if ui.input(|i| i.key_pressed(Key::ArrowDown)) {
                    self.console.hist_next();
                }
            }
            // Enter выполняет команду.
            if resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                let line = self.console.input.clone();
                self.console.input.clear();
                self.console.hist_idx = None;
                if !line.trim().is_empty() {
                    self.console.history.push(line.clone());
                }
                self.console_exec(&line);
                resp.request_focus();
            }
        });
    }

    /// Клик по `#индексу` в выводе: выбрать нейрон (карточка + ego-граф)
    /// и навести на него камеру.
    pub(crate) fn console_select(&mut self, idx: usize) {
        let ok = self.data.as_ref().is_some_and(|d| idx < d.n);
        if !ok {
            self.console.push_text(&format!("индекс #{idx} вне сети"));
            return;
        }
        self.select(idx);
        self.focus_on(idx);
        let (ty, nt) = {
            let d = self.data.as_ref().unwrap();
            (d.type_name(idx).to_string(), d.nt(idx).name().to_string())
        };
        self.console.push_text(&format!(
            "→ #{idx} {} · {nt}",
            if ty.is_empty() { "?".to_string() } else { ty }
        ));
    }

    /// Диспетчер команд консоли (общий синтаксис с CLI-комбайном).
    pub(crate) fn console_exec(&mut self, raw: &str) {
        let line = raw.trim().to_string();
        if line.is_empty() {
            return;
        }
        self.console.push_command(&line);
        let parts: Vec<&str> = line.split_whitespace().collect();
        let cmd = parts[0];
        match cmd {
            "help" => self.console.push_text(CONSOLE_HELP),
            "clear" => self.console.lines.clear(),
            "stats" => match self.data.as_ref() {
                Some(d) => self.console.push_text(&cli::stats_text(d)),
                None => self
                    .console
                    .push_text("нет данных — открой файл (Файл → Открыть)"),
            },
            "search" => {
                let q = parts[1..].join(" ");
                match self.data.as_ref() {
                    Some(d) => self.console.push_text(&cli::search_text(d, &q)),
                    None => self.console.push_text("нет данных"),
                }
            }
            "ego" => {
                let Some(idx) = parts.get(1).and_then(|s| s.parse::<usize>().ok()) else {
                    self.console.push_text("ego <idx> [--k N]");
                    return;
                };
                let Some(d) = self.data.as_mut() else {
                    self.console.push_text("нет данных");
                    return;
                };
                if idx >= d.n {
                    self.console
                        .push_text(&format!("индекс #{idx} вне сети ({})", d.n));
                    return;
                }
                match cli::ego_text(d, &parts) {
                    Ok(s) => self.console.push_text(&s),
                    Err(e) => self.console.push_text(&format!("⚠ {e}")),
                }
            }
            "sim" | "fullsim" => self.console_sim(&parts, cmd == "fullsim"),
            "disin" => {
                let Some(idx) = parts.get(1).and_then(|s| s.parse::<usize>().ok()) else {
                    self.console.push_text("disin <idx> [--top N]");
                    return;
                };
                let Some(d) = self.data.as_ref() else {
                    self.console.push_text("нет данных");
                    return;
                };
                if idx >= d.n {
                    self.console
                        .push_text(&format!("индекс #{idx} вне сети ({})", d.n));
                    return;
                }
                match cli::disin_text(d, &parts) {
                    Ok(s) => self.console.push_text(&s),
                    Err(e) => self.console.push_text(&format!("⚠ {e}")),
                }
            }
            "select" | "focus" => {
                let Some(idx) = parts.get(1).and_then(|s| s.parse::<usize>().ok()) else {
                    self.console
                        .push_text("select <idx> — выбрать нейрон и навести камеру");
                    return;
                };
                self.console_select(idx);
            }
            "model" => {
                let m = parts.get(1).map(|s| cli::parse_model(s)).unwrap_or(0);
                self.sim.model = m;
                self.console
                    .push_text(&format!("модель симуляции: {}", models::MODELS[m].name()));
            }
            "trace" => {
                let on = parts.get(1).is_none_or(|s| *s == "on");
                self.sim.show_trace = on;
                self.camera.atlas3d = None;
                self.console.push_text(&format!(
                    "трасса сигнала (3D): {}",
                    if on { "on" } else { "off" }
                ));
            }
            "camera" => {
                let yaw = parts.get(1).and_then(|s| s.parse::<f32>().ok());
                let pitch = parts.get(2).and_then(|s| s.parse::<f32>().ok());
                match (yaw, pitch) {
                    (Some(y), Some(p)) => {
                        self.camera.view3d = true;
                        self.camera.atlas = None;
                        self.set_camera(y, p);
                        self.console
                            .push_text(&format!("камера 3D: yaw={y} pitch={p}"));
                    }
                    _ => self
                        .console
                        .push_text("camera <yaw> <pitch> — пример: camera 0 -1.5"),
                }
            }
            _ => self
                .console
                .push_text(&format!("неизвестная команда: {cmd} — введи `help`")),
        }
    }

    /// `sim` / `fullsim` из консоли: результат ложится в `sim_result`
    /// (нижние графики) и, при полном прогоне, в 3D-трассу — как кнопка ▶.
    pub(crate) fn console_sim(&mut self, parts: &[&str], full: bool) {
        let drive = cli::flag_drive(parts).or(self.sel.selected);
        let Some(drive) = drive else {
            self.console
                .push_text("нужен --drive <idx> или сначала выбери нейрон кликом");
            return;
        };
        let model = cli::parse_model(cli::flag_val(parts, "--model", "lif"));
        let def_steps = if full { "200" } else { "800" };
        let steps = cli::flag_val(parts, "--steps", def_steps)
            .parse()
            .unwrap_or(if full { 200 } else { 800 });
        let bias = cli::flag_val(parts, "--bias", "20").parse().unwrap_or(20.0);
        let pulse = cli::flag_val(parts, "--pulse", "0.6")
            .parse()
            .unwrap_or(0.6);
        let gh = cli::flag_val(parts, "--gh", "0.3").parse().unwrap_or(0.3);
        // Кинетика препарата: флаги `--drug/--conc/--half-life` из консоли,
        // иначе — препарат карточки фармы.
        let pk = match cli::pharma_pk_from_flags(parts) {
            Ok(p) => p.or_else(|| self.pharma_pk()),
            Err(e) => {
                self.console.push_text(&format!("⚠ {e}"));
                return;
            }
        };
        let result = {
            let Some(d) = self.data.as_mut() else {
                self.console.push_text("нет данных");
                return;
            };
            if full {
                run_full_sim(d, drive, model, bias, steps, pulse, 0.5, gh, pk).map(SimOutcome::Full)
            } else {
                cli::run_local_sim(d, drive, model, steps, bias, pulse, gh, pk)
            }
        };
        match result {
            Ok(SimOutcome::Local(r)) => {
                let mut text = self
                    .data
                    .as_ref()
                    .map(|d| cli::local_sim_text(d, &r))
                    .unwrap_or_default();
                if !r.pharma_note.is_empty() {
                    text.push_str(&format!("\n{}", r.pharma_note));
                }
                let total: i32 = r.raster.iter().sum();
                let active = r.rates.iter().filter(|&&x| x > 0.0).count();
                let mut status = format!(
                    "{}: {} нейронов, {} спайков, активных {}/{}",
                    r.model,
                    r.idx.len(),
                    total,
                    active,
                    r.idx.len()
                );
                if !r.pharma_note.is_empty() {
                    status.push_str(&format!(" · {}", r.pharma_note));
                }
                self.sim.status = status;
                self.sim.result = Some(SimOutcome::Local(r));
                self.camera.atlas3d = None;
                self.console.push_text(&text);
                self.plots.enable_kind_if_none(true); // плоты ego-контура
            }
            Ok(SimOutcome::Full(r)) => {
                let mut text = self
                    .data
                    .as_ref()
                    .map(|d| cli::full_sim_text(d, &r))
                    .unwrap_or_default();
                if !r.pharma_note.is_empty() {
                    text.push_str(&format!("\n{}", r.pharma_note));
                }
                let mut status = format!(
                    "Полный VNC · {}: {} спайков, активных {}/{} · {} мс",
                    r.model,
                    r.total_spikes,
                    r.active_neurons,
                    self.data.as_ref().map(|d| d.n()).unwrap_or(0),
                    r.steps as f32 * r.dt_ms
                );
                if !r.pharma_note.is_empty() {
                    status.push_str(&format!(" · {}", r.pharma_note));
                }
                self.sim.status = status;
                self.sim.result = Some(SimOutcome::Full(r));
                self.camera.atlas3d = None;
                self.console.push_text(&text);
                self.plots.enable_kind_if_none(false); // научные плоты
            }
            Err(e) => self.console.push_text(&format!("⚠ {e}")),
        }
    }
}
