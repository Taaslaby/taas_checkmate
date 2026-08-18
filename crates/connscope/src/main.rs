//! # neoMyelin (бинарник connscope)
//!
//! Нативный egui-стенд для MANC VNC (102 369 нейронов, 5.2M синапсов):
//!
//! - **Атлас** — проекции XY/XZ/YZ по реальным voxel-координатам,
//!   нейроны окрашены по нейромедиатору (ACh возб. / GABA торм. / GLUT торм.).
//! - **Клик** — ego-граф нейрона (входящие/исходящие, топ по весу).
//! - **3D** — орбитальный вид облака точек (drag = вращение, перспектива).
//! - **Инспектор** — body_id, тип, класс, нога, сторона, НТ, степени.
//! - **Поиск** — по типу / классу / НТ / ноге / body_id.
//! - **Симуляция** — LIF / Izhikevich / HH / HAAP на локальном контуре,
//!   растр + частоты + мембранные потенциалы (V-трейсы), ползунок h-тока.
//! - **Растормаживание** — цепи `тормоз →⊣ тормоз →⊣ нейрон`.
//!
//! Модули: [`app`] (ядро GUI — воркбенч: `mode.rs` — трейт `Mode`
//! (контракт режима) + `ShellServices`; `shell/` — рамка приложения
//! (menu, status, shortcuts, profiler, theme); режимы `connectome/`
//! (`ConnectomeMode`: camera, picking, ego, sim_ui, canvas, plots, top,
//! rail, panels, tabs, console) и `neuroml/` (`NeuromlMode`)), [`render`]
//! (atlas 2D/3D + lod + splat-GPU), [`io`] (парсеры .bin/.taas/GraphML/npz),
//! [`core`] (НТ/классы/цвета), [`sim`] (симуляция), [`cli`] (CLI-комбайн).
//!
//! Тесты живут рядом с тестируемым кодом (`#[cfg(test)] mod tests`
//! в `render/atlas.rs`, `render/lod.rs`, `io/data.rs`, `sim/mod.rs` и др.).
//!
//! ## Данные
//! Самодостаточный `resources/examples/data/manc_viz.bin`, печётся один раз:
//! ```text
//! python scripts/bake_manc_viz.py
//! cargo run -p connscope --release
//! ```

mod app;
mod cli;
mod core;
mod io;
mod render;
mod sim;

/// Первая CLI-команда, если запущен не GUI.
fn is_cli_cmd(cmd: &str) -> bool {
    matches!(
        cmd,
        "stats" | "search" | "ego" | "sim" | "fullsim" | "disin" | "help"
    )
}

fn main() -> Result<(), eframe::Error> {
    // Ловим любую панику (в т.ч. в wgpu-колбэках) в connscope_panic.log —
    // иначе «вирубился» без единой строчки. Файл пишется рядом с CWD.
    install_panic_hook();
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && is_cli_cmd(&args[1]) {
        std::process::exit(cli::run());
    }
    app::run()
}

/// Пишет сообщение паники + файл:строка в `connscope_panic.log` и в stderr.
fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        use std::io::Write;
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic payload".to_owned());
        let loc = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "?".to_owned());
        let line = format!("[connscope] panic at {loc}: {msg}\n");
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("connscope_panic.log")
            .and_then(|mut f| f.write_all(line.as_bytes()));
        eprintln!("{line}");
    }));
}
