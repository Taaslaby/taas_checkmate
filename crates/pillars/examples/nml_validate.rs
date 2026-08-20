//! # NeuroML validator (`examples/nml_validate`)
//!
//! Стресс-тест парсера на **реальных** NeuroML2-файлах с GitHub. Проходит по
//! папке с `.nml`, пробует `load_file` + `run` каждый и честно репортит:
//! что загрузилось и спайкает, а что — нет и **почему**.
//!
//! ```text
//! cargo run -p pillars --example nml_validate -- <dir>
//! cargo run -p pillars --example nml_validate -- resources/examples/neuroml
//! ```
//!
//! Коды статуса:
//! * `OK   ` — загрузился и прогнался (спайки ≥ 0);
//! * `EMPTY` — загрузился, но 0 популяций (например, файл-зоопарк клеток без сети);
//! * `FAIL ` — парсер вернул понятную ошибку (фича вне сабсета).

use std::path::{Path, PathBuf};

use pillars::api::neuroml::load_file;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("resources/examples/neuroml"));
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("cannot read dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "nml"))
        .collect();
    files.sort();

    if files.is_empty() {
        eprintln!("no .nml files in {dir:?}");
        std::process::exit(2);
    }

    println!(
        "== NeuroML validation: {} file(s) in {dir:?}\n",
        files.len()
    );
    let mut ok = 0;
    let mut empty = 0;
    let mut fail = 0;
    for f in &files {
        print!(
            "{:<12} {}",
            status_tag(f),
            f.file_name().unwrap().to_string_lossy()
        );
        match validate(f) {
            Ok(Some((pops, syns, inputs, spikes))) => {
                ok += 1;
                println!(
                    "  pops={pops} syns={syns} inputs={inputs} spikes={}",
                    spikes.iter().sum::<usize>()
                );
            }
            Ok(None) => {
                empty += 1;
                println!("  (нет сети — файл-зоопарк компонентов)");
            }
            Err(e) => {
                fail += 1;
                println!("  ✗ {e}");
            }
        }
    }
    println!(
        "\n== итог: OK {ok}, EMPTY {empty}, FAIL {fail} (из {})",
        files.len()
    );
    if fail > 0 {
        std::process::exit(1);
    }
}

/// `OK` / `EMPTY` / `FAIL` — по предварительной эвристике, чтобы колонка была
/// читаемой до того, как мы увидим детали.
fn status_tag(f: &Path) -> &'static str {
    // Понятные фичи вне нашего сабсета → сразу FAIL (без лишнего шума).
    let bad = [
        "GapJunction",
        "AnalogSynapses",
        "gradedSynapse",
        "Inputs",
        "InhomogeneousParams",
        "FullCell",
        "FullNeuroML",
    ];
    let name = f.file_name().unwrap().to_string_lossy();
    if bad.iter().any(|b| name.contains(b)) {
        "FAIL"
    } else {
        "?"
    }
}

/// Итог валидации одного файла: (популяции, синапсы, входы, спайки по популяциям).
type Report = (usize, usize, usize, Vec<usize>);

/// Пробуем загрузить и прогнать. `Ok(None)` = загрузился, но 0 популяций.
fn validate(f: &Path) -> Result<Option<Report>, String> {
    let mut ln = load_file(f).map_err(|e| e.to_string())?;
    let pops = ln.network.num_populations();
    // Печатаем предупреждения о молчаливо пропущенных элементах (gap
    // junctions, graded synapses, …) — это честный сигнал «частичной» загрузки.
    for w in &ln.warnings {
        println!("      ⚠ {w}");
    }
    if pops == 0 {
        return Ok(None);
    }
    let syns = ln.network.graph().map(|g| g.num_synapses()).unwrap_or(0);
    let inputs = ln.inputs.len();
    let res = ln.run();
    Ok(Some((pops, syns, inputs, res.spike_counts)))
}
