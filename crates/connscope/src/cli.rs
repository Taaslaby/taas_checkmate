//! CLI «комбайн» для экспериментов с коннектомами: статистика, поиск,
//! ego-граф, локальная и полная симуляция, растормаживание — без GUI.
//!
//! Все отчёты строятся как строки (`*_text`), чтобы их можно было
//! переиспользовать во встроенной консоли GUI (`crate::app::shell::console`).
//!
//! ```text
//! connscope stats <file>
//! connscope search <file> <запрос>
//! connscope ego <file> <idx> [--k 10]
//! connscope sim <file> --drive <idx> [--model lif|izh|hh|haap] [--steps 200] [--bias 20] [--pulse 0.6] [--gh 0.3] [--drug имя] [--conc µM] [--half-life шаги]
//! connscope fullsim <file> --drive <idx> [--model lif] [--steps 200] [--bias 30] [--pulse 0.6] [--drug имя] [--conc µM] [--half-life шаги]
//! connscope disin <file> <idx> [--top 10]
//! ```

use std::path::Path;
use std::time::Instant;

use crate::core::nt::Nt;
use crate::core::pharma::DrugLibrary;
use crate::io::data::MancData;
use crate::sim::models;
use crate::sim::{
    find_disinhibition, run_full_sim, run_simulation, FullSimResult, PharmaDose, PharmaPk,
    SimOutcome, SimResult,
};

pub fn run() -> i32 {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");
    if cmd == "help" || args.len() < 3 {
        print_help();
        return if cmd == "help" { 0 } else { 1 };
    }
    let path = Path::new(&args[2]);
    let mut data = match MancData::detect(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("⚠ {e}");
            return 1;
        }
    };
    let parts: Vec<&str> = args[3..].iter().map(|s| s.as_str()).collect();
    let ok = match cmd {
        "stats" => {
            println!("{}", stats_text(&data));
            true
        }
        "search" => {
            println!("{}", search_text(&data, &parts.join(" ")));
            true
        }
        "ego" => report_stdout(ego_text(&mut data, &parts)),
        "sim" => report_stdout(cmd_sim(&mut data, &parts)),
        "fullsim" => report_stdout(cmd_fullsim(&data, &parts)),
        "disin" => report_stdout(disin_text(&data, &parts)),
        _ => {
            eprintln!("неизвестная команда: {cmd}");
            print_help();
            false
        }
    };
    if ok {
        0
    } else {
        1
    }
}

fn report_stdout(r: Result<String, String>) -> bool {
    match r {
        Ok(s) => {
            println!("{s}");
            true
        }
        Err(e) => {
            eprintln!("⚠ {e}");
            false
        }
    }
}

fn print_help() {
    println!("neoMyelin — комбайн для коннектомики (любой .bin/.taas)");
    println!();
    println!("GUI (без команд):    connscope [data.bin]");
    println!();
    println!("Эксперименты (CLI):");
    println!("  connscope stats <file>");
    println!("  connscope search <file> <запрос>");
    println!("  connscope ego <file> <idx> [--k 10]");
    println!("  connscope sim <file> --drive <idx> [--model lif|izh|hh|haap] [--steps 200] [--bias 20] [--pulse 0.6] [--gh 0.3] [--drug имя] [--conc µM] [--half-life шаги]");
    println!("  connscope fullsim <file> --drive <idx> [--model lif] [--steps 200] [--bias 30] [--pulse 0.6] [--drug имя] [--conc µM] [--half-life шаги]");
    println!("  connscope disin <file> <idx> [--top 10]");
    println!("  connscope help");
}

// ── общие парсеры флагов (общие с консолью GUI) ──────────────

pub(crate) fn flag_val<'a>(parts: &'a [&'a str], key: &str, def: &'a str) -> &'a str {
    parts
        .iter()
        .position(|&a| a == key)
        .and_then(|i| parts.get(i + 1))
        .copied()
        .unwrap_or(def)
}

pub(crate) fn flag_drive(parts: &[&str]) -> Option<usize> {
    parts
        .iter()
        .position(|&a| a == "--drive")
        .and_then(|i| parts.get(i + 1))
        .and_then(|s| s.parse().ok())
}

/// Индекс модели по имени (`lif`/`izh`/`izhikevich`/`hh`/`haap`);
/// неизвестное имя → 0 (LIF).
pub(crate) fn parse_model(s: &str) -> usize {
    models::index_of_name(s).unwrap_or(0)
}

/// Кинетика препарата из флагов `--drug <имя> [--conc <µM>] [--half-life <шаги>]`.
/// `None` — флага `--drug` нет; `Err` — неизвестный препарат (с подсказкой).
pub(crate) fn pharma_pk_from_flags(parts: &[&str]) -> Result<Option<PharmaPk>, String> {
    let name = flag_val(parts, "--drug", "");
    if name.is_empty() {
        return Ok(None);
    }
    let lib = DrugLibrary::load();
    let Some(drug) = lib.get(name) else {
        let names: Vec<&str> = lib.drugs.iter().map(|d| d.name.as_str()).collect();
        return Err(format!(
            "неизвестный препарат: {name} (в библиотеке: {})",
            names.join(", ")
        ));
    };
    let conc = flag_val(parts, "--conc", "1.0")
        .parse()
        .map_err(|_| "--conc должен быть числом (µM)".to_string())?;
    let half_life: f32 = flag_val(parts, "--half-life", "0")
        .parse()
        .map_err(|_| "--half-life должен быть числом (шагов)".to_string())?;
    Ok(Some(PharmaPk {
        dose: PharmaDose {
            drug: drug.clone(),
            conc_um: conc,
        },
        half_life_steps: half_life.max(0.0),
    }))
}

pub(crate) fn class_name(v: i8) -> &'static str {
    match v {
        -1 => "все",
        0 => "other",
        1 => "motor",
        2 => "sensory",
        3 => "ascending",
        4 => "descending",
        5 => "intrinsic",
        6 => "glia",
        7 => "efferent",
        _ => "?",
    }
}

fn type_or_q(d: &MancData, i: usize) -> String {
    let t = d.type_name(i);
    if t.is_empty() {
        "?".into()
    } else {
        t.to_string()
    }
}

// ── отчёты (строки) ──────────────────────────────────────────

pub(crate) fn stats_text(d: &MancData) -> String {
    let npos = d.has_pos.iter().filter(|&&x| x != 0).count();
    let mut s = format!("нейронов:        {}\n", d.n);
    s.push_str(&format!("синапсов:        {}\n", d.s));
    s.push_str(&format!("с 3D-позицией:   {}\n", npos));
    s.push_str(&format!("типов клеток:    {}\n", d.types.len()));
    let mut nt = [0usize; 7];
    let mut klass = [0usize; 8];
    let mut total_deg = 0usize;
    let mut min_deg = usize::MAX;
    let mut max_deg = 0usize;
    for i in 0..d.n {
        nt[(d.nt[i] as usize).min(6)] += 1;
        klass[(d.klass[i] as usize).min(7)] += 1;
        let od = d.out_deg(i);
        total_deg += od;
        min_deg = min_deg.min(od);
        max_deg = max_deg.max(od);
    }
    s.push_str("\nNT:\n");
    for (k, &count) in nt.iter().enumerate() {
        let n = Nt::from_code(k as u8);
        s.push_str(&format!("  {:<14} {}\n", n.name(), count));
    }
    s.push_str("\nКлассы:\n");
    for (k, &count) in klass.iter().enumerate() {
        s.push_str(&format!("  {:<12} {}\n", class_name(k as i8), count));
    }
    if d.n > 0 {
        s.push_str(&format!(
            "\nout-степень: min={min_deg} max={max_deg} средняя={:.1}",
            total_deg as f64 / d.n as f64
        ));
    }
    s
}

pub(crate) fn search_text(d: &MancData, q: &str) -> String {
    let q = q.trim().to_lowercase();
    if q.is_empty() {
        return "нужен запрос: search <запрос>".into();
    }
    let mut s = String::new();
    let mut found = 0;
    for i in 0..d.n {
        let hay = format!(
            "{} {} {} {} {} {}",
            d.type_name(i),
            d.klass(i).name(),
            d.nt(i).name(),
            d.leg_name(i),
            d.side_name(i),
            d.body_id[i]
        )
        .to_lowercase();
        if hay.contains(&q) {
            s.push_str(&format!(
                "#{i}  {} · {} · {} · {} {} · body={} · out={}\n",
                type_or_q(d, i),
                d.klass(i).name(),
                d.nt(i).name(),
                d.leg_name(i),
                d.side_name(i),
                d.body_id[i],
                d.out_deg(i)
            ));
            found += 1;
            if found >= 20 {
                s.push_str("… (показано 20 из многих)");
                break;
            }
        }
    }
    if found == 0 {
        s = format!("ничего не найдено: {q}");
    }
    s.trim_end().to_string()
}

pub(crate) fn ego_text(d: &mut MancData, parts: &[&str]) -> Result<String, String> {
    let Some(idx) = parts.first().and_then(|s| s.parse::<usize>().ok()) else {
        return Err("нужен индекс нейрона: ego <idx> [--k N]".into());
    };
    if idx >= d.n {
        return Err(format!("индекс {idx} вне сети ({})", d.n));
    }
    let k = flag_val(parts, "--k", "10").parse().unwrap_or(10);
    let od = d.out_deg(idx);
    let id_ = d.in_deg(idx);
    let mut s = format!(
        "#{idx}  {} · {} · {} · out={od} in={id_}\n",
        type_or_q(d, idx),
        d.klass(idx).name(),
        d.nt(idx).name()
    );
    for &(t, w, out) in &d.ego_edges(idx, k) {
        let dir = if out { "→" } else { "←" };
        s.push_str(&format!(
            "  {dir} #{t}  w={w:.1}  {} · {}\n",
            type_or_q(d, t),
            d.nt(t).name()
        ));
    }
    Ok(s.trim_end().to_string())
}

pub(crate) fn disin_text(d: &MancData, parts: &[&str]) -> Result<String, String> {
    let Some(idx) = parts.first().and_then(|s| s.parse::<usize>().ok()) else {
        return Err("нужен индекс нейрона: disin <idx> [--top N]".into());
    };
    if idx >= d.n {
        return Err(format!("индекс {idx} вне сети ({})", d.n));
    }
    let n = Nt::from_code(d.nt[idx]);
    if !n.inhibitory() {
        return Ok(format!(
            "#{idx} {} — возбуждающий, растормаживания нет",
            n.name()
        ));
    }
    let top = flag_val(parts, "--top", "10").parse().unwrap_or(10);
    let mut s = format!("{} цепей тормоз→⊣тормоз→⊣нейрон от #{idx}:\n", top);
    for (mid, y, w1, w2) in find_disinhibition(d, idx, top) {
        s.push_str(&format!(
            "  #{y} {} ({}) через #{mid} w={w1:.0}·{w2:.0}\n",
            type_or_q(d, y),
            d.nt(y).name()
        ));
    }
    Ok(s.trim_end().to_string())
}

/// Локальная симуляция ego-контура вокруг драйва → готовый результат (для GUI).
pub(crate) fn run_local_sim(
    d: &mut MancData,
    drive: usize,
    model: usize,
    steps: usize,
    bias: f32,
    pulse: f32,
    gh: f32,
    pk: Option<PharmaPk>,
) -> Result<SimOutcome, String> {
    if drive >= d.n {
        return Err(format!("драйв #{drive} вне сети ({})", d.n));
    }
    let k = 60usize;
    let mut nodes = vec![drive];
    for &(t, _, _) in &d.ego_edges(drive, k) {
        if nodes.len() >= k {
            break;
        }
        if !nodes.contains(&t) {
            nodes.push(t);
        }
    }
    run_simulation(d, &nodes, drive, model, bias, steps, pulse, 0.5, gh, pk).map(SimOutcome::Local)
}

pub(crate) fn local_sim_text(d: &MancData, r: &SimResult) -> String {
    let dur_ms = r.steps as f32 * r.dt_ms;
    let mut s = format!(
        "{} · контур из {} нейронов · {} шагов · {} мс",
        r.model,
        r.idx.len(),
        r.steps,
        dur_ms
    );
    for i in 0..r.idx.len() {
        let spikes = (r.rates[i] * dur_ms / 1000.0) as i64;
        s.push_str(&format!(
            "\n  #{} {:<18} {:6.1} Гц  ({} спайков)",
            r.idx[i],
            type_or_q(d, r.idx[i]),
            r.rates[i],
            spikes
        ));
    }
    s
}

pub(crate) fn full_sim_text(d: &MancData, r: &FullSimResult) -> String {
    let mut s = format!(
        "Полный сим · {} · {} шагов · {} мс",
        r.model,
        r.steps,
        r.steps as f32 * r.dt_ms
    );
    s.push_str(&format!(
        "\nспайков: {} · активных нейронов: {}/{} ({:.1}%)",
        r.total_spikes,
        r.active_neurons,
        d.n,
        r.active_neurons as f64 * 100.0 / d.n as f64
    ));
    let mut cls = [0i64; 8];
    for ca in &r.class_activity {
        for k in 0..8 {
            cls[k] += ca[k] as i64;
        }
    }
    s.push_str("\n\nспайков по классам:");
    for (k, &count) in cls.iter().enumerate() {
        s.push_str(&format!("\n  {:<12} {}", class_name(k as i8), count));
    }
    let mut order: Vec<usize> = (0..d.n).filter(|&i| r.first_fire[i] != u32::MAX).collect();
    order.sort_by(|&a, &b| r.spike_counts[b].cmp(&r.spike_counts[a]));
    s.push_str("\n\nтоп-10 активных нейронов (первый спайк на шаге):");
    for &i in order.iter().take(10) {
        s.push_str(&format!(
            "\n  #{} {:<18} {} спайков · шаг {}",
            i,
            type_or_q(d, i),
            r.spike_counts[i],
            r.first_fire[i]
        ));
    }
    s
}

// ── команды CLI (тонкие обёртки над отчётами) ────────────────

fn cmd_sim(d: &mut MancData, parts: &[&str]) -> Result<String, String> {
    let Some(drive) = flag_drive(parts) else {
        return Err("нужен --drive <idx>: sim <file> --drive 158".into());
    };
    let model = parse_model(flag_val(parts, "--model", "lif"));
    let steps = flag_val(parts, "--steps", "200").parse().unwrap_or(200);
    let bias = flag_val(parts, "--bias", "20").parse().unwrap_or(20.0);
    let pulse = flag_val(parts, "--pulse", "0.6").parse().unwrap_or(0.6);
    let gh = flag_val(parts, "--gh", "0.3").parse().unwrap_or(0.3);
    let pk = pharma_pk_from_flags(parts)?;
    let t0 = Instant::now();
    match run_local_sim(d, drive, model, steps, bias, pulse, gh, pk)? {
        SimOutcome::Local(r) => {
            let mut s = local_sim_text(d, &r);
            if !r.pharma_note.is_empty() {
                s.push_str(&format!("\n{}", r.pharma_note));
            }
            s.push_str(&format!("\n({:?})", t0.elapsed()));
            Ok(s)
        }
        _ => Err("неожиданный результат".into()),
    }
}

fn cmd_fullsim(d: &MancData, parts: &[&str]) -> Result<String, String> {
    let Some(drive) = flag_drive(parts) else {
        return Err("нужен --drive <idx>: fullsim <file> --drive 158".into());
    };
    let model = parse_model(flag_val(parts, "--model", "lif"));
    let steps = flag_val(parts, "--steps", "200").parse().unwrap_or(200);
    let bias = flag_val(parts, "--bias", "30").parse().unwrap_or(30.0);
    let pulse = flag_val(parts, "--pulse", "0.6").parse().unwrap_or(0.6);
    let gh = flag_val(parts, "--gh", "0.3").parse().unwrap_or(0.3);
    let pk = pharma_pk_from_flags(parts)?;
    let t0 = Instant::now();
    let r = run_full_sim(d, drive, model, bias, steps, pulse, 0.5, gh, pk)?;
    let mut s = full_sim_text(d, &r);
    if !r.pharma_note.is_empty() {
        s.push_str(&format!("\n{}", r.pharma_note));
    }
    s.push_str(&format!("\n({:?})", t0.elapsed()));
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::pharma_pk_from_flags;
    use crate::core::pharma::Effect;

    /// `--drug` без `--conc` → 1 µM по умолчанию; без флагов → None.
    #[test]
    fn dose_flags_parse() {
        let none = pharma_pk_from_flags(&["--drive", "5"]).unwrap();
        assert!(none.is_none(), "без --drug дозы нет");

        let d = pharma_pk_from_flags(&["--drug", "TTX"]).unwrap().unwrap();
        assert_eq!(d.dose.drug.name, "TTX");
        assert_eq!(d.dose.drug.effect, Effect::Block);
        assert_eq!(d.dose.conc_um, 1.0, "по умолчанию 1 µM");
        assert_eq!(d.half_life_steps, 0.0, "по умолчанию постоянная");

        let d = pharma_pk_from_flags(&["--drug", "ttx", "--conc", "0.01"])
            .unwrap()
            .unwrap();
        assert_eq!(d.dose.conc_um, 0.01);
        assert!(
            (d.dose.drug.scale(d.dose.conc_um) - 0.5).abs() < 1e-3,
            "IC50 → scale 0.5"
        );

        let d = pharma_pk_from_flags(&["--drug", "TTX", "--half-life", "150"])
            .unwrap()
            .unwrap();
        assert_eq!(d.half_life_steps, 150.0);
        // Через 150 шагов концентрация вдвое меньше — проверка кинетики.
        assert!((d.conc_at(150) - d.conc_at(0) / 2.0).abs() < 1e-3);
    }

    /// Регистронезависимый поиск + честная ошибка с подсказкой.
    #[test]
    fn dose_flags_unknown_drug_reports_library() {
        let err = pharma_pk_from_flags(&["--drug", "aspirin"]).unwrap_err();
        assert!(err.contains("aspirin"), "{err}");
        assert!(err.contains("TTX"), "подсказка со списком: {err}");
    }
}
