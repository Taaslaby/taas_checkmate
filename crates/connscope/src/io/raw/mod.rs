//! # Прямые форматы коннектомов (`raw`)
//!
//! Парсит сырые данные **без конвертации в .bin** — «данные разрознены,
//! но это один и тот же граф»:
//!
//! * **GraphML** (`.graphml`) — XML от networkx/igraph: узлы с именами
//!   и атрибутами (тип/класс/НТ), рёбра с весами, опционально координаты
//!   x/y/z. Если координат нет — синтетическая сфера (Фибоначчи), чтобы
//!   мозг был виден на атласе (статус помечает «синтетика»).
//! * **FAFB npz** (`.npz`) — zip-архив numpy-массивов CSR
//!   (`indptr/indices/data/neuron_ids/n_neurons`) + **координаты вторым
//!   файлом** `*_positions.npy` рядом + аннотации flywire
//!   (`consolidated_cell_types.csv.gz`, `classification.csv.gz`) — джойн
//!   по root-id (младшие 9 цифр: полный id = префикс `720575940` + номер).
//!
//! Модули-парсеры: [`graphml`] и [`fafb_npz`]; общие хелперы
//! (интернирование строк, коды НТ/классов, нормировка позиций,
//! синтетическая сфера) живут здесь.

mod fafb_npz;
mod graphml;

pub(crate) use fafb_npz::from_fafb_npz;
pub(crate) use graphml::from_graphml;

use std::collections::HashMap;

// ── интернирование строк (как в bake_manc_viz.py) ─────────────
fn intern(values: &[String]) -> Result<(Vec<u16>, Vec<String>), String> {
    let mut table: Vec<String> = Vec::new();
    let mut idx: Vec<u16> = Vec::with_capacity(values.len());
    let mut lookup: HashMap<String, u16> = HashMap::new();
    for v in values {
        let s = if v.is_empty() {
            "None".to_string()
        } else {
            v.clone()
        };
        if let Some(&i) = lookup.get(&s) {
            idx.push(i);
        } else {
            if table.len() >= u16::MAX as usize {
                return Err(format!(
                    "слишком много уникальных типов: {} (>65535, лимит u16)",
                    table.len() + 1
                ));
            }
            let i = table.len() as u16;
            table.push(s.clone());
            lookup.insert(s, i);
            idx.push(i);
        }
    }
    Ok((idx, table))
}

fn nt_code(label: &str) -> u8 {
    match label.trim().to_lowercase().as_str() {
        "acetylcholine" | "ach" | "cholinergic" => 1,
        "gaba" | "gabaergic" => 2,
        "glutamate" | "glu" | "glutamatergic" => 3,
        "da" | "dopamine" | "dopaminergic" => 4,
        "ser" | "serotonin" | "5ht" | "serotonergic" => 5,
        "oct" | "octopamine" | "octopaminergic" => 6,
        _ => 0,
    }
}

fn klass_code(label: &str) -> u8 {
    let s = label.trim().to_lowercase();
    if s.contains("motor") {
        1
    } else if s.contains("sensory") || s.contains("afferent") {
        2
    } else if s.contains("ascending") {
        3
    } else if s.contains("descending") {
        4
    } else if s.contains("intrinsic") || s.contains("interneuron") || s.contains("intern") {
        5
    } else if s.contains("glia") || s.contains("astro") {
        6
    } else if s.contains("efferent") {
        7
    } else {
        0
    }
}

fn norm_positions(pos: &[[f64; 3]], valid: &[bool]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<u8>) {
    let n = pos.len();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for i in 0..n {
        if valid[i] {
            for k in 0..3 {
                lo[k] = lo[k].min(pos[i][k]);
                hi[k] = hi[k].max(pos[i][k]);
            }
        }
    }
    let span = |k: usize| {
        if hi[k] - lo[k] > 0.0 {
            hi[k] - lo[k]
        } else {
            1.0
        }
    };
    let mut xs = vec![0.0f32; n];
    let mut ys = vec![0.0f32; n];
    let mut zs = vec![0.0f32; n];
    let mut has = vec![0u8; n];
    for i in 0..n {
        if valid[i] {
            xs[i] = ((pos[i][0] - lo[0]) / span(0)) as f32;
            ys[i] = ((pos[i][1] - lo[1]) / span(1)) as f32;
            zs[i] = ((pos[i][2] - lo[2]) / span(2)) as f32;
            has[i] = 1;
        }
    }
    (xs, ys, zs, has)
}

/// Синтетическая раскладка — равномерные точки на сфере (Фибоначчи):
/// детерминированная, O(n), подходит для мозгов без координат.
fn fibonacci_sphere(n: usize) -> Vec<[f64; 3]> {
    let ga = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let th = i as f64 * ga;
            [r * th.cos(), y, r * th.sin()]
        })
        .collect()
}

fn centroid(pos: &[[f64; 3]], valid: &[bool]) -> [f64; 3] {
    let mut c = [0.0f64; 3];
    let mut k = 0usize;
    for i in 0..pos.len() {
        if valid[i] {
            for d in 0..3 {
                c[d] += pos[i][d];
            }
            k += 1;
        }
    }
    if k > 0 {
        c.iter_mut().for_each(|v| *v /= k as f64);
    }
    c
}
