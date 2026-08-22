//! # grow_engine — развитие → структура → функция (общий модуль)
//!
//! Логика эксперимента «из генома — в гамма-ритм», вынесенная из примера
//! `genome_to_gamma.rs` в библиотеку, чтобы CLI-пример и GUI (connscope,
//! режим «Развитие») делили **один код**:
//!
//! ```text
//!   prometheus-grow (развитие)            pillars (динамика)
//! ──────────────────────────────────────────────────────────────
//!   клетки секретируют морфоген "A"  ──►  ConnectomeData (CSR)
//!   конусы роста лезут по градиенту  ──►  BrainRunner, SIMD LIF-α
//!   аксоны встречают сому → синапсы       (модель Shiu et al. 2024)
//!   └─► локально-кластерный E/I граф      └─► пространственно
//!                                            когерентные ансамбли
//! ```
//!
//! Публичный API: [`GrowConfig`] (крутилки эксперимента), [`grow_circuit`]
//! (рост), [`build_runner`] (мост в симулятор), [`shuffle_targets`] (контроль
//! с теми же степенями) и набор аналитических функций ([`clustering_coeff`],
//! [`spatial_corr`], [`dominant_freq`], [`osc_strength`]).

use crate::{BrainRunner, NeuronConfig};
use prometheus_grow::{Bounds, CellType, ChemoAttraction, Grid, Secretor, Substance, Vec3, World};

/// Все крутилки эксперимента. Меняйте поля — и рост, и симуляция пересоберутся.
#[derive(Clone, Debug)]
pub struct GrowConfig {
    /// Возбуждающие нейроны (растят 2 аксона).
    pub n_e: usize,
    /// Тормозные нейроны (растят 1 аксон).
    pub n_i: usize,
    /// Радиус шара, в котором размещаются сомы.
    pub cluster_radius: f64,
    /// Радиус контакта для образования синапса.
    pub synapse_dist: f64,
    /// Шагов развития (диффузия + рост + замыкание синапсов).
    pub grow_steps: usize,
    /// Шаг симуляции, мс.
    pub dt_ms: f32,
    /// Шагов симуляции (всего «времени» = dt_ms × sim_steps).
    pub sim_steps: usize,
    /// Тонический внешний ток, добавляемый к v на каждом шаге (мВ/шаг).
    pub drive: f32,
    /// Множитель возбуждающих весов.
    pub w_e_gain: f32,
    /// Множитель тормозных весов (E/I баланс: I ≈ 5× E).
    pub w_i_gain: f32,
    /// Постоянная времени альфа-синапса, мс (короче → быстрее петля E/I).
    pub tau_syn: f32,
    /// Сид роста и симуляции (детерминизм).
    pub seed: u64,
}

impl Default for GrowConfig {
    fn default() -> Self {
        Self {
            n_e: 160,
            n_i: 40,
            cluster_radius: 65.0,
            synapse_dist: 13.0,
            grow_steps: 300,
            dt_ms: 0.5,
            sim_steps: 50_000,
            drive: 0.2,
            w_e_gain: 0.15,
            w_i_gain: 0.8,
            tau_syn: 3.0,
            seed: 0xC0_FF_EE,
        }
    }
}

impl GrowConfig {
    pub fn n(&self) -> usize {
        self.n_e + self.n_i
    }
}

/// Результат развития: всё, что нужно и для симуляции, и для отрисовки.
#[derive(Clone, Debug)]
pub struct GrownCircuit {
    /// (pre, post, weight) — направленные синапсы, знак берётся из `nt`.
    pub edges: Vec<(usize, usize, f32)>,
    /// +1 возбуждающий, −1 тормозной.
    pub nt: Vec<f32>,
    /// Сомы: [x, y, z] в единицах мира.
    pub positions: Vec<[f64; 3]>,
    /// Нейриты: cell → neurite → ломаная [x, y, z].
    pub neurites: Vec<Vec<Vec<[f64; 3]>>>,
    /// Стенных секунд на развитие.
    pub growth_s: f64,
}

// ─── Тонкий детерминированный RNG (без лишних зависимостей) ─────────────────
struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ─── 1. Развитие ────────────────────────────────────────────────────────────
/// Вырастить E/I микросхему из правил развития: клетки секретируют морфоген,
/// аксоны лезут по градиенту и останавливаются после 30 µm (короткодействие →
/// локальность), синапсы замыкаются контактом в радиусе `synapse_dist`.
pub fn grow_circuit(cfg: &GrowConfig) -> GrownCircuit {
    let n = cfg.n();
    let grid = Grid::new(48, 48, 48, 5.0, Vec3::new(-120.0, -120.0, -120.0));
    let bounds = Bounds {
        min: Vec3::new(-120.0, -120.0, -120.0),
        max: Vec3::new(120.0, 120.0, 120.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 7);
    world.synapse_dist = cfg.synapse_dist;

    world.add_substance(Substance::new("A", 0.2, 0.001, &world.grid, vec![]));

    // Сомы — равномерно в шаре (детерминированно по сиду).
    let mut rng = XorShift(cfg.seed);
    let mut positions: Vec<Vec3> = Vec::with_capacity(n);
    while positions.len() < n {
        let (u, v) = (rng.unit() * 2.0 - 1.0, rng.unit() * 2.0 * std::f64::consts::PI);
        let c = u.cos();
        let r = cfg.cluster_radius * rng.unit().cbrt();
        let xy = (1.0 - u * u).max(0.0).sqrt() * r;
        positions.push(Vec3::new(xy * v.cos(), xy * v.sin(), c * r));
    }

    for (i, p) in positions.iter().enumerate() {
        let ty = if i < cfg.n_e {
            CellType::Excitatory
        } else {
            CellType::Inhibitory
        };
        let c = world.add_cell(*p, ty, 4.0);
        // Сома — источник морфогена (самоконструкция в духе CX3D).
        world.add_soma_module(c, Box::new(Secretor::new("A", 0.06)));
        // E-клетки растят 2 аксона, I — 1: богаче рекуррентное возбуждение,
        // плотнее торможение — классический рецепт гамма-ритма.
        let axons = if i < cfg.n_e { 2 } else { 1 };
        for _ in 0..axons {
            let nb = world.new_neurite(c, true, 0.5);
            let mut chemo = ChemoAttraction::new("A");
            chemo.noise_weight = 0.12;
            chemo.gradient_weight = 0.35;
            chemo.stop_concentration = 0.5;
            chemo.max_len_um = Some(30.0); // короткодействующие аксоны → локальность
            world.add_neurite_module(c, nb, Box::new(chemo));
        }
    }

    let t0 = std::time::Instant::now();
    for _ in 0..cfg.grow_steps {
        world.step();
    }
    let growth_s = t0.elapsed().as_secs_f64();

    // Нейриты для отрисовки.
    let neurites = world
        .cells
        .iter()
        .map(|c| {
            c.neurites
                .iter()
                .map(|nb| nb.points.iter().map(|p| [p.x, p.y, p.z]).collect())
                .collect()
        })
        .collect();

    let g = world.to_graph();
    let nt: Vec<f32> = g
        .nodes
        .iter()
        .map(|nd| match nd.cell_type {
            CellType::Inhibitory => -1.0,
            _ => 1.0,
        })
        .collect();
    let edges: Vec<(usize, usize, f32)> = g
        .edges
        .iter()
        .map(|e| (e.pre, e.post, e.weight as f32))
        .collect();

    GrownCircuit {
        edges,
        nt,
        positions: positions.iter().map(|p| [p.x, p.y, p.z]).collect(),
        neurites,
        growth_s,
    }
}

// ─── 2. Мост: граф → CSR → BrainRunner ──────────────────────────────────────
/// Собрать `BrainRunner` (SIMD LIF-α) из выращенной схемы. Веса масштабируются
/// по знаку пресинаптической клетки (E/I баланс), знак — от типа клетки.
pub fn build_runner(circuit: &GrownCircuit, cfg: &GrowConfig) -> BrainRunner {
    let n = cfg.n();
    let mut per: Vec<Vec<(usize, f32)>> = vec![vec![]; n];
    for &(pre, post, w) in &circuit.edges {
        let gain = if circuit.nt[pre] > 0.0 {
            cfg.w_e_gain
        } else {
            cfg.w_i_gain
        };
        per[pre].push((post, w * gain));
    }
    let mut row_ptr = vec![0i32; n + 1];
    let mut ci = Vec::with_capacity(circuit.edges.len());
    let mut vals = Vec::with_capacity(circuit.edges.len());
    for (i, list) in per.iter().enumerate() {
        row_ptr[i] = ci.len() as i32;
        for &(post, w) in list {
            ci.push(post as i32);
            vals.push(w);
        }
    }
    row_ptr[n] = ci.len() as i32;
    let conn = crate::ConnectomeData::new(row_ptr, ci, vals, circuit.nt.clone(), vec![]);
    BrainRunner::new(
        conn,
        NeuronConfig::LifAlpha {
            dt_ms: cfg.dt_ms,
            tau_ref: 2.2,
            tau_m: 20.0,
            v_rest: -52.0,
            v_thresh: -45.0,
            v_reset: -52.0,
            tau_syn: cfg.tau_syn,
            t_delay: 1.8,
        },
    )
    .expect("BrainRunner init")
}

// ─── 3. Контроль: та же степень, геометрия уничтожена ──────────────────────
/// Нулевая модель: перемешать мишени синапсов, сохранив степенную
/// последовательность. Для каждого источника — `d_i` случайных различных
/// мишеней из пула (конфигурационная модель). Всегда завершается: перестановка
/// пула Фишером–Йейтсом не требует повторных попыток.
pub fn shuffle_targets(edges: &[(usize, usize, f32)], n: usize, seed: u64) -> Vec<(usize, usize, f32)> {
    let mut rng = XorShift(seed);
    let mut out: Vec<(usize, usize, f32)> = Vec::with_capacity(edges.len());
    let mut cursor = 0usize;
    while cursor < edges.len() {
        let src = edges[cursor].0;
        let mut end = cursor;
        while end < edges.len() && edges[end].0 == src {
            end += 1;
        }
        let d = end - cursor;
        let mut pool: Vec<usize> = (0..n).filter(|&x| x != src).collect();
        for i in (1..pool.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            pool.swap(i, j);
        }
        for k in 0..d {
            out.push((src, pool[k], edges[cursor + k].2));
        }
        cursor = end;
    }
    out
}

// ─── 4. Анализ ──────────────────────────────────────────────────────────────
/// Ненаправленный коэффициент кластеризации — структурный отпечаток
/// пространственного замыкания (у геометрического графа он высок).
pub fn clustering_coeff(edges: &[(usize, usize, f32)], n: usize) -> f64 {
    let mut adj: Vec<std::collections::HashSet<usize>> = vec![std::collections::HashSet::new(); n];
    for &(a, b, _) in edges {
        adj[a].insert(b);
        adj[b].insert(a);
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        let neigh: Vec<usize> = adj[i].iter().copied().collect();
        let k = neigh.len();
        if k < 2 {
            continue;
        }
        let mut links = 0usize;
        for a in 0..k {
            for b in (a + 1)..k {
                if adj[neigh[a]].contains(&neigh[b]) {
                    links += 1;
                }
            }
        }
        total += 2.0 * links as f64 / (k * (k - 1)) as f64;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Популяционная частота, бины по `bin_ms`.
pub fn population_rate(spikes: &[i32], n: usize, steps: usize, dt_ms: f32, bin_ms: f32) -> Vec<f32> {
    let bin = (bin_ms / dt_ms).round().max(1.0) as usize;
    let nb = steps.div_ceil(bin);
    let mut rate = vec![0.0f32; nb];
    for t in 0..steps {
        let mut s = 0i32;
        for i in 0..n {
            s += spikes[t * n + i];
        }
        rate[t / bin] += s as f32;
    }
    let per_sec = 1000.0 / bin_ms;
    for r in rate.iter_mut() {
        *r *= per_sec / n as f32; // Гц (среднее по популяции)
    }
    rate
}

/// БПФ (рад-2, на месте): реальный вход, комплексный выход.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0usize;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..(len / 2) {
                let u = i + k;
                let v = u + len / 2;
                let (tr, ti) = (cr * re[v] - ci * im[v], cr * im[v] + ci * re[v]);
                re[v] = re[u] - tr;
                im[v] = im[u] - ti;
                re[u] += tr;
                im[u] += ti;
                let (ncr, nci) = (cr * wr - ci * wi, cr * wi + ci * wr);
                cr = ncr;
                ci = nci;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Спектр мощности популяционной частоты: (частоты, мощности).
pub fn spectrum(rate: &[f32], bin_ms: f32) -> (Vec<f64>, Vec<f64>) {
    let n = rate.len().next_power_of_two();
    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];
    for (i, r) in rate.iter().enumerate() {
        re[i] = *r as f64;
    }
    fft(&mut re, &mut im);
    let fs = 1000.0 / bin_ms as f64;
    let mut freqs = Vec::with_capacity(n / 2);
    let mut powers = Vec::with_capacity(n / 2);
    for k in 0..(n / 2) {
        freqs.push(k as f64 * fs / n as f64);
        powers.push(re[k] * re[k] + im[k] * im[k]);
    }
    (freqs, powers)
}

/// Доминантная частота в [lo, hi] Гц и её пик-к-фону отношение
/// (спектральная концентрация: 1.0 = плоский шум, ≫1 = осцилляция).
pub fn dominant_freq(rate: &[f32], bin_ms: f32, lo: f64, hi: f64) -> (f64, f64) {
    let (freqs, powers) = spectrum(rate, bin_ms);
    let mut best_f = 0.0f64;
    let mut best_p = 0.0f64;
    let mut band_sum = 0.0f64;
    let mut band_n = 0usize;
    for (f, p) in freqs.iter().zip(powers.iter()) {
        if *f >= lo && *f <= hi {
            band_sum += p;
            band_n += 1;
            if *p > best_p {
                best_p = *p;
                best_f = *f;
            }
        }
    }
    if band_n == 0 || best_p == 0.0 {
        return (0.0, 1.0);
    }
    (best_f, best_p / (band_sum / band_n as f64))
}

/// Сила осцилляции: пик нормированной автокорреляции сглаженной частоты
/// (0 = шум, ~1 = идеальная периодичность).
pub fn osc_strength(rate: &[f32], bin_ms: f32) -> f64 {
    let win: Vec<f32> = (0..(20.0 / bin_ms) as usize).map(|_| 1.0).collect();
    let n = rate.len();
    let hw = win.len() / 2;
    let mut s = vec![0.0f32; n];
    for i in 0..n {
        let mut acc = 0.0;
        for (k, &w) in win.iter().enumerate() {
            let j = i as isize + k as isize - hw as isize;
            if j >= 0 && j < n as isize {
                acc += rate[j as usize] * w;
            }
        }
        s[i] = acc;
    }
    let mean = s.iter().sum::<f32>() / n as f32;
    let mut var = 0.0f64;
    for x in &s {
        var += (*x as f64 - mean as f64).powi(2);
    }
    if var < 1e-9 {
        return 0.0;
    }
    let max_lag = ((200.0 / bin_ms) as usize).min(n / 4);
    let min_lag = ((10.0 / bin_ms) as usize).max(1);
    let mut best = 0.0f64;
    for lag in min_lag..max_lag {
        let mut acc = 0.0f64;
        for i in 0..(n - lag) {
            acc += (s[i] as f64 - mean as f64) * (s[i + lag] as f64 - mean as f64);
        }
        let c = acc / var;
        if c > best {
            best = c;
        }
    }
    best
}

/// Средняя попарная корреляция спайков как функция расстояния между сомами.
///
/// Бьёт трейн каждого нейрона в бины `bin_ms`, для каждой пары считает
/// корреляцию Пирсона бинов и евклидово расстояние между сомами. Возвращает
/// (середина корзины, средняя корреляция) для `n_buckets` корзин. Выращенные
/// схемы корофируют локально (корреляция спадает с расстоянием); у
/// геометрически-слепого контроля она плоская.
pub fn spatial_corr(
    spikes: &[i32],
    n: usize,
    steps: usize,
    dt_ms: f32,
    bin_ms: f32,
    positions: &[[f64; 3]],
    n_buckets: usize,
) -> Vec<(f64, f64)> {
    let bin = (bin_ms / dt_ms).round().max(1.0) as usize;
    let nb = steps.div_ceil(bin);
    let mut trains = vec![0.0f64; n * nb];
    for t in 0..steps {
        for i in 0..n {
            trains[i * nb + t / bin] += spikes[t * n + i] as f64;
        }
    }
    let mut mean = vec![0.0f64; n];
    for i in 0..n {
        mean[i] = trains[i * nb..(i + 1) * nb].iter().sum::<f64>() / nb as f64;
    }
    let mut max_d = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist3(&positions[i], &positions[j]);
            if d > max_d {
                max_d = d;
            }
        }
    }
    let bw = (max_d / n_buckets as f64).max(1e-9);
    let mut sum = vec![0.0f64; n_buckets];
    let mut cnt = vec![0usize; n_buckets];
    for i in 0..n {
        let mut vi = 0.0f64;
        for k in 0..nb {
            let d = trains[i * nb + k] - mean[i];
            vi += d * d;
        }
        if vi < 1e-12 {
            continue;
        }
        for j in (i + 1)..n {
            let mut vj = 0.0f64;
            for k in 0..nb {
                let d = trains[j * nb + k] - mean[j];
                vj += d * d;
            }
            if vj < 1e-12 {
                continue;
            }
            let mut cov = 0.0f64;
            for k in 0..nb {
                cov += (trains[i * nb + k] - mean[i]) * (trains[j * nb + k] - mean[j]);
            }
            let c = cov / (vi * vj).sqrt();
            let d = dist3(&positions[i], &positions[j]);
            let b = ((d / bw) as usize).min(n_buckets - 1);
            sum[b] += c;
            cnt[b] += 1;
        }
    }
    (0..n_buckets)
        .map(|b| {
            let mid = (b as f64 + 0.5) * bw;
            let m = if cnt[b] > 0 { sum[b] / cnt[b] as f64 } else { 0.0 };
            (mid, m)
        })
        .collect()
}

fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Один полный прогон: рост → мост → симуляция (grown и shuffled) → анализ.
/// Удобен для GUI и CLI: возвращает все числа и кривые, нужные для отчёта.
#[derive(Clone, Debug)]
pub struct GrowReport {
    pub circuit: GrownCircuit,
    pub shuffled: Vec<(usize, usize, f32)>,
    pub cc_grown: f64,
    pub cc_shuf: f64,
    pub sim_s: f64,
    pub sim_s_shuf: f64,
    pub spikes_grown: Vec<i32>,
    pub spikes_shuf: Vec<i32>,
    pub rates_g: Vec<f32>,
    pub rates_s: Vec<f32>,
    pub pop_grown: Vec<f32>,
    pub pop_shuf: Vec<f32>,
    pub fg: f64,
    pub cg: f64,
    pub fs: f64,
    pub cs: f64,
    pub og: f64,
    pub os: f64,
    pub scg: Vec<(f64, f64)>,
    pub scs: Vec<(f64, f64)>,
}

/// Запустить весь конвейер: рост → мост → симуляция выращенного и контроля →
/// анализ (популяционная частота, спектр, осцилляция, пространственная
/// синхронизация, кластеризация).
pub fn run_experiment(cfg: &GrowConfig) -> GrowReport {
    let n = cfg.n();

    let circuit = grow_circuit(cfg);
    let shuffled = shuffle_targets(&circuit.edges, n, cfg.seed ^ 0xDEAD_BEEF);
    let cc_grown = clustering_coeff(&circuit.edges, n);
    let cc_shuf = clustering_coeff(&shuffled, n);

    let mut grown = build_runner(&circuit, cfg);
    // У контроля свои рёбра — пересобираем CSR через отдельный runner.
    let shuf_edges_circuit = GrownCircuit {
        edges: shuffled.clone(),
        ..circuit.clone()
    };
    let mut shuf_runner = build_runner(&shuf_edges_circuit, cfg);

    let ext = vec![cfg.drive; n];

    let t0 = std::time::Instant::now();
    let spikes_grown = grown.run(cfg.sim_steps, &ext).to_vec();
    let sim_s = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let spikes_shuf = shuf_runner.run(cfg.sim_steps, &ext).to_vec();
    let sim_s_shuf = t0.elapsed().as_secs_f64();

    let rates_g = grown.firing_rates(cfg.dt_ms);
    let rates_s = shuf_runner.firing_rates(cfg.dt_ms);

    let bin_ms = 4.0;
    let pop_grown = population_rate(&spikes_grown, n, cfg.sim_steps, cfg.dt_ms, bin_ms);
    let pop_shuf = population_rate(&spikes_shuf, n, cfg.sim_steps, cfg.dt_ms, bin_ms);
    let (fg, cg) = dominant_freq(&pop_grown, bin_ms, 20.0, 120.0);
    let (fs, cs) = dominant_freq(&pop_shuf, bin_ms, 20.0, 120.0);
    let og = osc_strength(&pop_grown, bin_ms);
    let os = osc_strength(&pop_shuf, bin_ms);
    let scg = spatial_corr(&spikes_grown, n, cfg.sim_steps, cfg.dt_ms, 50.0, &circuit.positions, 5);
    let scs = spatial_corr(&spikes_shuf, n, cfg.sim_steps, cfg.dt_ms, 50.0, &circuit.positions, 5);

    GrowReport {
        circuit,
        shuffled,
        cc_grown,
        cc_shuf,
        sim_s,
        sim_s_shuf,
        spikes_grown,
        spikes_shuf,
        rates_g,
        rates_s,
        pop_grown,
        pop_shuf,
        fg,
        cg,
        fs,
        cs,
        og,
        os,
        scg,
        scs,
    }
}
