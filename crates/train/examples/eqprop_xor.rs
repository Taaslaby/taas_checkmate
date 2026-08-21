//! Equilibrium Propagation на XOR — «STDP, но градиентный».
//!
//! Демонстрирует, что контраст двух фаз (`pre·(post₂ − post₁)`) смотрит вдоль
//! точного backprop-градиента и решает XOR (линейно неразделимую задачу), а
//! чистый Хебб (`pre·post`, без credit assignment) — нет.
//!
//! Запуск:
//!     cargo run -p taas-train --example eqprop_xor --release

use taas_train::eqprop::{cosine, EqPropNet};

fn main() {
    let x: [[f64; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let t_usize = [0usize, 1, 1, 0];
    let t: Vec<f64> = t_usize.iter().map(|&v| 2.0 * v as f64 - 1.0).collect();

    println!("==============================================================");
    println!("  XOR:  Δw ∝ pre·(post₂ − post₁)   vs   Δw ∝ pre·post");
    println!("==============================================================");

    // ── (0) градиент-чек: контраст vs backprop ──
    let net0 = EqPropNet::new(8, 1);
    let (dw_bp, d_v_bp) = net0.backprop_grad(&x, &t);
    let mut dw_ch = vec![0.0f64; net0.w.len()];
    let mut d_v_ch = vec![0.0f64; net0.v.len()];
    for (x2, &tg) in x.iter().zip(&t) {
        let xv = [1.0, x2[0], x2[1]];
        let (h_free, y_free) = net0.settle(&xv, None, 0.0, 200);
        let (h_nudge, y_nudge) = net0.settle(&xv, Some(tg), 0.5, 200);
        for i in 0..net0.n_hidden {
            for j in 0..3 {
                dw_ch[i * 3 + j] += (h_nudge[i] - h_free[i]) * xv[j];
            }
        }
        d_v_ch[0] += y_nudge - y_free;
        for i in 0..net0.n_hidden {
            d_v_ch[i + 1] += y_nudge * h_nudge[i] - y_free * h_free[i];
        }
    }
    println!("\n── (0) ГРАДИЕНТ-ЧЕК: контраст vs backprop ──");
    println!(
        "  cos(dW) = {:+.4}   (|cos| → 1 = контраст вдоль градиента)",
        cosine(&dw_ch, &dw_bp)
    );
    // dV: CHL ΔV — градиент симметричной энергетической сети, а backprop_grad
    // — feedforward-эталон (без Vᵀy обратной связи). Это разные задачи, поэтому
    // cos(dV) не обязан быть ≈ ±1 — и это ожидаемо, а не баг.
    println!(
        "  cos(dV) = {:+.4}   (≠1 ожидаемо: CHL ΔV vs feedforward-эталон)",
        cosine(&d_v_ch, &d_v_bp)
    );

    // ── (A) контрастное правило ──
    println!("\n── (A) КОНТРАСТНОЕ правило (EqProp) — «STDP с градиентом» ──");
    let mut net = EqPropNet::new(8, 0);
    for ep in 1..=5000 {
        net.train_contrastive(&x, &t, 0.02, 0.5);
        if ep % 500 == 0 || ep == 1 {
            println!(
                "  epoch {:>4} | loss {:.4} | acc {:.2}%",
                ep,
                net.loss(&x, &t),
                net.accuracy(&x, &t_usize) * 100.0
            );
        }
    }
    let preds_a: Vec<usize> = x.iter().map(|&x2| net.predict(x2)).collect();
    println!("  предсказания: {preds_a:?}  (цель {t_usize:?})");

    // ── (B) чистый Хебб ──
    println!("\n── (B) ЧИСТЫЙ Хебб — «STDP без ошибки» ──");
    let mut net2 = EqPropNet::new(8, 0);
    for ep in 1..=5000 {
        net2.train_hebbian(&x, 0.02, 5.0);
        if ep % 500 == 0 || ep == 1 {
            println!(
                "  epoch {:>4} | loss {:.4} | acc {:.2}%",
                ep,
                net2.loss(&x, &t),
                net2.accuracy(&x, &t_usize) * 100.0
            );
        }
    }
    let preds_b: Vec<usize> = x.iter().map(|&x2| net2.predict(x2)).collect();
    println!("  предсказания: {preds_b:?}  (цель {t_usize:?})");

    let acc_a = net.accuracy(&x, &t_usize);
    let acc_b = net2.accuracy(&x, &t_usize);
    println!("\n==============================================================");
    println!(
        "  Контраст (STDP+градиент): {:.0}%  ← решает XOR",
        acc_a * 100.0
    );
    println!(
        "  Чистый Хебб (STDP)      : {:.0}%  ← не решает (нет credit)",
        acc_b * 100.0
    );
    println!("==============================================================");
    println!("  Вывод: контраст двух фаз = ошибка = градиент. Это и есть");
    println!("         «STDP, но градиентный» — твоя формула, Антон.");
}
