//! Бенчмарк скорости обучения Actor-Critic агента на простом MDP.
//!
//! Запуск: cargo run --example learning_bench --no-default-features
//!
//! MDP: 2 состояния, 2 действия.
//!   state 0, action 0 → reward +1, next state 1
//!   state 0, action 1 → reward -0.5, next state 1
//!   state 1, action 1 → reward +1, next state 0
//!   state 1, action 0 → reward -0.5, next state 0
//! Оптимум: всегда выбирать действие = состояние → reward +1 каждый шаг.

use taas_checkmate::rl::{ActorCriticAgent, test_mdp_episode};
use std::time::Instant;

fn main() {
    println!("=== Бенчмарк обучения ActorCriticAgent ===");
    println!();
    println!("MDP: 2 состояния, 2 действия");
    println!("Оптимальная стратегия: action = state → +1/шаг");
    println!("Субоптимальная: action ≠ state → -0.5/шаг");
    println!();

    // Параметры
    let configs = vec![
        ("default (lr=0.01, tau_e=500)", 0.01, 500.0),
        ("fast (lr=0.03, tau_e=300)",    0.03, 300.0),
        ("aggressive (lr=0.05, tau_e=200)", 0.05, 200.0),
    ];

    for (name, lr, tau_e) in &configs {
        println!("─── {name} ───");

        for &n_hidden in &[4, 8, 16] {
            let mut total_time = 0.0;
            let mut final_rewards = Vec::new();

            for trial in 0..3 {
                let mut agent = ActorCriticAgent::new(2, n_hidden, 2, 30, 1.0, trial + 42);
                agent.lr_actor = *lr;
                agent.lr_critic = *lr;
                agent.gamma = 0.9;
                agent.tau_e = *tau_e;

                let start = Instant::now();
                let mut rewards = Vec::new();
                let mut best_avg = f32::NEG_INFINITY;

                for ep in 0..30 {
                    agent.reset_episode();
                    let total = test_mdp_episode(&mut agent, 30);

                    // Скользящее среднее за 5 эпизодов
                    rewards.push(total);
                    if rewards.len() > 5 {
                        rewards.remove(0);
                    }
                    let avg: f32 = rewards.iter().sum::<f32>() / rewards.len() as f32;
                    if avg > best_avg { best_avg = avg; }

                    // Критерий сходимости: средняя награда > 25 за 5 эпизодов
                    if avg > 25.0 {
                        let conv_ep = ep + 1;
                        println!("  n_hidden={n_hidden:2} | сходимость за {conv_ep:2} эпизодов (avg={avg:.1})");
                        break;
                    }
                }
                let elapsed = start.elapsed();
                total_time += elapsed.as_secs_f64();
                final_rewards.push(best_avg);
            }

            let avg_reward: f32 = final_rewards.iter().sum::<f32>() / final_rewards.len() as f32;
            println!("  n_hidden={n_hidden:2} | среднее время: {:.3}s | лучшая награда: {:.1}", total_time / 3.0, avg_reward);
        }
        println!();
    }

    // Замер скорости inference на разных размерностях
    for &(ni, nh, na, ns, label) in &[
        (2, 8, 2, 30, "малая (ni=2, nh=8, na=2, ns=30)"),
        (4, 64, 4, 50, "большая (ni=4, nh=64, na=4, ns=50)"),
        (16, 64, 16, 50, "крупная (ni=16, nh=64, na=16, ns=50)"),
    ] {
        println!("─── Inference {label} ───");
        let mut agent = ActorCriticAgent::new(ni, nh, na, ns, 1.0, 42);
        let state = vec![0.5; ni];

        let start = Instant::now();
        let n_iter = 2000.max(20000 / ns / (nh / 8));
        for _ in 0..n_iter {
            agent.act(&state);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_secs_f64() / n_iter as f64;
        println!("  act() в среднем: {:.1}µs ({} вызовов)", per_call * 1_000_000.0, n_iter);
        println!("  что эквивалентно {:.0} шагов/сек", 1.0 / per_call);
        println!();
    }

    println!("─── Замер скорости обучения (шаг learn) ───");
    let mut agent = ActorCriticAgent::new(2, 8, 2, 30, 1.0, 42);
    let state = vec![0.5, -0.3];
    let (_, value) = agent.act(&state);
    let start = Instant::now();
    let n_iter = 100;
    for _ in 0..n_iter {
        agent.learn(1.0, value, true);
    }
    let elapsed = start.elapsed();
    let per_learn = elapsed.as_secs_f64() / n_iter as f64;
    println!("  learn() в среднем: {:.1}µs ({} вызовов)", per_learn * 1_000_000.0, n_iter);
}
