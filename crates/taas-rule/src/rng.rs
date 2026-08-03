//! Миниатюрный PRNG (xorshift64) — без внешних зависимостей.

/// Детерминированный генератор случайных чисел (xorshift64).
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// Создать генератор с посевом.
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }

    /// Следующее 64-битное значение.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Случайное число из [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new(0x9E3779B97F4A7C15)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_in_unit_range() {
        let mut r = Rng::new(1);
        for _ in 0..1000 {
            let x = r.next_f32();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn test_rng_deterministic() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
