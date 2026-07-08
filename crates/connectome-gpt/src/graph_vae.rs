use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Configuration for the Graph VAE (inference only).
#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
    pub seed: u64,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self { input_dim: 10, hidden_dim: 256, latent_dim: 32, seed: 42 }
    }
}

/// Learned linear layer — forward only.
#[derive(Debug, Clone)]
struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    n_in: usize,
    n_out: usize,
}

impl Linear {
    fn new(n_in: usize, n_out: usize, rng: &mut StdRng) -> Self {
        let scale = (2.0 / (n_in + n_out) as f32).sqrt();
        let w: Vec<f32> = (0..n_in * n_out)
            .map(|_| rng.gen::<f32>() * 2.0 * scale - scale)
            .collect();
        Self { w, b: vec![0.0f32; n_out], n_in, n_out }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let batch = x.len() / self.n_in;
        let mut out = vec![0.0f32; batch * self.n_out];
        for b in 0..batch {
            let bx = b * self.n_in;
            let bo = b * self.n_out;
            for j in 0..self.n_out {
                let mut sum = self.b[j];
                for k in 0..self.n_in {
                    sum += x[bx + k] * self.w[k * self.n_out + j];
                }
                out[bo + j] = sum;
            }
        }
        out
    }

    fn set_weights(&mut self, w: &[f32], b: &[f32]) {
        self.w.copy_from_slice(w);
        self.b.copy_from_slice(b);
    }
}

fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// Trained VAE weights embedded at compile time.
pub const VAE_WEIGHTS_BIN: &[u8] = include_bytes!("../models/vae_weights.bin");

/// Graph VAE — 7-layer deep architecture trained in Python.
#[derive(Debug, Clone)]
pub struct GraphVAE {
    pub config: VaeConfig,
    encoder_0: Linear,      // input → hidden
    encoder_2: Linear,      // hidden → hidden
    encoder_mu: Linear,     // hidden → latent
    encoder_logvar: Linear, // hidden → latent
    decoder_0: Linear,      // latent → hidden
    decoder_2: Linear,      // hidden → hidden
    decoder_4: Linear,      // hidden → input
    rng: StdRng,
}

impl GraphVAE {
    pub fn new(config: VaeConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let h = config.hidden_dim;
        let d = config.input_dim;
        let z = config.latent_dim;
        Self {
            encoder_0: Linear::new(d, h, &mut rng),
            encoder_2: Linear::new(h, h, &mut rng),
            encoder_mu: Linear::new(h, z, &mut rng),
            encoder_logvar: Linear::new(h, z, &mut rng),
            decoder_0: Linear::new(z, h, &mut rng),
            decoder_2: Linear::new(h, h, &mut rng),
            decoder_4: Linear::new(h, d, &mut rng),
            rng, config,
        }
    }

    /// Create a VAE with weights embedded at compile time.
    /// No file I/O at runtime — weight blob is baked into the binary.
    pub fn from_embedded(config: VaeConfig) -> Self {
        let mut vae = Self::new(config);
        vae.load(VAE_WEIGHTS_BIN);
        vae
    }

    pub fn encode(&self, features: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let h = relu(&self.encoder_0.forward(features));
        let h = relu(&self.encoder_2.forward(&h));
        let mu = self.encoder_mu.forward(&h);
        let logvar = self.encoder_logvar.forward(&h);
        (mu, logvar)
    }

    pub fn decode(&self, z: &[f32]) -> Vec<f32> {
        let h = relu(&self.decoder_0.forward(z));
        let h = relu(&self.decoder_2.forward(&h));
        self.decoder_4.forward(&h)
    }

    pub fn forward_deterministic(&self, features: &[f32]) -> Vec<f32> {
        let (mu, _) = self.encode(features);
        self.decode(&mu)
    }

    pub fn generate(&mut self) -> Vec<f32> {
        let z: Vec<f32> = (0..self.config.latent_dim)
            .map(|_| self.rng.gen::<f32>() * 4.0 - 2.0)
            .collect();
        self.decode(&z)
    }

    pub fn generate_from_z(&self, z: &[f32]) -> Vec<f32> {
        self.decode(z)
    }

    pub fn interpolate(&self, f1: &[f32], f2: &[f32], t: f32) -> Vec<f32> {
        let (mu1, _) = self.encode(f1);
        let (mu2, _) = self.encode(f2);
        let mu: Vec<f32> = mu1.iter().zip(mu2.iter()).map(|(&a, &b)| a * (1.0 - t) + b * t).collect();
        self.decode(&mu)
    }

    pub fn load_weights(&mut self, weights: &[Vec<f32>]) {
        let mut layers = [
            &mut self.encoder_0, &mut self.encoder_2,
            &mut self.encoder_mu, &mut self.encoder_logvar,
            &mut self.decoder_0, &mut self.decoder_2, &mut self.decoder_4,
        ];
        for (i, layer) in layers.iter_mut().enumerate() {
            if i * 2 + 1 < weights.len() {
                layer.set_weights(&weights[i * 2], &weights[i * 2 + 1]);
            }
        }
    }

    pub fn save(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let layers = [
            &self.encoder_0, &self.encoder_2,
            &self.encoder_mu, &self.encoder_logvar,
            &self.decoder_0, &self.decoder_2, &self.decoder_4,
        ];
        for l in &layers {
            buf.extend_from_slice(&(l.w.len() as u32).to_le_bytes());
            for &x in &l.w { buf.extend_from_slice(&x.to_le_bytes()); }
            buf.extend_from_slice(&(l.b.len() as u32).to_le_bytes());
            for &x in &l.b { buf.extend_from_slice(&x.to_le_bytes()); }
        }
        buf
    }

    pub fn load(&mut self, buf: &[u8]) {
        fn next(ptr: &mut &[u8]) -> Vec<f32> {
            let len = u32::from_le_bytes([ptr[0], ptr[1], ptr[2], ptr[3]]) as usize;
            *ptr = &ptr[4..];
            let v: Vec<f32> = ptr[..len * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            *ptr = &ptr[len * 4..];
            v
        }
        let mut p = buf;
        self.encoder_0.w = next(&mut p); self.encoder_0.b = next(&mut p);
        self.encoder_2.w = next(&mut p); self.encoder_2.b = next(&mut p);
        self.encoder_mu.w = next(&mut p); self.encoder_mu.b = next(&mut p);
        self.encoder_logvar.w = next(&mut p); self.encoder_logvar.b = next(&mut p);
        self.decoder_0.w = next(&mut p); self.decoder_0.b = next(&mut p);
        self.decoder_2.w = next(&mut p); self.decoder_2.b = next(&mut p);
        self.decoder_4.w = next(&mut p); self.decoder_4.b = next(&mut p);
    }
}
