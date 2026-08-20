/// Evaluate a NeuroML `inhomogeneousValue` density expression at path length
/// `p` (µm from the soma) and return the density in **S/cm²**.
///
/// The expression is a LEMS expression in the SI base unit of conductance
/// density, **S/m²** (`1 S/cm² = 1e4 S/m²`), so the raw result is divided by
/// `1e4`. Real Poirazi 2003 exports write e.g.
/// `1e4 * H(p_apical_trunk-300) * 7e-3` for a `7e-3 S/cm²` density. Supports
/// numbers, the path-length symbol (any bare identifier, e.g. `p_apical_trunk`),
/// `H(...)` (Heaviside step: 1 when the argument is ≥ 0, else 0 — jNeuroML
/// convention), `exp(...)` (natural exponential, for sigmoid densities), and
/// `+ - * /` with parentheses.
pub(crate) fn eval_inhomogeneous_expr(expr: &str, p: f32) -> f32 {
    let toks = inho_tokens(expr);
    let mut pos = 0usize;
    let val = inho_expr(&toks, &mut pos, p);
    if pos != toks.len() {
        return 0.0;
    }
    val * 1e-4 // S/m² → S/cm²
}

#[derive(Debug, Clone, PartialEq)]
enum InhoTok {
    Num(f32),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Comma,
    Ident(String),
}

fn inho_tokens(s: &str) -> Vec<InhoTok> {
    let b = s.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || c == '.' {
            let start = i;
            while i < b.len() && (b[i] as char).is_ascii_digit()
                || (i < b.len() && b[i] as char == '.')
            {
                if (b[i] as char).is_ascii_digit() || b[i] as char == '.' {
                    i += 1;
                } else {
                    break;
                }
            }
            if i < b.len() && (b[i] as char == 'e' || b[i] as char == 'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] as char == '+' || b[j] as char == '-') {
                    j += 1;
                }
                if j < b.len() && (b[j] as char).is_ascii_digit() {
                    i = j + 1;
                    while i < b.len() && (b[i] as char).is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            toks.push(InhoTok::Num(s[start..i].parse().unwrap_or(0.0)));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] as char == '_') {
                i += 1;
            }
            toks.push(InhoTok::Ident(s[start..i].to_string()));
        } else {
            match c {
                '+' => toks.push(InhoTok::Plus),
                '-' => toks.push(InhoTok::Minus),
                '*' => toks.push(InhoTok::Star),
                '/' => toks.push(InhoTok::Slash),
                '(' => toks.push(InhoTok::LParen),
                ')' => toks.push(InhoTok::RParen),
                ',' => toks.push(InhoTok::Comma),
                _ => {}
            }
            i += 1;
        }
    }
    toks
}

fn inho_expr(toks: &[InhoTok], pos: &mut usize, p: f32) -> f32 {
    let mut lhs = inho_term(toks, pos, p);
    loop {
        match toks.get(*pos) {
            Some(InhoTok::Plus) => {
                *pos += 1;
                lhs += inho_term(toks, pos, p);
            }
            Some(InhoTok::Minus) => {
                *pos += 1;
                lhs -= inho_term(toks, pos, p);
            }
            _ => return lhs,
        }
    }
}

fn inho_term(toks: &[InhoTok], pos: &mut usize, p: f32) -> f32 {
    let mut lhs = inho_factor(toks, pos, p);
    loop {
        match toks.get(*pos) {
            Some(InhoTok::Star) => {
                *pos += 1;
                lhs *= inho_factor(toks, pos, p);
            }
            Some(InhoTok::Slash) => {
                *pos += 1;
                let d = inho_factor(toks, pos, p);
                lhs /= if d != 0.0 { d } else { 1.0 };
            }
            _ => return lhs,
        }
    }
}

fn inho_factor(toks: &[InhoTok], pos: &mut usize, p: f32) -> f32 {
    match toks.get(*pos) {
        Some(InhoTok::Num(n)) => {
            *pos += 1;
            *n
        }
        Some(InhoTok::Minus) => {
            *pos += 1;
            -inho_factor(toks, pos, p)
        }
        Some(InhoTok::LParen) => {
            *pos += 1;
            let v = inho_expr(toks, pos, p);
            if toks.get(*pos) == Some(&InhoTok::RParen) {
                *pos += 1;
            }
            v
        }
        // Identifiers: `H(...)` — Heaviside; `exp(...)` — natural exponential;
        // a bare path-length symbol — p.
        Some(InhoTok::Ident(name)) => {
            *pos += 1;
            if toks.get(*pos) == Some(&InhoTok::LParen) {
                *pos += 1;
                let arg = inho_expr(toks, pos, p);
                if toks.get(*pos) == Some(&InhoTok::RParen) {
                    *pos += 1;
                }
                match name.as_str() {
                    // H(x): 1 when x ≥ 0, else 0 (jNeuroML step convention).
                    "H" => {
                        if arg >= 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    // exp(x): natural exponential (sigmoid densities, e.g. hdend).
                    "exp" => arg.exp(),
                    // Unknown function: fall back to the path length (lenient).
                    _ => p,
                }
            } else {
                p
            }
        }
        _ => 0.0,
    }
}
