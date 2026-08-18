//! Парсер FAFB npz + npy + csv.gz (FlyWire).

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use super::{intern, klass_code, norm_positions};
use crate::io::data::MancData;

/// Парсит npz (zip из .npy): CSR `indptr/indices/data/neuron_ids/n_neurons`,
/// позиции — второй файл `<имя>_positions.npy` рядом, аннотации —
/// `consolidated_cell_types.csv.gz` + `classification.csv.gz` (flywire).
pub fn from_fafb_npz(path: &Path) -> Result<MancData, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("не читается {path:?}: {e}"))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| format!("npz не zip: {e}"))?;

    let mut read_npy = |name: &str| -> Result<(String, Vec<usize>, Vec<u8>), String> {
        let mut f = zip.by_name(name).map_err(|_| format!("нет {name} в npz"))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        parse_npy(&buf)
    };

    let (d1, _, ib) = read_npy("indptr.npy")?;
    let (d2, _, cb) = read_npy("indices.npy")?;
    let (d3, _, vb) = read_npy("data.npy")?;
    let (d4, _, nb) = read_npy("neuron_ids.npy")?;
    let (d5, n_shape, n_b) = read_npy("n_neurons.npy")?;

    // dtype-устойчивое чтение: в разных npz могут быть i4/i8/f4/f8
    let indptr = npy_f64(&ib, &d1)?;
    let indices = npy_f64(&cb, &d2)?;
    let values = npy_f64(&vb, &d3)?;
    let nids = npy_f64(&nb, &d4)?;
    let nn = npy_f64(&n_b, &d5)?;
    // n_neurons: 0-d (скаляр) или 1-d; страхуемся длиной neuron_ids
    let n = if n_shape.is_empty() {
        nn.first().map(|&v| v as usize).unwrap_or(nids.len())
    } else {
        nids.len()
    };
    if nids.len() != n || indptr.len() != n + 1 {
        return Err(format!(
            "npz размеры не сходятся: n_neurons={n} ids={} indptr={}",
            nids.len(),
            indptr.len()
        ));
    }
    let s = indices.len();
    if values.len() != s {
        return Err("npz: indices и data разной длины".to_string());
    }

    // позиции вторым файлом: <имя>_positions.npy рядом с npz
    let pos_path = sibling_positions(path)?;
    let (pos_desc, pos_shape, pos_bytes) = if let Some(pp) = &pos_path {
        match std::fs::read(pp) {
            Ok(b) => parse_npy(&b)?,
            Err(_) => return Err(format!("не читаются позиции {pp:?}")),
        }
    } else {
        return Err("рядом с npz нет *_positions.npy".to_string());
    };
    if pos_shape.len() != 2 || pos_shape[0] != n || pos_shape[1] != 3 {
        return Err(format!("позиции: shape {pos_shape:?} не (n,3)"));
    }
    let pos_raw = npy_f64(&pos_bytes, &pos_desc)?;
    let mut pos = vec![[0.0f64; 3]; n];
    for i in 0..n {
        pos[i] = [pos_raw[i * 3], pos_raw[i * 3 + 1], pos_raw[i * 3 + 2]];
    }
    let valid: Vec<bool> = pos
        .iter()
        .map(|p| p.iter().all(|v| v.is_finite()))
        .collect();
    let (xs, ys, zs, has_pos) = norm_positions(&pos, &valid);

    // аннотации flywire (по root-id, младшие 9 цифр)
    let dir = path.parent().unwrap_or(Path::new("."));
    let type_map = load_csv_gz(
        &dir.join("consolidated_cell_types.csv.gz"),
        &["primary_type"],
    )?;
    let class_map = load_csv_gz(
        &dir.join("classification.csv.gz"),
        &["flow", "class", "side"],
    )?;

    let nt = vec![0u8; n];
    let mut kl = vec![0u8; n];
    let mut sd = vec![0u8; n];
    let mut type_names: Vec<String> = Vec::with_capacity(n);
    for (i, &rid) in nids.iter().enumerate() {
        let key = rid as u64 % 1_000_000_000;
        let t = type_map
            .get(&key)
            .map(|v| v[0].clone())
            .filter(|s| !s.is_empty())
            .or_else(|| {
                class_map
                    .get(&key)
                    .map(|v| v[1].clone())
                    .filter(|s| !s.is_empty())
            })
            .unwrap_or_else(|| "flywire".to_string());
        type_names.push(t);
        if let Some(c) = class_map.get(&key) {
            kl[i] = klass_code(&format!("{} {}", c[0], c[1]));
            let side = c[2].trim().to_lowercase();
            if side.starts_with('l') {
                sd[i] = 1;
            } else if side.starts_with('r') {
                sd[i] = 2;
            }
        }
    }
    let nt_types: Vec<f32> = vec![1.0; n];
    let body_id: Vec<u64> = nids.iter().map(|&v| v as u64).collect();
    let (type_idx, types) = intern(&type_names)?;

    let row_ptr: Vec<i32> = indptr.iter().map(|&v| v as i64 as i32).collect();
    let col: Vec<i32> = indices.iter().map(|&v| v as i64 as i32).collect();
    let val: Vec<f32> = values.iter().map(|&v| v as f32).collect();

    Ok(MancData {
        n,
        s,
        xs,
        ys,
        zs,
        has_pos,
        nt,
        klass: kl,
        leg: vec![0u8; n],
        side: sd,
        nt_types,
        body_id,
        type_idx,
        subclass_idx: vec![0u16; n],
        target_idx: vec![0u16; n],
        types,
        subclasses: vec!["None".into()],
        targets: vec!["None".into()],
        row_ptr,
        col,
        val,
        rev_row_ptr: None,
        rev_col: None,
        rev_val: None,
    })
}

/// Имя файла позиций рядом с npz: `fafb_brain_connectome.npz` →
/// `fafb_brain_positions.npy`.
fn sibling_positions(npz: &Path) -> Result<Option<PathBuf>, String> {
    let name = npz.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let mut cands = vec![PathBuf::from(format!(
        "{}_positions.npy",
        name.replace("_connectome", "_positions")
    ))];
    cands.push(PathBuf::from(format!(
        "{}_positions.npy",
        name.replace("_connectome", "")
    )));
    cands.push(PathBuf::from(format!("{name}_positions.npy")));
    let dir = npz.parent().unwrap_or(Path::new("."));
    for c in &cands {
        let p = dir.join(c);
        if p.exists() {
            return Ok(Some(p));
        }
    }
    Ok(None)
}

/// csv.gz → HashMap<root_id(младшие 9 цифр), Vec<значений колонок>>.
fn load_csv_gz(path: &Path, cols: &[&str]) -> Result<HashMap<u64, Vec<String>>, String> {
    let mut out: HashMap<u64, Vec<String>> = HashMap::new();
    if !path.exists() {
        return Ok(out);
    }
    let f = std::fs::File::open(path).map_err(|e| format!("{path:?}: {e}"))?;
    let gz = flate2::read::GzDecoder::new(f);
    let mut rdr = csv::Reader::from_reader(gz);
    let header: Vec<String> = rdr
        .headers()
        .map_err(|e| e.to_string())?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let ki = header.iter().position(|h| h == "root_id");
    let mut vis: Vec<usize> = Vec::new();
    for c in cols {
        if let Some(i) = header.iter().position(|h| h == *c) {
            vis.push(i);
        }
    }
    let (Some(ki), false) = (ki, vis.is_empty()) else {
        return Ok(out);
    };
    for rec in rdr.records().flatten() {
        let Some(rid) = rec.get(ki).and_then(|v| v.trim().parse::<u64>().ok()) else {
            continue;
        };
        let vals: Vec<String> = vis
            .iter()
            .map(|&i| rec.get(i).unwrap_or("").to_string())
            .collect();
        out.insert(rid % 1_000_000_000, vals);
    }
    Ok(out)
}

/// Парсит .npy: возвращает (descr, shape, сырые байты данных).
fn parse_npy(buf: &[u8]) -> Result<(String, Vec<usize>, Vec<u8>), String> {
    if buf.len() < 10 || &buf[0..6] != b"\x93NUMPY" {
        return Err("не npy (нет magic)".to_string());
    }
    let major = buf[6];
    let (hdr_len, data_off) = match major {
        1 => (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10),
        2 => (
            u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize,
            12,
        ),
        v => return Err(format!("npy версия {v} не поддерживается")),
    };
    let hdr = std::str::from_utf8(&buf[data_off..data_off + hdr_len])
        .map_err(|_| "npy header не utf8".to_string())?;
    // {'descr': '<i8', 'fortran_order': False, 'shape': (139190,), }
    let descr = between(hdr, "'descr':", ",")
        .map(|s| s.trim().trim_matches('\'').to_string())
        .unwrap_or_default();
    let shape = between(hdr, "'shape': (", ")")
        .map(|s| {
            s.split(',')
                .filter_map(|p| p.trim().parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_default();
    Ok((descr, shape, buf[data_off + hdr_len..].to_vec()))
}

fn between<'a>(hay: &'a str, from: &str, to: &str) -> Option<&'a str> {
    let a = hay.find(from)? + from.len();
    let rest = &hay[a..];
    let b = rest.find(to)?;
    Some(&rest[..b])
}

/// Читает данные .npy как f64 независимо от dtype в descr
/// (`<i4`, `|u8`, `<f4`, `<f8`, …) — единый интерфейс для всех массивов npz.
fn npy_f64(buf: &[u8], descr: &str) -> Result<Vec<f64>, String> {
    let d = descr.trim().trim_start_matches(['<', '>', '=', '|']);
    match d {
        "i1" | "u1" => Ok(buf.iter().map(|&v| v as f64).collect()),
        "i2" => Ok(buf
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect()),
        "u2" => Ok(buf
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect()),
        "i4" => Ok(as_i32(buf)?.iter().map(|&v| v as f64).collect()),
        "u4" => Ok(buf
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect()),
        "i8" => Ok(as_i64(buf)?.iter().map(|&v| v as f64).collect()),
        "u8" => Ok(buf
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect()),
        "f2" => Ok(buf
            .chunks_exact(2)
            .map(|c| f32::from_le_bytes([c[0], c[1], 0, 0]) as f64)
            .collect()),
        "f4" => Ok(as_f32(buf)?.iter().map(|&v| v as f64).collect()),
        "f8" => as_f64(buf),
        other => Err(format!("неизвестный dtype '{other}' в npy")),
    }
}

fn as_i64(b: &[u8]) -> Result<Vec<i64>, String> {
    if !b.len().is_multiple_of(8) {
        return Err("i64: размер не кратен 8".to_string());
    }
    Ok(b.chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn as_i32(b: &[u8]) -> Result<Vec<i32>, String> {
    if !b.len().is_multiple_of(4) {
        return Err("i32: размер не кратен 4".to_string());
    }
    Ok(b.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn as_f32(b: &[u8]) -> Result<Vec<f32>, String> {
    if !b.len().is_multiple_of(4) {
        return Err("f32: размер не кратен 4".to_string());
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn as_f64(b: &[u8]) -> Result<Vec<f64>, String> {
    if !b.len().is_multiple_of(8) {
        return Err("f64: размер не кратен 8".to_string());
    }
    Ok(b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

// ── Тесты ─────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use std::io::Write as _;
    use std::path::PathBuf;

    use super::{from_fafb_npz, npy_f64, parse_npy};

    /// Собирает .npy v1 из descr/shape/данных — с корректным выравниванием
    /// заголовка (10 байт префикса + словарь + '\n' кратно 64), как numpy.
    fn npy_v1(descr: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let mut dict =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape_str},), }}");
        // выравнивание: 10 (magic+ver+len) + len(dict) + '\n' кратно 64
        let rem = (10 + dict.len() + 1) % 64;
        if rem != 0 {
            dict.push_str(&" ".repeat(64 - rem));
        }
        let mut out = Vec::with_capacity(10 + dict.len() + 1 + data.len());
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1); // major
        out.push(0); // minor
                     // длина заголовка ВКЛЮЧАЕТ завершающий '\n' (формат numpy)
        out.extend_from_slice(&((dict.len() + 1) as u16).to_le_bytes());
        out.extend_from_slice(dict.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(data);
        out
    }

    /// zip-архив в памяти (npz = zip из .npy).
    fn zip_bytes(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut zw = zip::ZipWriter::new(std::io::Cursor::new(&mut out));
            for (name, data) in entries {
                zw.start_file(*name, zip::write::SimpleFileOptions::default())
                    .unwrap();
                zw.write_all(data).unwrap();
            }
            zw.finish().unwrap();
        }
        out
    }

    fn gzip(bytes: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(bytes).unwrap();
        enc.finish().unwrap()
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("connscope_npz_test_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &std::path::Path, bytes: &[u8]) {
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn test_parse_npy_v1_and_dtypes() {
        // i8 массив [0, 2, 3, 5]
        let mut i8 = Vec::new();
        for v in [0i64, 2, 3, 5] {
            i8.extend_from_slice(&v.to_le_bytes());
        }
        let b = npy_v1("<i8", &[4], &i8);
        let (descr, shape, data) = parse_npy(&b).unwrap();
        assert_eq!(descr, "<i8");
        assert_eq!(shape, vec![4]);
        assert_eq!(npy_f64(&data, &descr).unwrap(), vec![0.0, 2.0, 3.0, 5.0]);

        // f4 массив (проверка чтения float32 → f64)
        let mut f4 = Vec::new();
        for v in [1.0f32, 0.5] {
            f4.extend_from_slice(&v.to_le_bytes());
        }
        let b = npy_v1("<f4", &[2], &f4);
        let (descr, shape, data) = parse_npy(&b).unwrap();
        assert_eq!(shape, vec![2]);
        let vals = npy_f64(&data, &descr).unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-6 && (vals[1] - 0.5).abs() < 1e-6);

        // скаляр (0-d): shape = ()
        let mut one = Vec::new();
        one.extend_from_slice(&3i64.to_le_bytes());
        let b = npy_v1("<i8", &[], &one);
        let (_, shape, data) = parse_npy(&b).unwrap();
        assert!(shape.is_empty());
        assert_eq!(npy_f64(&data, "<i8").unwrap(), vec![3.0]);

        // 2-d: shape (2,3)
        let mut f8 = Vec::new();
        for _ in 0..24 {
            f8.extend_from_slice(&0.0f64.to_le_bytes());
        }
        let b = npy_v1("<f8", &[2, 3], &f8);
        let (_, shape, _) = parse_npy(&b).unwrap();
        assert_eq!(shape, vec![2, 3]);
    }

    #[test]
    fn test_parse_npy_rejects_non_npy() {
        assert!(parse_npy(b"not a npy file!").is_err());
    }

    #[test]
    fn test_from_fafb_npz_roundtrip() {
        // 3 нейрона, 5 синапсов: 0→1, 0→2, 1→2, 2→0, 2→1
        let n = 3usize;
        let s = 5usize;
        let indptr: Vec<i64> = vec![0, 2, 3, 5];
        let indices: Vec<i64> = vec![1, 2, 2, 0, 1];
        let values: Vec<f64> = vec![1.0, 0.5, 2.0, 3.0, 0.25];
        let nids: Vec<i64> = vec![1, 2, 3]; // младшие 9 цифр root-id
        let nn: Vec<i64> = vec![3];
        // позиции: x 0..10, y 0..10, z = 0
        let pos: Vec<f64> = vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0];

        let i8 = |v: &[i64]| {
            let mut b = Vec::new();
            for x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            b
        };
        let f8 = |v: &[f64]| {
            let mut b = Vec::new();
            for x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            b
        };

        let npz = zip_bytes(&[
            ("indptr.npy", npy_v1("<i8", &[n + 1], &i8(&indptr))),
            ("indices.npy", npy_v1("<i8", &[s], &i8(&indices))),
            ("data.npy", npy_v1("<f8", &[s], &f8(&values))),
            ("neuron_ids.npy", npy_v1("<i8", &[n], &i8(&nids))),
            ("n_neurons.npy", npy_v1("<i8", &[], &i8(&nn))),
        ]);

        let dir = temp_dir("roundtrip");
        let npz_path = dir.join("fly_brain.npz");
        write(&npz_path, &npz);
        write(
            &dir.join("fly_brain_positions.npy"),
            &npy_v1("<f8", &[n, 3], &f8(&pos)),
        );
        // аннотации flywire (по root-id)
        write(
            &dir.join("consolidated_cell_types.csv.gz"),
            &gzip(b"root_id,primary_type\n1,motor_neuron\n2,sensory_neuron\n3,interneuron\n"),
        );
        write(
            &dir.join("classification.csv.gz"),
            &gzip(
                b"root_id,flow,class,side\n1,efferent,motor,right\n2,afferent,sensory,left\n3,intrinsic,interneuron,right\n",
            ),
        );

        let d = from_fafb_npz(&npz_path).expect("npz должен распарситься");

        assert_eq!(d.n, 3);
        assert_eq!(d.s, 5);
        assert_eq!(d.row_ptr, vec![0, 2, 3, 5]);
        assert_eq!(d.col, vec![1, 2, 2, 0, 1]);
        assert_eq!(d.val, vec![1.0, 0.5, 2.0, 3.0, 0.25]);
        assert_eq!(d.body_id, vec![1, 2, 3]);
        // позиции нормированы на [0,1]: x и y растянуты 0..10
        assert_eq!(d.has_pos, vec![1, 1, 1]);
        assert!((d.xs[0] - 0.0).abs() < 1e-6 && (d.xs[1] - 1.0).abs() < 1e-6);
        assert!((d.ys[2] - 1.0).abs() < 1e-6);
        assert!(d.zs.iter().all(|&z| z.abs() < 1e-6));
        // аннотации: типы из consolidated_cell_types, класс/сторона из classification
        assert_eq!(
            d.types,
            vec!["motor_neuron", "sensory_neuron", "interneuron"]
        );
        assert_eq!(d.type_idx, vec![0, 1, 2]);
        assert_eq!(d.klass, vec![1, 2, 5]); // motor / sensory / interneuron
        assert_eq!(d.side, vec![2, 1, 2]); // right / left / right

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_from_fafb_npz_missing_array_errors() {
        let mut i8 = Vec::new();
        for x in [0i64, 1, 2] {
            i8.extend_from_slice(&x.to_le_bytes());
        }
        let npz = zip_bytes(&[("indptr.npy", npy_v1("<i8", &[3], &i8))]);
        let dir = temp_dir("missing");
        let npz_path = dir.join("bad.npz");
        write(&npz_path, &npz);
        let err = match from_fafb_npz(&npz_path) {
            Err(e) => e,
            Ok(_) => panic!("ожидалась ошибка о пропущенном массиве"),
        };
        assert!(err.contains("indices.npy"), "ошибка: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_from_fafb_npz_size_mismatch_errors() {
        // indptr (4) не сходится с n_neurons=2 (ожидается 3)
        let i8 = |v: &[i64]| {
            let mut b = Vec::new();
            for x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            b
        };
        let f8 = |v: &[f64]| {
            let mut b = Vec::new();
            for x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            b
        };
        let npz = zip_bytes(&[
            ("indptr.npy", npy_v1("<i8", &[4], &i8(&[0, 1, 2, 3]))),
            ("indices.npy", npy_v1("<i8", &[3], &i8(&[0, 1, 2]))),
            ("data.npy", npy_v1("<f8", &[3], &f8(&[0.0, 1.0, 2.0]))),
            ("neuron_ids.npy", npy_v1("<i8", &[2], &i8(&[1, 2]))),
            ("n_neurons.npy", npy_v1("<i8", &[], &i8(&[2]))),
        ]);
        let dir = temp_dir("mismatch");
        let npz_path = dir.join("bad.npz");
        write(&npz_path, &npz);
        let err = match from_fafb_npz(&npz_path) {
            Err(e) => e,
            Ok(_) => panic!("ожидалась ошибка о несовпадении размеров"),
        };
        assert!(err.contains("не сходятся"), "ошибка: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
