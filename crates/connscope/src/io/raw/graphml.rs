//! Парсер GraphML (`.graphml`) — XML от networkx/igraph.

use std::collections::HashMap;
use std::path::Path;

use super::{centroid, fibonacci_sphere, intern, klass_code, norm_positions, nt_code};
use crate::io::data::MancData;

/// Парсит GraphML в `MancData`. Тип = атрибут `name`/`type1`/id узла;
/// вес ребра = `weight`/`chemical_weight`/`electrical_weight`/1.0;
/// координаты — атрибуты x/y/z, иначе синтетическая сфера.
pub fn from_graphml(path: &Path) -> Result<MancData, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("не читается {path:?}: {e}"))?;
    from_graphml_str(&text).map_err(|e| format!("не GraphML {path:?}: {e}"))
}

/// Ядро парсера: GraphML как строка (без файловой системы) — удобно для тестов.
pub fn from_graphml_str(text: &str) -> Result<MancData, String> {
    let doc = roxmltree::Document::parse(text).map_err(|e| format!("не XML: {e}"))?;
    let graph = doc
        .descendants()
        .find(|n| n.has_tag_name("graph"))
        .ok_or_else(|| "нет <graph> в файле".to_string())?;

    // <key id for attr.name attr.type> → attr.name (для узлов и рёбер)
    let mut keys: HashMap<String, String> = HashMap::new();
    for k in doc.descendants().filter(|n| n.has_tag_name("key")) {
        if let (Some(id), Some(name)) = (k.attribute("id"), k.attribute("attr.name")) {
            keys.insert(id.to_string(), name.to_string());
        }
    }
    // attr.name → (key-id, data-value) для узла/ребра
    let resolve = |data: &HashMap<String, String>, wanted: &[&str]| -> Option<String> {
        for w in wanted {
            if let Some((kid, _)) = keys.iter().find(|(_, name)| name.as_str() == *w) {
                if let Some(v) = data.get(kid) {
                    if !v.is_empty() {
                        return Some(v.clone());
                    }
                }
            }
        }
        None
    };
    let resolve_f = |data: &HashMap<String, String>, wanted: &str| -> Option<f64> {
        resolve(data, &[wanted])?.trim().parse::<f64>().ok()
    };

    // узлы в порядке документа
    let mut order: HashMap<String, usize> = HashMap::new();
    let mut names: Vec<String> = Vec::new();
    let mut positions: Vec<Option<[f64; 3]>> = Vec::new();
    let mut extra: Vec<HashMap<String, String>> = Vec::new();
    for (i, n) in graph
        .children()
        .filter(|c| c.has_tag_name("node"))
        .enumerate()
    {
        let id = n.attribute("id").unwrap_or_default().to_string();
        order.insert(id.clone(), i);
        let data: HashMap<String, String> = n
            .children()
            .filter(|c| c.has_tag_name("data"))
            .map(|c| {
                (
                    c.attribute("key").unwrap_or_default().to_string(),
                    c.text().unwrap_or_default().to_string(),
                )
            })
            .collect();
        let name = resolve(&data, &["name", "type1"]).unwrap_or(id);
        names.push(name);
        positions.push(
            match (
                resolve_f(&data, "x"),
                resolve_f(&data, "y"),
                resolve_f(&data, "z"),
            ) {
                (Some(x), Some(y), Some(z)) => Some([x, y, z]),
                _ => None,
            },
        );
        extra.push(data);
    }
    let n = names.len();
    if n == 0 {
        return Err("в GraphML нет узлов".to_string());
    }

    // рёбра: вес из атрибута
    let weight_col = ["weight", "chemical_weight", "electrical_weight"]
        .iter()
        .find(|w| keys.values().any(|name| name.as_str() == **w))
        .copied();
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    for e in graph.children().filter(|c| c.has_tag_name("edge")) {
        let (Some(src), Some(dst)) = (e.attribute("source"), e.attribute("target")) else {
            continue;
        };
        let (Some(&si), Some(&di)) = (order.get(src), order.get(dst)) else {
            continue;
        };
        let data: HashMap<String, String> = e
            .children()
            .filter(|c| c.has_tag_name("data"))
            .map(|c| {
                (
                    c.attribute("key").unwrap_or_default().to_string(),
                    c.text().unwrap_or_default().to_string(),
                )
            })
            .collect();
        let w = match weight_col {
            Some(col) => resolve_f(&data, col).unwrap_or(1.0) as f32,
            None => 1.0,
        };
        edges.push((si, di, w));
    }
    edges.sort_by_key(|e| e.0);

    // позиции: реальные где есть, синтетика для остальных (или всех)
    let mut pos: Vec<[f64; 3]> = vec![[0.0; 3]; n];
    let mut valid = vec![false; n];
    let mut any_real = false;
    for (i, p) in positions.iter().enumerate() {
        if let Some(p) = p {
            pos[i] = *p;
            valid[i] = true;
            any_real = true;
        }
    }
    if !any_real {
        pos = fibonacci_sphere(n);
        valid = vec![true; n];
    } else {
        for (i, _) in positions.iter().enumerate() {
            if !valid[i] {
                // одна точка в центре масс реальных
                let c = centroid(&pos, &valid);
                pos[i] = c;
            }
        }
    }
    let (xs, ys, zs, has_pos) = norm_positions(&pos, &valid);

    // атрибуты: НТ и класс, если объявлены в ключах
    let mut nt = vec![0u8; n];
    let mut kl = vec![0u8; n];
    for (i, data) in extra.iter().enumerate() {
        if let Some(v) = resolve(data, &["nt", "neurotransmitter", "predictedNt"]) {
            nt[i] = nt_code(&v);
        }
        if let Some(v) = resolve(data, &["class", "klass", "flow"]) {
            kl[i] = klass_code(&v);
        }
    }
    let nt_types: Vec<f32> = nt
        .iter()
        .map(|&c| if c == 2 || c == 3 { -1.0 } else { 1.0 })
        .collect();
    let (type_idx, types) = intern(&names)?;

    // CSR из рёбер (уже отсортированы по источнику)
    let mut row_ptr = vec![0i32; n + 1];
    for &(si, _, _) in &edges {
        row_ptr[si + 1] += 1;
    }
    for i in 0..n {
        row_ptr[i + 1] += row_ptr[i];
    }
    let col: Vec<i32> = edges.iter().map(|&(_, di, _)| di as i32).collect();
    let val: Vec<f32> = edges.iter().map(|&(_, _, w)| w).collect();
    let body_id: Vec<u64> = (1..=n as u64).collect();

    Ok(MancData {
        n,
        s: edges.len(),
        xs,
        ys,
        zs,
        has_pos,
        nt,
        klass: kl,
        leg: vec![0u8; n],
        side: vec![0u8; n],
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Минимальный GraphML в стиле networkx: 3 узла с именами и координатами,
    /// 2 ребра с весами (ключи объявлены через `attr.name`).
    const FIXTURE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="node" attr.name="name" attr.type="string"/>
  <key id="d1" for="node" attr.name="x" attr.type="double"/>
  <key id="d2" for="node" attr.name="y" attr.type="double"/>
  <key id="d3" for="node" attr.name="z" attr.type="double"/>
  <key id="d4" for="edge" attr.name="weight" attr.type="double"/>
  <graph id="G" edgedefault="directed">
    <node id="n0">
      <data key="d0">sensory</data>
      <data key="d1">1.0</data>
      <data key="d2">2.0</data>
      <data key="d3">3.0</data>
    </node>
    <node id="n1">
      <data key="d0">inter</data>
      <data key="d1">4.0</data>
      <data key="d2">5.0</data>
      <data key="d3">6.0</data>
    </node>
    <node id="n2">
      <data key="d0">motor</data>
      <data key="d1">7.0</data>
      <data key="d2">8.0</data>
      <data key="d3">9.0</data>
    </node>
    <edge source="n0" target="n1">
      <data key="d4">2.5</data>
    </edge>
    <edge source="n0" target="n2">
      <data key="d4">7.0</data>
    </edge>
  </graph>
</graphml>"#;

    #[test]
    fn parses_nodes_edges_and_weights() {
        let m = from_graphml_str(FIXTURE).expect("фикстура должна парситься");
        assert_eq!(m.n, 3);
        assert_eq!(m.s, 2);
        assert_eq!(m.body_id, vec![1, 2, 3]);
        // Имена узлов сохранились (type = name узла).
        assert!(m.types.iter().any(|t| t == "sensory"));
        assert!(m.types.iter().any(|t| t == "motor"));
        // Позиции реальные (не синтетическая сфера) — координаты прошли.
        assert!(m.has_pos.iter().all(|&v| v != 0));
        // CSR: оба ребра из узла 0 → row_ptr = [0, 2, 2, 2], веса на месте.
        assert_eq!(m.row_ptr, vec![0, 2, 2, 2]);
        let mut w = m.val.clone();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(w, vec![2.5, 7.0]);
        assert!(m.col.iter().all(|&c| c == 1 || c == 2));
    }

    #[test]
    fn rejects_garbage_and_empty_graph() {
        assert!(from_graphml_str("not xml <").is_err());
        assert!(from_graphml_str("<graphml><graph/></graphml>").is_err()); // нет узлов
    }
}
