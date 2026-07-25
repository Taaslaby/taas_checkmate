use crate::reader::TaasReader;

pub fn inspect_taas(path: &str) -> Result<String, String> {
    let reader = TaasReader::from_file(path)?;
    let mut lines = Vec::new();

    lines.push(format!(".taas archive: {}", path));
    lines.push(format!("   Version: {}", reader.manifest.version));
    lines.push(format!("   Created: {}", reader.manifest.created));
    lines.push(format!(
        "   Components: {}",
        reader.manifest.components.len()
    ));
    lines.push(String::new());

    for comp in &reader.manifest.components {
        let n_elems: usize = comp.shape.iter().product();
        let size_str = if comp.size > 1024 * 1024 {
            format!("{:.1} MB", comp.size as f64 / (1024.0 * 1024.0))
        } else if comp.size > 1024 {
            format!("{:.1} KB", comp.size as f64 / 1024.0)
        } else {
            format!("{} B", comp.size)
        };
        lines.push(format!(
            "   {:<25} {}  {:>8} {} {:?}",
            comp.name, comp.component_type, size_str, comp.dtype, comp.shape
        ));
        lines.push(format!(
            "   {:>29} offset={}  n_elems={}",
            "", comp.offset, n_elems
        ));
    }

    if !reader.manifest.metadata.is_empty() {
        lines.push(String::new());
        lines.push("   Metadata:".to_string());
        for (k, v) in &reader.manifest.metadata {
            lines.push(format!("      {:<20} = {}", k, v));
        }
    }

    lines.push(String::new());
    let file_len = reader.total_size();
    let total_size = if file_len > 1024 * 1024 {
        format!("{:.1} MB", file_len as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} KB", file_len as f64 / 1024.0)
    };
    lines.push(format!("   Total size: {}", total_size));

    Ok(lines.join("\n"))
}
