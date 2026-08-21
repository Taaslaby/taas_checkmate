//! Windows resources — app icon + file associations (old Apple Aqua skeuomorph).

fn main() {
    #[cfg(target_os = "windows")]
    {
        let mut res = winresource::WindowsResource::new();
        // App icon — neoMyelin (fallback to TAAS skeuo if missing)
        res.set_icon("../../resources/branding/icon_taas_skeuo.ico");
        // File associations — .taas (TAAS native), .nml (NeuroML), .swc (morphology)
        res.set_manifest(
            r#"
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true</dpiAware>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>
    </windowsSettings>
  </application>
</assembly>
"#,
        );
        if let Err(e) = res.compile() {
            eprintln!("winresource: {e}");
        }
    }
}
