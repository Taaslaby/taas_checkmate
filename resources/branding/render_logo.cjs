// Rasterize neoMyelin.svg → neoMyelin.png (white on transparent).
// Usage: npx -y -p @resvg/resvg-js node render_logo.cjs
const { Resvg } = require("@resvg/resvg-js");
const fs = require("fs");
const path = require("path");

const here = __dirname;
const svg = fs.readFileSync(path.join(here, "neoMyelin.svg"), "utf8");

// Full wordmark (for the About dialog, tinted to theme at runtime).
// The window icon (neoMyelin_icon.png) is built from this PNG by make_icon.py.
const full = new Resvg(svg, {
  fitTo: { mode: "width", value: 1400 },
  background: "rgba(0,0,0,0)",
});
fs.writeFileSync(path.join(here, "neoMyelin.png"), full.render().asPng());

console.log("rendered neoMyelin.png");
