//! # GPU-рендерер облака точек: «сортировка + объёмные сферы» (`splat`)
//!
//! Свежий рендерер взамен первого наброска. Чем он отличается:
//!
//! 1. **CPU-сортировка по глубине каждый кадр** — точки проецируются и
//!    сортируются от дальних к ближним (far → near), как это делала
//!    программная запечка (`atlas::build_atlas_3d`). Ближние рисуются
//!    последними и аккуратно перекрывают дальние — ничего не «перемешивается»
//!    в кашу, как при неупорядоченном рисовании.
//! 2. **Объёмные сферы, а не плоские сплэты** — из uv-координаты биллборда
//!    восстанавливается нормаль выпуклой сферы, дальше полноценное
//!    освещение: диффузный свет сверху-слева, блик Phong, rim-свечение по
//!    контуру и затенение по глубине — каждый нейрон выглядит как шарик.
//! 3. **Нормальный альфа-блендинг** — перекрытия складываются в непрозрачное
//!    цветное ядро, а не выжигаются в белое (аддитивное сложение убрано),
//!    край сферы мягкий через `smoothstep` по альфе.
//! 4. **Крупные точки** — базовый размер 7 px, растут при зуме и у ближних
//!    к камере точек; размер и непрозрачность крутятся ползунками в GUI.
//!
//! Архитектура та же, что в `egui_wgpu::CallbackTrait`-рендерере: вершины —
//! биллборд-квады (TriangleStrip, 4 вершины на точку), точки читаются из
//! storage-буфера по `instance_index` (порядок = порядок сортировки),
//! uniform-буфер камеры общий для вершинного и фрагментного шейдеров.

use wgpu::util::DeviceExt as _;

/// Вершинные данные точки (совпадает с `struct PointData` в шейдере).
///
/// ВНИМАНИЕ (миспаринг раскладки, из-за которого «ломались» рендеры):
/// в WGSL `vec3<f32>` имеет выравнивание **16** байт, поэтому структура
/// выравнивается до 16 и storage-массив читается с шагом **64 байта**.
/// Rust `#[repr(C)]` выравнивает только по 4 — поэтому хвостовой паддинг
/// (`_pad1` + `_pad2`) добиваем руками до ровно 64 байт на точку.
/// Проверка `const _SIZE` ниже ловит рассинхрон на этапе компиляции.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PointData {
    pub pos: [f32; 3],
    /// Паддинг под выравнивание `vec3<f32>` (align 16) в WGSL.
    pub _pad0: f32,
    pub col: [f32; 4],
    /// Нормированное время первого спайка [0..1]; 0 = нет трассы.
    pub trace: f32,
    /// 1 = выбранный нейрон (белый + крупнее).
    pub sel: f32,
    /// Множитель размера (трасса/выбор).
    pub boost: f32,
    /// Живое свечение [0..1]: спарк от потоковой симуляции — белый + крупнее.
    pub glow: f32,
    /// WGSL выравнивает struct до 16 → хвостовой паддинг до 64 байт.
    pub _pad1: f32,
    pub _pad2: [f32; 3],
}

/// Компиляторная проверка: Rust-структура обязана быть ровно 64 байта,
/// иначе storage-массив в шейдере читает точки со сдвигом.
const _SIZE: () = assert!(std::mem::size_of::<PointData>() == 64);

/// Uniforms камеры (совпадает с `struct Uniforms` в шейдере): 64 байта.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Uniforms {
    pub yaw: f32,
    pub pitch: f32,
    pub persp: f32,
    pub ppp: f32,         // pixels per point (HiDPI)
    pub scale: f32,       // zoom * BASE — мировые единицы → точки
    pub base_size: f32,   // базовый размер точки в точках (ползунок «размер»)
    pub alpha: f32,       // непрозрачность точки [0..1] (ползунок «непр»)
    pub render_mode: f32, // 0 spheres, 1 crisp points, 2 neon
    pub pan: [f32; 2],    // в точках, относительно левого верхнего угла канваса
    pub rect: [f32; 2],   // размер канваса в точках
    pub _pad: [f32; 4],
}

/// WGSL-шейдер: вершина — биллборд, фрагмент — гауссово пятно.
const SHADER: &str = r#"
struct Uniforms {
    yaw: f32, pitch: f32, persp: f32, ppp: f32,
    scale: f32, base_size: f32, alpha: f32, render_mode: f32,
    pan: vec2<f32>, rect: vec2<f32>,
    _pad: vec4<f32>,
};

struct PointData {
    pos: vec3<f32>,
    col: vec4<f32>,
    trace: f32,
    sel: f32,
    boost: f32,
    glow: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> pts: array<PointData>;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) depth: f32, // 0..1: 0 = далеко, 1 = близко
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
    let p = pts[ii];
    // Вращение yaw (вокруг Z) + pitch (вокруг X) + перспектива —
    // та же математика, что в atlas::project3d.
    let q = p.pos - vec3(0.5);
    let sy = sin(u.yaw);
    let cy = cos(u.yaw);
    let sp = sin(u.pitch);
    let cp = cos(u.pitch);
    let x1 = q.x * cy - q.y * sy;
    let y1 = q.x * sy + q.y * cy;
    let y2 = y1 * cp - q.z * sp;
    let z2 = y1 * sp + q.z * cp;
    // Перспектива: 0 = орто (s = 1), иначе точки ближе — крупнее.
    // ВАЖНО: +0.5 в конце — ровно как в atlas::project3d. Без него облако
    // уезжает на полэкрана вверх-влево (координаты остаются в [-0.5,0.5]²).
    let denom = max(0.18, 1.0 - z2 * u.persp);
    let s = min(3.0, 1.0 / denom);
    let scr = vec2(x1 * s + 0.5, y2 * s + 0.5); // [0,1]²
    let px = scr * u.scale + u.pan;              // точки, от угла канваса
    var ndc = px / u.rect * 2.0 - 1.0;
    ndc.y = -ndc.y;                          // экранный Y вниз
    // Размер: base_size * ppp (пиксели), ближе — крупнее, при зуме растёт.
    let zoom_k = clamp(u.scale / 720.0, 0.4, 10.0);
    let size_px = u.base_size * u.ppp * (0.35 + 0.65 * s) * (1.0 + p.boost + p.glow * 1.5) * (0.55 + 0.45 * zoom_k);
    let span = vec2(size_px / (u.rect.x * u.ppp), size_px / (u.rect.y * u.ppp));
    let corner = vec2(f32(vi & 1u), f32((vi >> 1u) & 1u));
    var out: VSOut;
    out.pos = vec4(ndc + (corner - vec2(0.5)) * span, 0.5 - 0.5 * z2, 1.0);
    out.uv = corner;
    out.depth = 0.5 + 0.5 * z2;
    // Цвет: NT → heatmap трассы → бело-горячий спарк живого сигнала → белый выбор.
    var c = p.col;
    if p.trace > 0.0 {
        let heat = mix(vec3(0.08, 0.30, 0.85), vec3(0.95, 0.20, 0.12), p.trace);
        c = vec4(heat, 1.0);
    }
    if p.glow > 0.0 {
        // Свечение уводит цвет в белый (ядро вспышки), оставляя оттенок базового.
        let g = min(p.glow, 1.0);
        c = mix(c, vec4(1.0, 1.0, 1.0, 1.0), g * 0.9);
    }
    if p.sel > 0.5 {
        c = vec4(1.0, 1.0, 1.0, 1.0);
    }
    out.color = c;
    return out;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>, @location(1) color: vec4<f32>, @location(2) depth: f32) -> @location(0) vec4<f32> {
    let dx = uv.x - 0.5;
    let dy = uv.y - 0.5;
    let r2 = dx * dx + dy * dy;
    if (r2 > 0.25) {
        discard; // вне сферы — круглый край, не квадрат
    }
    if u.render_mode > 0.5 && u.render_mode < 1.5 {
        return vec4(color.rgb, color.a * u.alpha);
    }
    // Нормаль выпуклой сферы: из uv автоматически выходит единичная.
    let n = vec3(2.0 * dx, 2.0 * dy, 2.0 * sqrt(max(0.0, 0.25 - r2)));
    // Свет сверху-слева-спереди: диффузное освещение.
    let L = normalize(vec3(-0.45, 0.6, 0.65));
    let diff = max(dot(n, L), 0.0);
    // Блик Phong (камера в +Z).
    let h = normalize(L + vec3(0.0, 0.0, 1.0));
    let spec = pow(max(dot(n, h), 0.0), 48.0);
    // Rim: край сферы мягко подсвечивается — шары читаются на фоне
    // (смягчён: раньше выжигал контур на ярких точках).
    let rim = pow(1.0 - n.z, 2.0) * 0.22;
    // Затенение по глубине: дальние тусклее, ближние ярче.
    let shade = 0.30 + 0.80 * depth;
    var lit = color.rgb * (0.35 + 0.85 * diff) + vec3(spec * 0.75) + color.rgb * rim;
    if u.render_mode > 1.5 {
        lit = color.rgb * 1.45 + vec3(0.12, 0.42, 0.30) + vec3(spec * 1.2);
    }
    // Дистанционный туман к цвету фона: глубинный кью — дальние точки
    // тают в фоне, облако читается объёмнее (смешение по view depth).
    let fog = pow(1.0 - depth, 2.0) * 0.55;
    let bg = vec3(0.055, 0.055, 0.094); // тот же тёмный индиго, что фон атласа
    let shaded = mix(lit * shade, bg, fog);
    // Мягкий край по альфе — без квадратных углов биллборда.
    let a = smoothstep(0.25, 0.20, r2) * u.alpha;
    return vec4(shaded, color.a * a);
}
"#;

/// Нормальный альфа-блендинг (src_alpha / 1-src_alpha): ближние точки
/// перекрывают дальние (порядок задан сортировкой), центр — непрозрачное
/// цветное ядро, а не белая каша от аддитивного сложения.
const ALPHA_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

/// Постоянные GPU-ресурсы, живущие в `CallbackResources` между кадрами.
struct SplatGpu {
    pipeline: wgpu::RenderPipeline,
    points: wgpu::Buffer, // storage
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    capacity: usize,
}

impl SplatGpu {
    /// Создаёт пайплайн + буферы под `n` точек. Формат поверхности берём из
    /// `RenderState` при старте (см. `ConnApp::gpu_target_format`).
    fn new(
        device: &wgpu::Device,
        points: &[PointData],
        uniforms: &Uniforms,
        format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // uniform читают и вершина (камера), и фрагмент (alpha).
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat_pl"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[], // все данные — из storage-буфера по instance_index
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(ALPHA_BLEND),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let points_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("splat_points"),
            contents: bytemuck::cast_slice(points),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("splat_uniforms"),
            contents: bytemuck::bytes_of(uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat_bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: points_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            points: points_buf,
            uniforms: uniforms_buf,
            bind_group,
            capacity: points.len(),
        }
    }
}

/// Колбэк: каждый кадр пишет свежеотсортированные точки и uniform'ы камеры,
/// затем рисует их в рендер-пасс egui.
pub(crate) struct SplatCallback {
    points: Vec<PointData>,
    uniforms: Uniforms,
    format: wgpu::TextureFormat,
}

const MAX_SPLAT_POINTS: usize = 120_000;

impl SplatCallback {
    pub(crate) fn new(
        points: Vec<PointData>,
        uniforms: Uniforms,
        format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let points = if points.is_empty() {
            vec![PointData {
                pos: [0.5, 0.5, 0.5],
                _pad0: 0.0,
                col: [0.0; 4],
                trace: 0.0,
                sel: 0.0,
                boost: -1.0,
                glow: 0.0,
                _pad1: 0.0,
                _pad2: [0.0; 3],
            }]
        } else {
            points
        };
        Self {
            points,
            uniforms,
            // Формат поверхности известен только при wgpu-бэкенде (glow = None).
            format: format.unwrap_or(wgpu::TextureFormat::Bgra8Unorm),
        }
    }
}

impl SplatCallback {
    /// Внутренняя подготовка GPU-ресурсов (вызывается из `prepare` внутри
    /// `catch_unwind` — не trait-метод).
    fn prepare_inner(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let n = self.points.len().min(MAX_SPLAT_POINTS);
        // Пересоздаём ресурсы, если точек стало больше (изменились фильтры
        // или включилась трасса); иначе — просто пишем свежие данные.
        let gpu = match resources.get_mut::<SplatGpu>() {
            Some(g) if g.capacity >= n => g,
            _ => {
                resources.remove::<SplatGpu>();
                resources.insert(SplatGpu::new(
                    device,
                    &self.points[..n],
                    &self.uniforms,
                    self.format,
                ));
                resources.get_mut::<SplatGpu>().expect("just inserted")
            }
        };
        if n > 0 {
            queue.write_buffer(&gpu.points, 0, bytemuck::cast_slice(&self.points[..n]));
        }
        queue.write_buffer(&gpu.uniforms, 0, bytemuck::bytes_of(&self.uniforms));
        Vec::new()
    }
}

impl egui_wgpu::CallbackTrait for SplatCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // GPU-ошибка (вал неправильных данных, OOM буфера) не должна ронять
        // процесс: пропускаем кадр, следующий кадр попробует снова.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.prepare_inner(device, queue, screen_descriptor, egui_encoder, resources)
        }))
        .unwrap_or_default()
    }

    fn paint(
        &self,
        _info: eframe::egui::epaint::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(gpu) = resources.get::<SplatGpu>() else {
            return;
        };
        render_pass.set_pipeline(&gpu.pipeline);
        render_pass.set_bind_group(0, &gpu.bind_group, &[]);
        // 4 вершины (квад-биллборд) на каждую точку; порядок точек =
        // порядок сортировки far → near.
        render_pass.draw(0..4, 0..self.points.len().min(MAX_SPLAT_POINTS) as u32);
    }
}
