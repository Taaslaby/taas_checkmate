//! # Карточка нейрона (`cards::neuron`)
//!
//! Поля «ключ: значение», плашки класса/НТ/степеней 2×2, топ-5 связей и
//! кнопки ego-графа. Сборка данных — в чистой функции [`neuron_card_data`]
//! (owned-строки, без UI), рендер — в [`render_neuron_card`], который
//! возвращает команду [`NeuronAction`]. Оркестратор (`card_neuron` в
//! `inspector.rs`) применяет команду к view-состоянию и атласу.

use eframe::egui;

use elegance::{BadgeTone, Button as ElegButton};

use egui_phosphor::regular as ic;

use crate::app::tokens::{self, icon_label};
use crate::io::data::MancData;

/// Собранная карточка нейрона — owned-данные для рендера.
pub(crate) struct NeuronCardData {
    pub heading: String,
    /// Пары «поле → значение» + пояснение в тултипе.
    pub fields: Vec<(&'static str, String, &'static str)>,
    /// Топ синаптических партнёров: «← входит от», «→ выходит в».
    pub conns: Vec<String>,
    pub klass: String,
    pub nt: String,
    pub od: usize,
    pub id_: usize,
}

/// Чистое «что показать» для карточки нейрона: по данным и индексу собирает
/// owned-строки (заголовок, поля, связи, плашки). Не трогает UI и состояние —
/// легко тестировать и переиспользовать.
pub(crate) fn neuron_card_data(d: &mut MancData, i: usize) -> NeuronCardData {
    let ty = d.type_name(i);
    // Заголовок — id + тип: сразу понятно, что за нейрон.
    let heading = if ty.is_empty() {
        format!("#{i}")
    } else {
        format!("#{i} {ty}")
    };
    // В MANC пустая аннотация записана строкой «None» — показываем «—».
    let dash = |s: &str| {
        if s.is_empty() || s == "None" {
            "—".to_string()
        } else {
            s.to_string()
        }
    };
    // Ключи — человекочитаемые подписи на русском, а не сырые имена полей.
    let fields = vec![
        (
            "ID тела",
            d.body_id[i].to_string(),
            "Уникальный номер нейрона в базе MANC",
        ),
        (
            "Подкласс",
            dash(d.subclass(i)),
            "Функциональная группа клетки (код из аннотаций)",
        ),
        (
            "Мишень",
            dash(d.target(i)),
            "Область нейропиля, в которую проецируется нейрон",
        ),
        (
            "Нога",
            dash(d.leg_name(i)),
            "Сегмент ноги, с которым связан нейрон (T1/T2/T3…)",
        ),
        (
            "Сторона",
            dash(d.side_name(i)),
            "Латеральность: L = лево, R = право",
        ),
    ];
    // Связи: топ-5 партнёров (мультисинапсы агрегированы) — отвечает на
    // вопрос «что делает этот нейрон», даже если аннотаций у него нет.
    // Сортируем по весу desc, чтобы сильный партнёр всегда сверху.
    let mut edges = d.ego_edges_agg(i, 5);
    edges.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let conns: Vec<String> = edges
        .into_iter()
        .map(|(t, w, out, n)| {
            let ty = d.type_name(t);
            let name = if ty.is_empty() {
                format!("#{t}")
            } else {
                ty.to_string()
            };
            let syn = if n > 1 {
                format!(" · ×{n} синапсов")
            } else {
                String::new()
            };
            // w — tabular-nums: равная ширина цифр для сравнения весов
            format!("{} {name} · w={w:.0}{syn}", if out { "→" } else { "←" })
        })
        .collect();
    NeuronCardData {
        heading,
        fields,
        conns,
        klass: d.klass(i).name().to_string(),
        nt: d.nt(i).name().to_string(),
        od: d.out_deg(i),
        id_: d.in_deg(i),
    }
}

/// Команда карточки нейрона — рендер не знает про `ConnApp`, только про
/// собранные данные и счётчики ego-графа; действие применяет оркестратор.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NeuronAction {
    /// 🌐 — открыть большой ego-таб.
    OpenTab,
    /// ⤢ — открепить ego-граф в перетаскиваемое окно.
    UndockWindow,
    /// ⌖ — навести атлас на окрестность выбранного нейрона.
    FocusNeighborhood,
}

/// Чистый рендер карточки нейрона: принимает собранные [`NeuronCardData`]
/// и счётчики ego-графа, а действия возвращает как [`NeuronAction`] —
/// оркестратор применяет их к view-состоянию/атласу. Это и есть граница:
/// карточка физически не может трогать то, что ей не передали.
pub(crate) fn render_neuron_card(
    ui: &mut egui::Ui,
    card: &NeuronCardData,
    ego_nn: usize,
    ego_ne: usize,
) -> Option<NeuronAction> {
    let mut action = None;
    crate::app::widgets::section_card(ui, ic::BRAIN, &card.heading, |ui| {
        crate::app::widgets::key_value_fields(ui, &card.fields);
        ui.add_space(tokens::SP_SM);
        // Плашки ровной сеткой 2×2: каждая растянута на
        // половину карточки — одинаковые ширина и высота.
        crate::app::widgets::badge_grid(
            ui,
            2,
            &[
                (&card.klass, BadgeTone::Info),
                (&card.nt, BadgeTone::Neutral),
            ],
        );
        ui.add_space(tokens::SP_XS);
        crate::app::widgets::badge_grid(
            ui,
            2,
            &[
                (&format!("out {}", card.od), BadgeTone::Neutral),
                (&format!("in {}", card.id_), BadgeTone::Neutral),
            ],
        );
        // Связи — «что делает» нейрон: даже без аннотаций
        // видно, кто его драйвит и куда он сам влияет.
        if !card.conns.is_empty() {
            ui.add_space(tokens::SP_SM);
            ui.label(tokens::eyebrow("Связи (топ-5, по весу)"));
            for c in &card.conns {
                // Моноширинный w — tabular-nums для быстрого сравнения весов.
                ui.label(egui::RichText::new(c).family(egui::FontFamily::Monospace).size(tokens::TEXT_MONO))
                    .on_hover_text(c);
            }
        }
        ui.add_space(tokens::SP_MD);
        // ego-граф: открыть большой таб / открепить окно / навести атлас.
        ui.horizontal_wrapped(|ui| {
            if ui
                .add(ElegButton::new(icon_label(
                    ic::GRAPH,
                    &format!("Ego-граф ({ego_nn}/{ego_ne})"),
                    tokens::ICON_SM,
                )))
                .clicked()
            {
                action = Some(NeuronAction::OpenTab);
            }
            if ui
                .add(ElegButton::new(tokens::icon(
                    ic::ARROWS_OUT,
                    tokens::ICON_SM,
                )))
                .on_hover_text("Открепить в окно")
                .clicked()
            {
                action = Some(NeuronAction::UndockWindow);
            }
            if ui
                .add(ElegButton::new(tokens::icon(
                    ic::CROSSHAIR,
                    tokens::ICON_SM,
                )))
                .on_hover_text("Навести атлас на окрестность")
                .clicked()
            {
                action = Some(NeuronAction::FocusNeighborhood);
            }
        });
    });
    action
}
