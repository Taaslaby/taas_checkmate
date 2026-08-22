//! # Подсветка синтаксиса NeuroML (`app::neuroml::highlight`)
//!
//! Чистый, не знающий про UI слой подсветки: разбивает XML-текст на токены
//! ([`xml_tokens`]) и собирает [`LayoutJob`] для `TextEdit::code_editor`
//! ([`build_job`]). Никакого состояния редактора — только входной текст +
//! номера строк с ошибками, поэтому всё тестируется как чистые функции.
//!
//! Отрисовка (окно, тулбар, список ошибок) живёт в `super::editor` — здесь
//! только «как раскрасить байты».

use std::ops::Range;

use eframe::egui::{
    epaint::text::ByteIndex,
    text::{LayoutJob, LayoutSection, TextFormat},
    Color32, FontId, Stroke,
};

use crate::app::shell::theme::DANGER;

// ── Подсветка синтаксиса (XML) ─────────────────────────────────

/// Вид токена для подсветки.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tok {
    /// `<!-- … -->`
    Comment,
    /// Имя тега (`izhikevichCell` в `<izhikevichCell …>`).
    TagName,
    /// Имя атрибута (`id` в `id="rs"`).
    AttrName,
    /// Значение атрибута (`"rs"`).
    AttrValue,
    /// Пунктуация тега (`<`, `>`, `/`, `=`).
    TagPunct,
    /// Всё остальное (текст между тегами).
    Text,
}

/// Разбить XML-текст на подсвечиваемые токены (чистая функция — тестируется).
///
/// Токены не перекрываются и покрывают не весь текст — промежутки между ними
/// (например, пробелы между атрибутами) заполняются default-секциями в
/// [`build_job`].
pub(crate) fn xml_tokens(text: &str) -> Vec<(Range<usize>, Tok)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let n = bytes.len();
    while i < n {
        if text[i..].starts_with("<!--") {
            let end = text[i + 4..]
                .find("-->")
                .map(|p| i + 4 + p + 3)
                .unwrap_or(n);
            out.push((i..end, Tok::Comment));
            i = end;
            continue;
        }
        if bytes[i] == b'<' {
            // Тег: идём до `>` вне кавычек.
            let mut j = i + 1;
            let mut quote = None;
            while j < n {
                match bytes[j] {
                    b'"' | b'\'' if quote.is_none() => quote = Some(bytes[j]),
                    q if Some(q) == quote => quote = None,
                    b'>' if quote.is_none() => break,
                    _ => {}
                }
                j += 1;
            }
            let end = if j < n { j + 1 } else { n };
            tag_tokens(text, i, end, &mut out);
            i = end;
            continue;
        }
        // Текст до следующего тега/комментария.
        let mut j = i;
        while j < n && bytes[j] != b'<' {
            j += 1;
        }
        if j > i {
            out.push((i..j, Tok::Text));
        }
        i = j;
    }
    out
}

/// Токены внутри `<…>`: имя тега, затем пары «имя = значение».
fn tag_tokens(text: &str, start: usize, end: usize, out: &mut Vec<(Range<usize>, Tok)>) {
    let bytes = text.as_bytes();
    out.push((start..(start + 1).min(end), Tok::TagPunct)); // `<`
    let mut i = start + 1;
    // Имя тега: до пробела, `/`, `>`.
    let name_start = i;
    while i < end && !bytes[i].is_ascii_whitespace() && bytes[i] != b'>' && bytes[i] != b'/' {
        i += 1;
    }
    if i > name_start {
        out.push((name_start..i, Tok::TagName));
    }
    // Атрибуты.
    while i < end {
        while i < end && (bytes[i].is_ascii_whitespace() || bytes[i] == b'/') {
            i += 1;
        }
        if i >= end || bytes[i] == b'>' {
            break;
        }
        let a_start = i;
        while i < end && !bytes[i].is_ascii_whitespace() && bytes[i] != b'=' && bytes[i] != b'>' {
            i += 1;
        }
        if i > a_start {
            out.push((a_start..i, Tok::AttrName));
        }
        while i < end && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < end && bytes[i] == b'=' {
            out.push((i..i + 1, Tok::TagPunct));
            i += 1;
            while i < end && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < end && (bytes[i] == b'"' || bytes[i] == b'\'') {
                let q = bytes[i];
                let v_start = i;
                i += 1;
                while i < end && bytes[i] != q {
                    i += 1;
                }
                // Кавычка не закрылась — значение тянется до конца тега и
                // съедает возможный `>`: отдельного закрывающего `>` нет,
                // иначе тот же байт попал бы в два токена (перекрытие).
                let unterminated = i >= end;
                i = (i + 1).min(end);
                out.push((v_start..i, Tok::AttrValue));
                if unterminated {
                    return;
                }
            }
        }
    }
    // Закрывающий `>` (или `/>`).
    if end > start && bytes[end - 1] == b'>' {
        out.push((end - 1..end, Tok::TagPunct));
    }
}

// Палитра подсветки — VS Code Dark+ на проф тёмном BG_INSET.
// Имя тега — акцент темы (параметр `accent`), остальное — проверенные тона
// данных, не хром оболочки.
/// Имя атрибута — песочный (VS Code #9CDCFE-аналог → кремовый читаем на тёмном).
const SYN_ATTR: Color32 = Color32::from_rgb(0xDC, 0xDC, 0xAA);
/// Значение атрибута — терракотовый (#CE9178).
const SYN_VALUE: Color32 = Color32::from_rgb(0xCE, 0x91, 0x78);
/// Комментарий — приглушённый зелёный (#6A9955), не серый: сразу видно, что это не код.
const SYN_COMMENT: Color32 = Color32::from_rgb(0x6A, 0x99, 0x55);
/// Пунктуация тега — нейтральный 60% белый.
const SYN_PUNCT: Color32 = Color32::from_rgb(0x8A, 0x8A, 0x8A);
/// Обычный текст — тёплый белый #D4D4D4.
const SYN_TEXT: Color32 = Color32::from_rgb(0xD4, 0xD4, 0xD4);

/// Цвет токена по виду (см. палитру выше). Имя тега — текущий акцент.
fn tok_color(t: Tok, accent: Color32) -> Color32 {
    match t {
        Tok::Comment => SYN_COMMENT,
        Tok::TagName => accent,
        Tok::AttrName => SYN_ATTR,
        Tok::AttrValue => SYN_VALUE,
        Tok::TagPunct => SYN_PUNCT,
        Tok::Text => SYN_TEXT,
    }
}

/// Собрать `LayoutJob` для TextEdit: токены + фон строк с ошибками.
///
/// `accent` — текущий сигнальный акцент (`theme::accent(ui)`): им красится
/// имя тега, чтобы подсветка следовала выбранному `AccentPreset`.
/// `error_lines` — 1-based номера строк с ошибками. Маппинг «байт → строка»
/// считается один раз (двоичный поиск по началам строк), а не по секции.
pub(crate) fn build_job(
    text: &str,
    error_lines: &[u32],
    cursor_line: u32,
    wrap_width: f32,
    font: FontId,
    accent: Color32,
) -> LayoutJob {
    let mut job = LayoutJob {
        text: text.to_owned(),
        ..Default::default()
    };
    job.wrap.max_width = wrap_width;
    job.break_on_newline = true;

    // Начала строк (byte-офсет) — по ним бинарным поиском получаем 1-based
    // номер строки любого байта. Считается один раз на кадр, а не на секцию.
    let mut line_starts = vec![0usize];
    for (k, b) in text.bytes().enumerate() {
        if b == b'\n' {
            line_starts.push(k + 1);
        }
    }
    let mut sorted_errors = error_lines.to_vec();
    sorted_errors.sort_unstable();
    let line_of_byte = |byte: usize| -> u32 {
        // Число начал строк ≤ byte (линия 1 — 0..a1).
        line_starts.partition_point(|&s| s <= byte) as u32
    };
    let line_has_error =
        |byte: usize| -> bool { sorted_errors.binary_search(&line_of_byte(byte)).is_ok() };
    // Подсветка строки курсора (1-based; 0 = нет). Фон рисуется в LayoutJob —
    // под глифами, а не поверх, и прокручивается вместе с текстом.
    let line_has_cursor =
        |byte: usize| -> bool { cursor_line > 0 && line_of_byte(byte) == cursor_line };

    let base = TextFormat {
        font_id: font.clone(),
        color: tok_color(Tok::Text, accent),
        ..Default::default()
    };

    // Полное покрытие текста секциями: токены + default-промежутки.
    let tokens = xml_tokens(text);
    let mut cursor = 0usize;
    for (range, tok) in &tokens {
        if range.start > cursor {
            push_section(
                &mut job,
                cursor..range.start,
                &base,
                line_has_error(cursor),
                line_has_cursor(cursor),
            );
        }
        let mut fmt = TextFormat {
            font_id: font.clone(),
            color: tok_color(*tok, accent),
            ..Default::default()
        };
        if *tok == Tok::Comment {
            fmt.italics = true;
        }
        push_section(
            &mut job,
            range.clone(),
            &fmt,
            line_has_error(range.start),
            line_has_cursor(range.start),
        );
        cursor = range.end;
    }
    if cursor < text.len() {
        push_section(
            &mut job,
            cursor..text.len(),
            &base,
            line_has_error(cursor),
            line_has_cursor(cursor),
        );
    }
    job
}

fn push_section(
    job: &mut LayoutJob,
    range: Range<usize>,
    fmt: &TextFormat,
    err: bool,
    cursor: bool,
) {
    let mut f = fmt.clone();
    if err {
        // squiggle + лёгкий красноватый фон (14% DANGER) — как в VS Code
        f.background = Color32::from_rgba_unmultiplied(DANGER.r(), DANGER.g(), DANGER.b(), 0x24);
        f.underline = Stroke::new(1.4, DANGER);
    } else if cursor {
        // Строка курсора — заметнее: 6% белого на тёмном BG_INSET читается
        f.background = Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, 16);
    }
    job.sections.push(LayoutSection {
        leading_space: 0.0,
        byte_range: ByteIndex(range.start)..ByteIndex(range.end),
        format: f,
    });
}

// ── Тесты ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = r#"<!-- комментарий -->
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="t">
  <izhikevichCell id="rs" v0="-70mV" a="0.02"/>
  текст между тегами
</neuroml>
"#;

    /// Фиксированный акцент для тестов (не зависит от установленной темы).
    const TEST_ACCENT: Color32 = Color32::from_rgb(0x5A, 0xD6, 0xB2);

    #[test]
    fn tokens_do_not_overlap() {
        let tokens = xml_tokens(DOC);
        let mut prev = 0usize;
        for (r, _) in &tokens {
            assert!(r.start >= prev, "перекрытие токенов: {r:?}");
            assert!(r.end <= DOC.len(), "токен за границей: {r:?}");
            prev = r.end;
        }
        assert!(!tokens.is_empty());
    }

    /// `build_job` заполняет пробелы между токенами default-секциями — итог
    /// должен покрывать весь текст без перекрытий (иначе LayoutJob сломан).
    #[test]
    fn build_job_covers_all_text() {
        let job = build_job(DOC, &[2], 0, 200.0, FontId::monospace(12.5), TEST_ACCENT);
        let mut prev = 0usize;
        for s in &job.sections {
            let start = s.byte_range.start.0;
            let end = s.byte_range.end.0;
            assert!(start >= prev, "перекрытие секций: {start} vs {prev}");
            prev = end;
        }
        assert_eq!(prev, DOC.len());
        assert_eq!(job.text, DOC);
    }

    /// Фон ошибки должен попадать ровно на строку с ошибкой (1-based), а не
    /// на соседнюю — регрессия на off-by-one в маппинге «байт → строка».
    #[test]
    fn error_background_targets_the_right_line() {
        // Ошибка только на строке 2 (<neuroml …>) — строка 3 чистая.
        let job = build_job(DOC, &[2], 0, 200.0, FontId::monospace(12.5), TEST_ACCENT);
        let section = |needle: &str| {
            job.sections
                .iter()
                .find(|s| &DOC[s.byte_range.start.0..s.byte_range.end.0] == needle)
                .unwrap_or_else(|| panic!("секция `{needle}` не найдена"))
        };
        // Имя тега на строке 2 — с фоном ошибки (не прозрачный).
        assert_ne!(section("neuroml").format.background, Color32::TRANSPARENT);
        // Имя тега на строке 3 — без фона.
        assert_eq!(
            section("izhikevichCell").format.background,
            Color32::TRANSPARENT
        );
    }

    /// Строка курсора получает фоновую подсветку, остальные — прозрачный фон.
    #[test]
    fn cursor_line_gets_background() {
        let job = build_job(DOC, &[], 3, 200.0, FontId::monospace(12.5), TEST_ACCENT);
        // Секция имени тега на строке 3 (<izhikevichCell …>) — с фоном курсора.
        let section = |needle: &str| {
            job.sections
                .iter()
                .find(|s| &DOC[s.byte_range.start.0..s.byte_range.end.0] == needle)
                .unwrap_or_else(|| panic!("секция `{needle}` не найдена"))
        };
        assert_ne!(
            section("izhikevichCell").format.background,
            Color32::TRANSPARENT
        );
        assert_eq!(section("neuroml").format.background, Color32::TRANSPARENT);
    }

    #[test]
    fn tokens_recognize_kinds() {
        let tokens = xml_tokens(DOC);
        assert!(tokens.iter().any(|(_, t)| *t == Tok::Comment));
        assert!(tokens.iter().any(|(_, t)| *t == Tok::TagName));
        assert!(tokens.iter().any(|(_, t)| *t == Tok::AttrName));
        assert!(tokens.iter().any(|(_, t)| *t == Tok::AttrValue));
        assert!(tokens.iter().any(|(_, t)| *t == Tok::Text));
        // Имя тега — именно izhikevichCell.
        let tag: Vec<&str> = tokens
            .iter()
            .filter(|(_, t)| *t == Tok::TagName)
            .map(|(r, _)| &DOC[r.clone()])
            .collect();
        assert!(tag.contains(&"izhikevichCell"), "got {tag:?}");
    }

    #[test]
    fn attr_value_keeps_quotes() {
        let tokens = xml_tokens(r#"<a id="rs"/>"#);
        let v: Vec<&str> = tokens
            .iter()
            .filter(|(_, t)| *t == Tok::AttrValue)
            .map(|(r, _)| &r#"<a id="rs"/>"#[r.clone()])
            .collect();
        assert_eq!(v, vec!["\"rs\""]);
    }

    #[test]
    fn single_quoted_attr_value() {
        let doc = "<a id='rs'/>";
        let tokens = xml_tokens(doc);
        let v: Vec<&str> = tokens
            .iter()
            .filter(|(_, t)| *t == Tok::AttrValue)
            .map(|(r, _)| &doc[r.clone()])
            .collect();
        assert_eq!(v, vec!["'rs'"]);
    }

    /// `>` внутри кавычек — часть значения, а не конец тега.
    #[test]
    fn greater_than_inside_quoted_value() {
        let doc = "<a id=\"a>b\"/>";
        let tokens = xml_tokens(doc);
        let v: Vec<&str> = tokens
            .iter()
            .filter(|(_, t)| *t == Tok::AttrValue)
            .map(|(r, _)| &doc[r.clone()])
            .collect();
        assert_eq!(v, vec!["\"a>b\""]);
    }

    /// Незакрытая кавычка в значении атрибута съедает `>` — без перекрытия:
    /// значение покрывает хвост тега, отдельного закрывающего `>` нет.
    #[test]
    fn unterminated_attr_value_swallows_tag_end_without_overlap() {
        let doc = "<a id=\"x>";
        let tokens = xml_tokens(doc);
        let mut prev = 0usize;
        for (r, _) in &tokens {
            assert!(r.start >= prev, "перекрытие токенов: {r:?}");
            prev = r.end;
        }
        let v: Vec<&str> = tokens
            .iter()
            .filter(|(_, t)| *t == Tok::AttrValue)
            .map(|(r, _)| &doc[r.clone()])
            .collect();
        assert_eq!(v, vec!["\"x>"]);
    }

    /// Атрибут без значения (`disabled`) не паникует и не съедает `>`.
    #[test]
    fn boolean_attribute_without_value() {
        let doc = "<a disabled>";
        let tokens = xml_tokens(doc);
        let names: Vec<&str> = tokens
            .iter()
            .filter(|(_, t)| *t == Tok::AttrName)
            .map(|(r, _)| &doc[r.clone()])
            .collect();
        assert_eq!(names, vec!["disabled"]);
        assert!(!tokens.iter().any(|(_, t)| *t == Tok::AttrValue));
        // Закрывающий `>` всё же распознан пунктуацией.
        assert!(tokens
            .iter()
            .any(|(r, t)| *t == Tok::TagPunct && &doc[r.clone()] == ">"));
    }

    /// Незакрытый комментарий тянется до конца файла — без паники и пропусков.
    #[test]
    fn unterminated_comment_runs_to_eof() {
        let doc = "<!-- never closed";
        let tokens = xml_tokens(doc);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].0, 0..doc.len());
        assert_eq!(tokens[0].1, Tok::Comment);
    }

    /// CRLF не ломает покрытие текста секциями (вся строка считается целой).
    #[test]
    fn crlf_line_endings_cover_all_text() {
        let doc = "<a id=\"x\"/>\r\n<b id=\"y\"/>\r\n";
        let job = build_job(doc, &[], 0, 200.0, FontId::monospace(12.5), TEST_ACCENT);
        let mut prev = 0usize;
        for s in &job.sections {
            let start = s.byte_range.start.0;
            assert!(start >= prev, "перекрытие секций: {start} vs {prev}");
            prev = s.byte_range.end.0;
        }
        assert_eq!(prev, doc.len());
    }

    /// Детерминированный fuzz: случайные/искажённые входы не должны давать
    /// перекрытие токенов, выход за границы или панику — подсветка обязана
    /// деградировать к тексту, а не ломать LayoutJob.
    #[test]
    fn fuzz_malformed_inputs_never_overlap_or_panic() {
        let mut rng = XorShift::new(0x9E37_79B9_7F4A_7C15);
        let alphabet: &[u8] = b"<>/=\"'-! \t\nabcXYZ0123";
        for _ in 0..2_000 {
            let len = rng.next() as usize % 64;
            let mut s = String::with_capacity(len);
            for _ in 0..len {
                s.push(alphabet[rng.next() as usize % alphabet.len()] as char);
            }

            // Токены: без перекрытий и выходов за границу.
            let tokens = xml_tokens(&s);
            let mut prev = 0usize;
            for (r, _) in &tokens {
                assert!(r.start >= prev, "перекрытие токенов на `{s:?}`");
                assert!(r.end <= s.len(), "токен за границей на `{s:?}`");
                prev = r.end;
            }

            // LayoutJob: секции покрывают весь текст без перекрытий.
            let job = build_job(&s, &[], 0, 120.0, FontId::monospace(12.5), TEST_ACCENT);
            let mut prev = 0usize;
            for sec in &job.sections {
                let start = sec.byte_range.start.0;
                let end = sec.byte_range.end.0;
                assert!(start >= prev, "перекрытие секций на `{s:?}`");
                prev = end;
            }
            assert_eq!(prev, s.len(), "секции не покрывают весь текст `{s:?}`");
        }
    }

    /// Крошечный xorshift64 — детерминированный источник случайности для fuzz
    /// (без внешних зависимостей и без рантайм-зерна).
    struct XorShift(u64);

    impl XorShift {
        fn new(seed: u64) -> Self {
            Self(seed.max(1))
        }

        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }
}
