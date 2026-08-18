//! Встроенная консоль («комбайн») — команды CLI прямо в GUI.
//!
//! Здесь живёт только состояние консоли и UI; логика команд — в
//! `crate::app::ConnApp::console_exec` (ей нужен доступ к приватным
//! полям приложения). Строки вывода разбираются на кликабельные
//! `#индексы` нейронов (`CLine::parse`).

/// Строка вывода консоли; `neuron` = кликабельный индекс (если есть).
pub struct CLine {
    pub text: String,
    pub neuron: Option<usize>,
}

impl CLine {
    /// Ищет в строке первое вхождение `#<digits>` и помечает его как нейрон.
    pub fn parse(text: &str) -> CLine {
        let bytes = text.as_bytes();
        let mut neuron = None;
        let mut i = 0;
        while i + 1 < bytes.len() {
            if bytes[i] == b'#' && bytes[i + 1].is_ascii_digit() {
                let mut j = i + 1;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                if let Ok(v) = text[i + 1..j].parse::<usize>() {
                    neuron = Some(v);
                    break;
                }
                i = j;
            } else {
                i += 1;
            }
        }
        CLine {
            text: text.to_string(),
            neuron,
        }
    }
}

/// Состояние консоли: ввод, история, вывод (кольцевой буфер).
pub struct Console {
    pub input: String,
    pub history: Vec<String>,
    pub hist_idx: Option<usize>,
    pub lines: Vec<CLine>,
    pub max_lines: usize,
}

impl Console {
    pub fn new() -> Self {
        let mut c = Console {
            input: String::new(),
            history: Vec::new(),
            hist_idx: None,
            lines: Vec::new(),
            max_lines: 600,
        };
        c.push_text("neoMyelin-консоль — комбайн прямо в GUI. Введи `help` для списка команд.");
        c.push_text("Кликабельные #индексы выбирают нейрон и центрируют камеру.");
        c
    }

    /// Добавляет строку введённой команды (подсветка «❯»).
    pub fn push_command(&mut self, line: &str) {
        self.lines.push(CLine {
            text: format!("❯ {line}"),
            neuron: None,
        });
        self.trim();
    }

    /// Добавляет многострочный текст (каждая строка — отдельная `CLine`).
    pub fn push_text(&mut self, text: &str) {
        for l in text.lines() {
            if l.trim().is_empty() {
                continue;
            }
            self.lines.push(CLine::parse(l));
        }
        self.trim();
    }

    fn trim(&mut self) {
        if self.lines.len() > self.max_lines {
            let drop = self.lines.len() - self.max_lines;
            self.lines.drain(0..drop);
        }
    }

    pub fn hist_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let idx = match self.hist_idx {
            None => self.history.len() - 1,
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.hist_idx = Some(idx);
        self.input = self.history[idx].clone();
    }

    pub fn hist_next(&mut self) {
        match self.hist_idx {
            Some(i) if i + 1 < self.history.len() => {
                self.hist_idx = Some(i + 1);
                self.input = self.history[i + 1].clone();
            }
            Some(_) => {
                self.hist_idx = None;
                self.input.clear();
            }
            None => {}
        }
    }
}
