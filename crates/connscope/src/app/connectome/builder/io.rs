//! Сериализация графа — JSON save/load.

use serde::{Deserialize, Serialize};

/// Файл графа — версия + ноды + рёбра.
#[derive(Serialize, Deserialize)]
pub(crate) struct FileGraph {
    pub version: u32,
    pub nodes: Vec<FileNode>,
    pub edges: Vec<FileEdge>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct FileNode {
    pub kind: u8,
    pub x: f32,
    pub y: f32,
    pub out_reg: usize,
    pub c: f32,
    pub c2: f32,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct FileEdge {
    pub from: usize,
    pub from_pin: usize,
    pub to: usize,
    pub to_pin: usize,
}
