//! ⚡ Routing Pillar: тензорная шина обмена данными между нейронами и синапсами.
//!
//! Реализует «Чистую шину» — все данные текут через плоские массивы индексов,
//! без ООП-ссылок. Gather (нейроны→синапсы) и Scatter (синапсы→нейроны).

/// ⚡ TensorBridge: Routing Pillar — бесветвящаяся шина данных между слоями.
pub struct TensorBridge {
    /// Входящие пре-синаптические спайки (для синапсов)
    pub i_pre_bus: Vec<f32>,
    /// Входящие пост-синаптические спайки (для CaPlasticity)
    pub i_post_bus: Vec<f32>,
    /// Агрегированный ток инжекции (для нейронов)
    pub i_inj_bus: Vec<f32>,

    /// Индекс pre-нейрона для каждого синапса
    pub synapse_pre_indices: Vec<usize>,
    /// Индекс post-нейрона для каждого синапса
    pub synapse_post_indices: Vec<usize>,
}

impl TensorBridge {
    pub fn new(neuron_count: usize, synapse_count: usize) -> Self {
        Self {
            i_pre_bus: vec![0.0; synapse_count],
            i_post_bus: vec![0.0; synapse_count],
            i_inj_bus: vec![0.0; neuron_count],
            synapse_pre_indices: vec![0; synapse_count],
            synapse_post_indices: vec![0; synapse_count],
        }
    }

    /// Gather: нейроны → синапсы.
    /// Копирует спайки/суррогаты нейронов на шины синапсов по индексам.
    pub fn route_neurons_to_synapses(&mut self, neuron_surrogates: &[f32], neuron_spikes: &[f32]) {
        for (syn_idx, &pre_idx) in self.synapse_pre_indices.iter().enumerate() {
            self.i_pre_bus[syn_idx] = neuron_spikes[pre_idx];
        }
        for (syn_idx, &post_idx) in self.synapse_post_indices.iter().enumerate() {
            self.i_post_bus[syn_idx] = neuron_surrogates[post_idx];
        }
    }

    /// Scatter: синапсы → нейроны.
    /// Суммирует синаптические токи в шину инжекции нейронов.
    pub fn accumulate_synaptic_currents(&mut self, synapse_currents: &[f32]) {
        self.i_inj_bus.fill(0.0);
        for (syn_idx, &post_idx) in self.synapse_post_indices.iter().enumerate() {
            self.i_inj_bus[post_idx] += synapse_currents[syn_idx];
        }
    }
}
