//! Raw C/CUDA FFI bindings.

use crate::context::GpuCtx;
use std::ffi::{c_float, c_int, c_void};

extern "C" {
    pub(crate) fn cuda_set_device(device: c_int) -> c_int;
    pub(crate) fn cuda_get_device_count(count: *mut c_int) -> c_int;

    pub(crate) fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    pub(crate) fn cudaMemset(ptr: *mut c_void, val: c_int, count: usize) -> c_int;
    pub(crate) fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;
    pub(crate) fn cudaFree(ptr: *mut c_void) -> c_int;

    pub(crate) fn gpu_context_create(
        n_neurons: c_int,
        n_synapses: c_int,
        n_decoders: c_int,
    ) -> *mut GpuCtx;
    pub(crate) fn gpu_context_destroy(ctx: *mut GpuCtx);
    pub(crate) fn gpu_context_sync(ctx: *mut GpuCtx) -> c_int;

    pub(crate) fn cuda_morris_lecar_step(
        v: *mut c_float,
        w: *mut c_float,
        I_in: *mut c_float,
        spikes: *mut c_int,
        params: *const c_float,
        dt: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_nmda_mg_block_step(
        v: *const c_float,
        g_act: *const c_float,
        i_out: *mut c_float,
        params: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_tsodyks_markram_step(
        x: *mut c_float,
        u: *mut c_float,
        spikes: *const c_int,
        w_eff: *mut c_float,
        params: *const c_float,
        dt: c_float,
        n: c_int,
    ) -> c_int;

    pub(crate) fn gpu_context_upload_neurons(
        ctx: *mut GpuCtx,
        v: *const c_float,
        u: *const c_float,
        a: *const c_float,
        b: *const c_float,
        c: *const c_float,
        d: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_context_upload_kuramoto(
        ctx: *mut GpuCtx,
        phases: *const c_float,
        omegas: *const c_float,
        coupling: *const c_float,
        n: c_int,
    ) -> c_int;

    pub(crate) fn gpu_izhikevich_step_async(
        ctx: *mut GpuCtx,
        dt: c_float,
        trace_decay: c_float,
    ) -> c_int;
    pub(crate) fn gpu_lif_step_async(ctx: *mut GpuCtx) -> c_int;
    pub(crate) fn gpu_context_upload_lif_params(
        ctx: *mut GpuCtx,
        dt: c_float,
        tau_m: *const c_float,
        v_rest: *const c_float,
        v_thresh: *const c_float,
        v_reset: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_stdp_update_async(
        ctx: *mut GpuCtx,
        lr_ltd: c_float,
        lr_ltp: c_float,
        w_min: c_float,
        w_max: c_float,
    ) -> c_int;
    pub(crate) fn gpu_kuramoto_prepare_async(ctx: *mut GpuCtx) -> c_int;
    pub(crate) fn gpu_kuramoto_update_async(ctx: *mut GpuCtx, dt: c_float) -> c_int;
    pub(crate) fn gpu_kuramoto_csr_coupling_async(ctx: *mut GpuCtx) -> c_int;
    pub(crate) fn gpu_bptt_loss_grad_async(
        ctx: *mut GpuCtx,
        d_phases_traj: *mut c_float,
        d_coupling_traj: *mut c_float,
        d_dtheta: *mut c_float,
        coherence_start: c_int,
        coherence_weight: c_float,
        temporal_scale: c_float,
        dt: c_float,
        n_steps: c_int,
    ) -> c_int;
    pub(crate) fn gpu_bptt_step_async(
        ctx: *mut GpuCtx,
        d_grad_weights: *mut c_float,
        d_dtheta: *mut c_float,
        d_dtheta_next: *const c_float,
        d_phases: *const c_float,
        dt: c_float,
        n_synapses: c_int,
    ) -> c_int;

    pub(crate) fn gpu_context_copy_phases_to_traj_async(
        ctx: *mut GpuCtx,
        d_traj: *mut c_float,
        step: c_int,
    ) -> c_int;
    pub(crate) fn gpu_context_copy_coupling_to_traj_async(
        ctx: *mut GpuCtx,
        d_traj: *mut c_float,
        step: c_int,
    ) -> c_int;

    pub(crate) fn gpu_zero_grad_weights_async(
        ctx: *mut GpuCtx,
        d_grad: *mut c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_weight_update_async(
        ctx: *mut GpuCtx,
        d_grad_weights: *const c_float,
        lr: c_float,
        sparsity_scale: c_float,
        homeostasis_scale: c_float,
        weight_clamp: c_float,
    ) -> c_int;
    pub(crate) fn gpu_grad_norm_async(
        ctx: *mut GpuCtx,
        d_grad: *const c_float,
        d_norm: *mut c_float,
        d_partial: *mut c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_grad_clip_async(
        ctx: *mut GpuCtx,
        d_grad: *mut c_float,
        d_norm: *const c_float,
        max_norm: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_bptt_epoch_async(
        ctx: *mut GpuCtx,
        d_phases_traj: *mut c_float,
        d_coupling_traj: *mut c_float,
        d_dtheta_traj: *mut c_float,
        d_grad_weights: *mut c_float,
        d_grad_norm: *mut c_float,
        d_grad_norm_partial: *mut c_float,
        n_steps: c_int,
        dt: c_float,
        coherence_start: c_int,
        coherence_weight: c_float,
        temporal_weight: c_float,
        grad_clip_norm: c_float,
    ) -> c_int;

    pub(crate) fn gpu_safety_clamp_async(
        ctx: *mut GpuCtx,
        min_val: c_float,
        max_val: c_float,
    ) -> c_int;

    pub(crate) fn gpu_fused_izhikevich_stdp_async(
        ctx: *mut GpuCtx,
        dt: c_float,
        trace_decay: c_float,
        lr_ltd: c_float,
        lr_ltp: c_float,
        w_min: c_float,
        w_max: c_float,
    ) -> c_int;

    pub(crate) fn gpu_context_upload_connectome(
        ctx: *mut GpuCtx,
        row_ptr: *const c_int,
        col_indices: *const c_int,
        values: *const c_float,
        nt_types: *const c_float,
        external_I: *const c_float,
        n_neurons: c_int,
        n_synapses: c_int,
    ) -> c_int;
    pub(crate) fn gpu_connectome_csr_async(ctx: *mut GpuCtx) -> c_int;
    pub(crate) fn gpu_connectome_csr_plastic_async(ctx: *mut GpuCtx) -> c_int;
    pub(crate) fn gpu_connectome_stdp_async(
        ctx: *mut GpuCtx,
        lr_ltp: c_float,
        lr_ltd: c_float,
        w_min: c_float,
        w_max: c_float,
    ) -> c_int;
    pub(crate) fn gpu_context_init_plastic_weights(ctx: *mut GpuCtx) -> c_int;
    pub(crate) fn gpu_count_spikes_async(ctx: *mut GpuCtx, d_total_spikes: *mut c_int) -> c_int;

    pub(crate) fn gpu_context_upload_I(ctx: *mut GpuCtx, I: *const c_float, n: c_int) -> c_int;
    pub(crate) fn gpu_context_download_spikes(
        ctx: *mut GpuCtx,
        spikes: *mut c_int,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_context_batch_lif_steps(
        ctx: *mut GpuCtx,
        external_I: *const c_float,
        n_external: c_int,
        dev_spike_buf: *mut c_int,
        host_spike_buf: *mut c_int,
        host_v: *mut c_float,
        n_steps: c_int,
        n_neurons: c_int,
    ) -> c_int;
    pub(crate) fn gpu_context_batch_izhikevich_steps(
        ctx: *mut GpuCtx,
        external_I: *const c_float,
        n_external: c_int,
        dt: c_float,
        trace_decay: c_float,
        dev_spike_buf: *mut c_int,
        host_spike_buf: *mut c_int,
        host_v: *mut c_float,
        host_u: *mut c_float,
        n_steps: c_int,
        n_neurons: c_int,
    ) -> c_int;
    pub(crate) fn gpu_context_batch_hh_steps(
        ctx: *mut GpuCtx,
        external_I: *const c_float,
        n_external: c_int,
        host_spike_buf: *mut c_int,
        host_v: *mut c_float,
        host_I_syn: *mut c_float,
        n_steps: c_int,
        n_neurons: c_int,
        dt: c_float,
        tau_ref: c_float,
        tau_syn: c_float,
        weight_scale: c_float,
    ) -> c_int;
    pub(crate) fn gpu_context_upload_v_only(ctx: *mut GpuCtx, v: *const c_float, n: c_int)
        -> c_int;
    pub(crate) fn gpu_context_download_v(ctx: *mut GpuCtx, v: *mut c_float, n: c_int) -> c_int;
    pub(crate) fn gpu_context_download_u(ctx: *mut GpuCtx, u: *mut c_float, n: c_int) -> c_int;
    pub(crate) fn gpu_context_download_phases(
        ctx: *mut GpuCtx,
        phases: *mut c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_context_download_adex_w(ctx: *mut GpuCtx, w: *mut c_float, n: c_int)
        -> c_int;
    pub(crate) fn gpu_context_upload_adex_w(ctx: *mut GpuCtx, w: *const c_float, n: c_int)
        -> c_int;
    pub(crate) fn gpu_context_download_weights(
        ctx: *mut GpuCtx,
        w: *mut c_float,
        n: c_int,
    ) -> c_int;

    pub(crate) fn gpu_context_ensure_all(ctx: *mut GpuCtx) -> c_int;
    pub(crate) fn gpu_context_get_d_noise(ctx: *mut GpuCtx) -> *mut c_void;
    pub(crate) fn gpu_graph_begin_capture(ctx: *mut GpuCtx) -> c_int;
    pub(crate) fn gpu_graph_end_capture(ctx: *mut GpuCtx) -> c_int;
    pub(crate) fn gpu_graph_replay(ctx: *mut GpuCtx) -> c_int;
    pub(crate) fn gpu_graph_is_captured(ctx: *mut GpuCtx) -> c_int;
    pub(crate) fn gpu_gco_graph_replay(ctx: *mut GpuCtx, dt: c_float, tau: c_float) -> c_int;
    pub(crate) fn gpu_gco_graph_invalidate(ctx: *mut GpuCtx) -> c_int;

    pub(crate) fn cuda_izhikevich_step(
        v: *mut c_float,
        u: *mut c_float,
        trace: *mut c_float,
        a: *const c_float,
        b: *const c_float,
        I: *const c_float,
        dt: c_float,
        trace_decay: c_float,
        v_thresh: c_float,
        spikes: *mut c_int,
        c: *const c_float,
        d: *const c_float,
        refractory: *mut c_float,
        tau_ref: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_lif_step(
        v: *mut c_float,
        I: *mut c_float,
        spikes: *mut c_int,
        params: *const c_float,
        refractory: *mut c_float,
        tau_ref: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_adex_step(
        v: *mut c_float,
        u: *mut c_float,
        I: *mut c_float,
        spikes: *mut c_int,
        params: *const c_float,
        refractory: *mut c_float,
        tau_ref: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_stdp_update(
        weights: *mut c_float,
        traces_pre: *const c_float,
        traces_post: *const c_float,
        lr_ltd: c_float,
        lr_ltp: c_float,
        w_min: c_float,
        w_max: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_saw_step(
        v: *mut c_float,
        u: *mut c_float,
        trace: *mut c_float,
        a: *const c_float,
        b: *const c_float,
        I: *mut c_float,
        dt: c_float,
        trace_decay: c_float,
        noise_amp: c_float,
        spikes: *mut c_int,
        refractory: *mut c_float,
        tau_ref: c_float,
        noise_state: *mut u32,
        v_thresh: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_gco_step(
        v: *mut c_float,
        u: *mut c_float,
        row_ptr: *const c_int,
        col_indices: *const c_int,
        values: *const c_float,
        i_ext: *const c_float,
        dt: c_float,
        tau: c_float,
        n_neurons: c_int,
    ) -> c_int;

    pub(crate) fn cuda_edge_predictor(
        d_states: *const c_float,
        d_output: *mut c_float,
        d_w1: *const c_float,
        d_b1: *const c_float,
        d_w2: *const c_float,
        d_b2: *const c_float,
        d_w3: *const c_float,
        d_b3: *const c_float,
        N: c_int,
        state_dim: c_int,
        hidden_dim: c_int,
    ) -> c_int;
    pub(crate) fn cuda_gmm_sample(
        d_weights: *const c_float,
        d_means: *const c_float,
        d_covars: *const c_float,
        d_cum_weights: *const c_float,
        d_states: *mut c_float,
        N: c_int,
        dim: c_int,
        n_components: c_int,
        seed: u32,
    ) -> c_int;
    pub(crate) fn cuda_topk_select(
        d_logits: *const c_float,
        d_target_degs: *const c_int,
        d_edge_counts: *mut c_int,
        d_scratch: *mut c_int,
        N: c_int,
        threshold: c_float,
        topk_mult: c_int,
        max_k: c_int,
    ) -> c_int;
    pub(crate) fn cuda_csr_build(
        d_edge_counts: *const c_int,
        d_row_ptr: *const c_int,
        d_scratch_indices: *const c_int,
        d_col_indices: *mut c_int,
        d_values: *mut c_float,
        max_k: c_int,
        N: c_int,
    ) -> c_int;
    pub(crate) fn cuda_hebbian_update(
        weights: *mut c_float,
        cos_diff: *const c_float,
        lr: c_float,
        decay: c_float,
        dt: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_kuramoto_prepare(
        phases: *const c_float,
        cos_out: *mut c_float,
        sin_out: *mut c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_kuramoto_update(
        phases: *mut c_float,
        omegas: *const c_float,
        coupling: *const c_float,
        dt: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_kuramoto_csr_coupling(
        coupling: *mut c_float,
        sin_buf: *const c_float,
        cos_buf: *const c_float,
        row_ptr: *const c_int,
        col_indices: *const c_int,
        values: *const c_float,
        n_neurons: c_int,
    ) -> c_int;
    pub(crate) fn cuda_upload_csr_values(
        ctx: *mut GpuCtx,
        values: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_download_csr_values(
        ctx: *mut GpuCtx,
        values: *mut c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_kuramoto_bptt_loss_grad(
        phases_traj: *const c_float,
        coupling_traj: *const c_float,
        omegas: *const c_float,
        dtheta: *mut c_float,
        coherence_start: c_int,
        coherence_weight: c_float,
        temporal_scale: c_float,
        dt: c_float,
        n_neurons: c_int,
        n_steps: c_int,
    ) -> c_int;
    pub(crate) fn cuda_kuramoto_bptt_step(
        grad_weights: *mut c_float,
        dtheta: *mut c_float,
        dtheta_next: *const c_float,
        phases: *const c_float,
        row_ptr: *const c_int,
        col_indices: *const c_int,
        values: *const c_float,
        dt: c_float,
        n_neurons: c_int,
        n_synapses: c_int,
    ) -> c_int;
    pub(crate) fn cuda_safety_clamp(
        data: *mut c_float,
        min_val: c_float,
        max_val: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_stochastic_encoder(
        target_I: *mut c_float,
        inputs: *const c_float,
        noise: *const c_float,
        threshold: c_float,
        spike_val: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_thalamic_gate(
        target_I: *mut c_float,
        source_v: *const c_float,
        gate_v: *const c_float,
        threshold: c_float,
        gain: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_sigmoid_gate(
        output: *mut c_float,
        input: *const c_float,
        gain: c_float,
        threshold: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_phase_gate_apply(
        output: *mut c_float,
        input: *const c_float,
        r: *const c_float,
        gain: c_float,
        threshold: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_t3_drive_encoder(
        drives: *mut c_float,
        inputs: *const c_float,
        gain: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_t3_inverse_kuramoto(
        drive_out: *mut c_float,
        r: *const c_float,
        r_prev: *const c_float,
        alpha: c_float,
        gain: c_float,
        eps: c_float,
        decay: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_nlms_train(
        weights: *mut c_float,
        bias: *mut c_float,
        states: *const c_float,
        error: c_float,
        lr: c_float,
        eps: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_kalman_update(
        weights: *mut c_float,
        cov: *mut c_float,
        states: *const c_float,
        error: c_float,
        s: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_echo_tap(
        output: *mut c_float,
        buffer: *const c_float,
        n_osc: c_int,
        delay_len: c_int,
        delay_idx: c_int,
    ) -> c_int;
    pub(crate) fn cuda_ring_synthesis(
        output: *mut c_float,
        amplitudes: *const c_float,
        trig_buf: *const c_float,
        weights: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_interference(
        output: *mut c_float,
        phases_a: *const c_float,
        phases_b: *const c_float,
        na: c_int,
        nb: c_int,
    ) -> c_int;
    pub(crate) fn cuda_interference_mean(
        output: *mut c_float,
        phases_a: *const c_float,
        phases_b: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_phase_coherence(
        output: *mut c_float,
        sin_buf: *const c_float,
        cos_buf: *const c_float,
        amplitudes: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_band_energy(
        band_energies: *mut c_float,
        amplitudes: *const c_float,
        omegas: *const c_float,
        band_edges: *const c_float,
        n_bands: c_int,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_winding_matrix(
        output: *mut c_float,
        phases_a: *const c_float,
        phases_b: *const c_float,
        na: c_int,
        nb: c_int,
    ) -> c_int;
    pub(crate) fn cuda_lazy_sincos_update(
        sin_out: *mut c_float,
        cos_out: *mut c_float,
        phases: *const c_float,
        phases_prev: *mut c_float,
        eps: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_batch_coherence_sweep(
        output: *mut c_float,
        areas: *const c_float,
        areas_sin: *const c_float,
        areas_cos: *const c_float,
        n_areas: c_int,
        n_elem: c_int,
    ) -> c_int;
    pub(crate) fn cuda_t3_tact_generator(
        pulses: *mut c_float,
        ph1: *const c_float,
        ph2: *const c_float,
        ph3: *const c_float,
        threshold: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_hh_step(
        v: *mut c_float,
        gates: *mut c_float,
        Ii: *const c_float,
        spikes: *mut c_int,
        refractory: *mut c_float,
        g_leak: *const c_float,
        E_leak: *const c_float,
        C_m: *const c_float,
        n: c_int,
        n_gates: c_int,
        dt: c_float,
        tau_ref: c_float,
        E_rev_arr: *const c_float,
        g_max_arr: *const c_float,
        powers: *const u32,
        n_gates_per_ch: *const c_int,
        n_channels: c_int,
    ) -> c_int;
    pub(crate) fn cuda_t3_update(
        phases: *mut c_float,
        amplitudes: *mut c_float,
        omegas: *const c_float,
        coupling: *const c_float,
        drives: *const c_float,
        dt: c_float,
        alpha: c_float,
        gain: c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn cuda_t3_prepare(
        phases: *mut c_float,
        cos_out: *mut c_float,
        sin_out: *mut c_float,
        n: c_int,
    ) -> c_int;

    pub(crate) fn gpu_t3_prepare_async(ctx: *mut GpuCtx) -> c_int;
    pub(crate) fn gpu_t3_update_async(
        ctx: *mut GpuCtx,
        dt: c_float,
        alpha: c_float,
        gain: c_float,
    ) -> c_int;
    pub(crate) fn gpu_t3_stuart_landau_async(
        ctx: *mut GpuCtx,
        dt: c_float,
        alpha: c_float,
        gain: c_float,
    ) -> c_int;
    pub(crate) fn gpu_t3_upload(
        ctx: *mut GpuCtx,
        phases: *const c_float,
        amplitudes: *const c_float,
        omegas: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_t3_upload_drives(ctx: *mut GpuCtx, drives: *const c_float, n: c_int)
        -> c_int;
    pub(crate) fn gpu_t3_upload_coupling(
        ctx: *mut GpuCtx,
        coupling: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_t3_download(
        ctx: *mut GpuCtx,
        phases: *mut c_float,
        amplitudes: *mut c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_t3_download_cos_sin(
        ctx: *mut GpuCtx,
        cos_out: *mut c_float,
        sin_out: *mut c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_t3_upload_omegas(ctx: *mut GpuCtx, omegas: *const c_float, n: c_int)
        -> c_int;
    pub(crate) fn gpu_t3_upload_async(
        ctx: *mut GpuCtx,
        phases: *const c_float,
        amplitudes: *const c_float,
        omegas: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_t3_upload_drives_async(
        ctx: *mut GpuCtx,
        drives: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_t3_upload_coupling_async(
        ctx: *mut GpuCtx,
        coupling: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_t3_upload_omegas_async(
        ctx: *mut GpuCtx,
        omegas: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_t3_download_async(
        ctx: *mut GpuCtx,
        phases: *mut c_float,
        amplitudes: *mut c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_t3_download_cos_sin_async(
        ctx: *mut GpuCtx,
        cos_out: *mut c_float,
        sin_out: *mut c_float,
        n: c_int,
    ) -> c_int;

    pub(crate) fn gpu_gco_hamiltonian_async(ctx: *mut GpuCtx) -> c_int;
    pub(crate) fn gpu_gco_step_async(ctx: *mut GpuCtx, dt: c_float, tau: c_float) -> c_int;
    pub(crate) fn gpu_gco_combined_step_async(ctx: *mut GpuCtx, dt: c_float, tau: c_float)
        -> c_int;
    pub(crate) fn gpu_gco_upload_async(
        ctx: *mut GpuCtx,
        host_v: *const c_float,
        host_u: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_gco_download_async(
        ctx: *mut GpuCtx,
        host_v: *mut c_float,
        host_u: *mut c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_gco_upload_drive_async(
        ctx: *mut GpuCtx,
        host_i_ext: *const c_float,
        n: c_int,
    ) -> c_int;

    pub(crate) fn cuda_csr_spmv_tanh(
        out: *mut c_float,
        in_state: *const c_float,
        row_ptr: *const c_int,
        col_indices: *const c_int,
        values: *const c_float,
        input_vec: *const c_float,
        input_scale: c_float,
        n: c_int,
        stream: *mut c_void,
    ) -> c_int;

    pub(crate) fn gpu_hh_step_async(ctx: *mut GpuCtx, dt: c_float, tau_ref: c_float) -> c_int;
    pub(crate) fn gpu_hh_upload_state(
        ctx: *mut GpuCtx,
        v: *const c_float,
        gates: *const c_float,
        Ii: *const c_float,
        g_leak: *const c_float,
        E_leak: *const c_float,
        C_m: *const c_float,
        n: c_int,
        n_gates: c_int,
    ) -> c_int;
    pub(crate) fn gpu_hh_upload_channels(
        ctx: *mut GpuCtx,
        E_rev_arr: *const c_float,
        g_max_arr: *const c_float,
        powers: *const u32,
        n_gates_per_ch: *const c_int,
        n_gates: c_int,
        n_channels: c_int,
    ) -> c_int;

    pub(crate) fn gpu_aber_upload_async(
        ctx: *mut GpuCtx,
        host_real: *const c_float,
        host_imag: *const c_float,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_aber_download_async(
        ctx: *mut GpuCtx,
        host_real: *mut c_float,
        host_imag: *mut c_float,
        host_collapsed: *mut c_int,
        n: c_int,
    ) -> c_int;
    pub(crate) fn gpu_aber_step_async(
        ctx: *mut GpuCtx,
        gamma: c_float,
        omega: c_float,
        gain: c_float,
        threshold: c_float,
        collapse_scale: c_float,
        dt: c_float,
        use_synaptic_field: c_int,
    ) -> c_int;
    pub(crate) fn gpu_gco_aberton_combined_async(
        ctx: *mut GpuCtx,
        dt: c_float,
        tau: c_float,
        aber_gamma: c_float,
        aber_omega: c_float,
        aber_gain: c_float,
        aber_threshold: c_float,
        aber_collapse_scale: c_float,
        use_synaptic_field: c_int,
    ) -> c_int;
    pub(crate) fn gpu_lif_at_step(ctx: *mut GpuCtx, host_params: *const c_float, n: c_int)
        -> c_int;
    pub(crate) fn gpu_intrinsic_plasticity(
        ctx: *mut GpuCtx,
        target_rate: c_float,
        eta: c_float,
        decay: c_float,
    ) -> c_int;
}
