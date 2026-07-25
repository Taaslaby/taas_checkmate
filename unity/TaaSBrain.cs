using System;
using System.Runtime.InteropServices;

/// <summary>
/// Unity C# wrapper for TaaS neural engine (taas_unity).
///
/// Usage:
/// <code>
/// // Create a 100K LIF brain with a simple ring connectome
/// var brain = new TaaSBrain(100000, "lif");
/// var conn = TaaSConnectome.LoadRing(100000, 1.0f);
///
/// // Each frame:
/// int spikes = brain.Step(conn, externalI, 0.5f);
///
/// // Read state
/// float[] v = brain.GetV();
/// int[] spikes = brain.GetSpikes();
///
/// // Cleanup
/// conn.Dispose();
/// brain.Dispose();
/// </code>
/// </summary>
public unsafe class TaaSBrain : IDisposable
{
    private IntPtr _handle;
    private int _nNeurons;
    private bool _disposed;

    // ═════════════════════════════════════════════════════
    //  P/Invoke declarations
    // ═════════════════════════════════════════════════════

    #region DllImports

    // Lifecycle
    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr brain_new(int n_neurons,
        [MarshalAs(UnmanagedType.LPStr)] string model);

    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern void brain_free(IntPtr brain);

    // Connectome
    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr connectome_load(
        int[] rowPtr, int nNeurons,
        int[] colIndices, float[] values,
        float[] ntTypes, int nSynapses);

    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern void connectome_free(IntPtr conn);

    // Step
    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern int brain_step(IntPtr brain, IntPtr conn,
        float[] externalI, float dtMs);

    // State readers
    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern int brain_get_v(IntPtr brain, float[] outBuf, int maxN);

    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern int brain_get_u(IntPtr brain, float[] outBuf, int maxN);

    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern int brain_get_spikes(IntPtr brain, int[] outBuf, int maxN);

    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern int brain_get_I_buf(IntPtr brain, float[] outBuf, int maxN);

    // Param setters
    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern int brain_set_lif_params(IntPtr brain,
        float[] tauM, float[] vRest, float[] vThresh, float[] vReset,
        int n, float tauRef, float dtMs);

    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern int brain_set_izhikevich_params(IntPtr brain,
        float[] a, float[] b, float[] c, float[] d,
        int n, float tauRef, float dtMs);

    // Reset
    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern void brain_reset(IntPtr brain);

    // Info
    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern int brain_n_neurons(IntPtr brain);

    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern int connectome_n_neurons(IntPtr conn);

    [DllImport("taas_checkmate", CallingConvention = CallingConvention.Cdecl)]
    private static extern int connectome_n_synapses(IntPtr conn);

    #endregion

    // ═════════════════════════════════════════════════════
    //  Brain lifecycle
    // ═════════════════════════════════════════════════════

    /// <summary>
    /// Create a brain simulator.
    /// </summary>
    /// <param name="nNeurons">Number of neurons.</param>
    /// <param name="model">"lif" or "izhikevich"</param>
    public TaaSBrain(int nNeurons, string model)
    {
        _handle = brain_new(nNeurons, model);
        if (_handle == IntPtr.Zero)
            throw new InvalidOperationException(
                $"brain_new failed: nNeurons={nNeurons}, model={model}");

        _nNeurons = brain_n_neurons(_handle);
        if (_nNeurons <= 0)
            throw new InvalidOperationException("brain_n_neurons returned invalid count");
    }

    /// <summary>
    /// Number of neurons in this brain.
    /// </summary>
    public int NeuronCount => _nNeurons;

    /// <summary>
    /// True if the native handle has been freed.
    /// </summary>
    public bool IsDisposed => _disposed;

    /// <summary>
    /// Run one simulation step.
    /// </summary>
    /// <param name="conn">Connectome graph.</param>
    /// <param name="externalI">External input current per neuron (can be null for zeros).</param>
    /// <param name="dtMs">Timestep in milliseconds.</param>
    /// <returns>Number of spikes fired this step.</returns>
    public int Step(TaaSConnectome conn, float[] externalI, float dtMs)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(conn);

        return brain_step(_handle, conn.Handle, externalI, dtMs);
    }

    /// <summary>
    /// Run one simulation step with zero external input.
    /// </summary>
    public int Step(TaaSConnectome conn, float dtMs)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(conn);
        return brain_step(_handle, conn.Handle, null, dtMs);
    }

    // ═════════════════════════════════════════════════════
    //  State accessors
    // ═════════════════════════════════════════════════════

    /// <summary>
    /// Get membrane potentials (v) for all neurons.
    /// </summary>
    public float[] GetV()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var buf = new float[_nNeurons];
        int n = brain_get_v(_handle, buf, _nNeurons);
        if (n > 0 && n < _nNeurons)
            Array.Resize(ref buf, n);
        return buf;
    }

    /// <summary>
    /// Get recovery variables (u) — only meaningful for Izhikevich.
    /// </summary>
    public float[] GetU()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var buf = new float[_nNeurons];
        int n = brain_get_u(_handle, buf, _nNeurons);
        if (n > 0 && n < _nNeurons)
            Array.Resize(ref buf, n);
        return buf;
    }

    /// <summary>
    /// Get spike flags (0/1) for the current timestep.
    /// </summary>
    public int[] GetSpikes()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var buf = new int[_nNeurons];
        int n = brain_get_spikes(_handle, buf, _nNeurons);
        if (n > 0 && n < _nNeurons)
            Array.Resize(ref buf, n);
        return buf;
    }

    /// <summary>
    /// Get input current buffer (external + synaptic).
    /// </summary>
    public float[] GetIBuf()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var buf = new float[_nNeurons];
        int n = brain_get_I_buf(_handle, buf, _nNeurons);
        if (n > 0 && n < _nNeurons)
            Array.Resize(ref buf, n);
        return buf;
    }

    // ═════════════════════════════════════════════════════
    //  Parameter overrides
    // ═════════════════════════════════════════════════════

    /// <summary>
    /// Override LIF parameters.
    /// Pass null for any array to keep current values.
    /// </summary>
    public void SetLifParams(float[] tauM = null, float[] vRest = null,
        float[] vThresh = null, float[] vReset = null,
        float tauRef = -1, float dtMs = -1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int ret = brain_set_lif_params(
            _handle, tauM, vRest, vThresh, vReset,
            tauM?.Length ?? vRest?.Length ?? vThresh?.Length ?? vReset?.Length ?? 0,
            tauRef, dtMs);
        if (ret < 0)
            throw new InvalidOperationException($"brain_set_lif_params failed: error {ret}");
    }

    /// <summary>
    /// Override Izhikevich parameters.
    /// Pass null for any array to keep current values.
    /// </summary>
    public void SetIzhikevichParams(float[] a = null, float[] b = null,
        float[] c = null, float[] d = null,
        float tauRef = -1, float dtMs = -1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int ret = brain_set_izhikevich_params(
            _handle, a, b, c, d,
            a?.Length ?? b?.Length ?? c?.Length ?? d?.Length ?? 0,
            tauRef, dtMs);
        if (ret < 0)
            throw new InvalidOperationException($"brain_set_izhikevich_params failed: error {ret}");
    }

    // ═════════════════════════════════════════════════════
    //  Reset
    // ═════════════════════════════════════════════════════

    /// <summary>
    /// Reset brain state to initial values.
    /// </summary>
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        brain_reset(_handle);
    }

    // ═════════════════════════════════════════════════════
    //  IDisposable
    // ═════════════════════════════════════════════════════

    public void Dispose()
    {
        if (!_disposed && _handle != IntPtr.Zero)
        {
            brain_free(_handle);
            _handle = IntPtr.Zero;
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~TaaSBrain() => Dispose();
}

/// <summary>
/// Connectome graph for TaaS brain simulation.
/// Created from CSR arrays and passed to TaaSBrain.Step().
/// </summary>
public unsafe class TaaSConnectome : IDisposable
{
    private IntPtr _handle;
    private int _nNeurons;
    private int _nSynapses;
    private bool _disposed;

    /// <summary>
    /// Load a connectome from CSR arrays.
    /// </summary>
    /// <param name="rowPtr">CSR row pointer [nNeurons + 1].</param>
    /// <param name="colIndices">CSR column indices [nSynapses].</param>
    /// <param name="values">Synaptic weights [nSynapses].</param>
    /// <param name="ntTypes">Neurotransmitter type: +1 excitatory, -1 inhibitory [nNeurons].</param>
    public TaaSConnectome(int[] rowPtr, int[] colIndices,
        float[] values, float[] ntTypes)
    {
        ArgumentNullException.ThrowIfNull(rowPtr);
        ArgumentNullException.ThrowIfNull(colIndices);
        ArgumentNullException.ThrowIfNull(values);

        int nNeurons = rowPtr.Length - 1;
        int nSynapses = values.Length;

        if (nNeurons <= 0)
            throw new ArgumentException("rowPtr must have at least 2 elements", nameof(rowPtr));

        _handle = connectome_load(rowPtr, nNeurons, colIndices, values, ntTypes, nSynapses);
        if (_handle == IntPtr.Zero)
            throw new InvalidOperationException("connectome_load failed");

        _nNeurons = nNeurons;
        _nSynapses = nSynapses;
    }

    /// <summary>
    /// Create a simple ring connectome: each neuron connects to the next.
    /// </summary>
    /// <param name="nNeurons">Number of neurons.</param>
    /// <param name="weight">Synaptic weight (default 1.0).</param>
    /// <returns>A ring connectome.</returns>
    public static TaaSConnectome LoadRing(int nNeurons, float weight = 1.0f)
    {
        var rp = new int[nNeurons + 1];
        var ci = new int[nNeurons];
        var vals = new float[nNeurons];
        var nt = new float[nNeurons];

        for (int i = 0; i < nNeurons; i++)
        {
            ci[i] = (i + 1) % nNeurons;
            vals[i] = weight;
            nt[i] = 1.0f;
            rp[i + 1] = i + 1;
        }

        return new TaaSConnectome(rp, ci, vals, nt);
    }

    /// <summary>
    /// Native handle for P/Invoke calls.
    /// </summary>
    internal IntPtr Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _handle;
        }
    }

    /// <summary>
    /// Number of neurons in the connectome.
    /// </summary>
    public int NeuronCount => _nNeurons;

    /// <summary>
    /// Number of synapses in the connectome.
    /// </summary>
    public int SynapseCount => _nSynapses;

    public void Dispose()
    {
        if (!_disposed && _handle != IntPtr.Zero)
        {
            connectome_free(_handle);
            _handle = IntPtr.Zero;
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~TaaSConnectome() => Dispose();
}
