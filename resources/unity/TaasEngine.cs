// ═══════════════════════════════════════════════════════════════════════
// TaasEngine.cs — C# P/Invoke bridge for the TaaS neural engine (Unity).
//
// Drop-in wrapper over the `taas_checkmate` C ABI (src/unity.rs). Handles
// the full lifecycle: brain + connectome creation, stepping, state read/write,
// parameter overrides and reset. Uses GCHandle pinning for zero-copy array
// marshaling (avoids `float[]` → `IntPtr` copies and GC churn every frame).
//
// REQUIREMENTS
//  1. Build the native library once:
//     cargo build --release --features unity
//     cp target/release/taas_checkmate.dll <UnityProject>/Assets/Plugins/
//  2. macOS/Linux: rename to libtaas_checkmate.dylib / .so accordingly.
//  3. For IL2CPP: the library is pure C ABI (no managed deps) — safe.
//
// USAGE (minimal)
//   var engine = new TaasEngine(1000, "lif");
//   engine.LoadConnectome(rowPtr, colIdx, weights, ntTypes);
//   for (int t = 0; t < 1000; t++) {
//       int spikes = engine.Step(inputCurrents, dtMs: 0.5f);
//       engine.CopySpikes(spikeBuf);
//   }
//   engine.Dispose();
//
// Threading: all methods marshal to the native side; the engine is NOT
// thread-safe (Unity main thread only — Job System would need a mutex).
// ═══════════════════════════════════════════════════════════════════════

using System;
using System.Runtime.InteropServices;

namespace Taas
{
    /// <summary>Step-pipeline flags — mirror STEP_* in src/unity.rs.</summary>
    [Flags]
    public enum StepFlags
    {
        SetInput    = 1 << 0, // copy external I into I_buf
        Neuron      = 1 << 1, // one SIMD neuron step
        Push        = 1 << 2, // push spikes into the CSR queue
        Propagate   = 1 << 3, // distribute currents to post-synaptic neurons
        Full        = 1 | 2 | 4 | 8,
    }

    /// <summary>Neuron models accepted by brain_new.</summary>
    public enum BrainModel
    {
        Lif, Izhikevich, Rate, Gco, Hh,
    }

    /// <summary>
    /// Managed wrapper over the TaaS C ABI. One instance owns one brain and
    /// optionally one connectome; both are freed on Dispose.
    /// </summary>
    public sealed class TaasEngine : IDisposable
    {
        // ── Native entry points (must match src/unity.rs exactly) ─────────
        [DllImport("taas_checkmate")]
        private static extern IntPtr brain_new(int n_neurons, [MarshalAs(UnmanagedType.LPStr)] string model);

        [DllImport("taas_checkmate")]
        private static extern void brain_free(IntPtr brain);

        [DllImport("taas_checkmate")]
        private static extern IntPtr connectome_load(IntPtr row_ptr, int n_neurons,
            IntPtr col_indices, IntPtr values, IntPtr nt_types, int n_synapses);

        [DllImport("taas_checkmate")]
        private static extern void connectome_free(IntPtr conn);

        [DllImport("taas_checkmate")]
        private static extern int brain_step(IntPtr brain, IntPtr conn, IntPtr external_I, float dt_ms);

        [DllImport("taas_checkmate")]
        private static extern int brain_step_ex(IntPtr brain, IntPtr conn, IntPtr external_I,
            float dt_ms, int flags);

        [DllImport("taas_checkmate")]
        private static extern int brain_step_n(IntPtr brain, IntPtr conn, IntPtr external_I,
            float dt_ms, int n_steps);

        [DllImport("taas_checkmate")]
        private static extern int brain_get_v(IntPtr brain, IntPtr out_buf, int max_n);

        [DllImport("taas_checkmate")]
        private static extern int brain_get_u(IntPtr brain, IntPtr out_buf, int max_n);

        [DllImport("taas_checkmate")]
        private static extern int brain_get_spikes(IntPtr brain, IntPtr out_buf, int max_n);

        [DllImport("taas_checkmate")]
        private static extern int brain_get_I_buf(IntPtr brain, IntPtr out_buf, int max_n);

        [DllImport("taas_checkmate")]
        private static extern int brain_set_lif_params(IntPtr brain, IntPtr tau_m, IntPtr v_rest,
            IntPtr v_thresh, IntPtr v_reset, int n, float tau_ref, float dt_ms);

        [DllImport("taas_checkmate")]
        private static extern int brain_set_izhikevich_params(IntPtr brain, IntPtr a, IntPtr b,
            IntPtr c, IntPtr d, int n, float tau_ref, float dt_ms);

        [DllImport("taas_checkmate")]
        private static extern int brain_write_v(IntPtr brain, IntPtr src, int n);

        [DllImport("taas_checkmate")]
        private static extern int brain_write_u(IntPtr brain, IntPtr src, int n);

        [DllImport("taas_checkmate")]
        private static extern int brain_write_I_buf(IntPtr brain, IntPtr src, int n);

        [DllImport("taas_checkmate")]
        private static extern int brain_inject_current(IntPtr brain, int neuron_idx, float current);

        [DllImport("taas_checkmate")]
        private static extern void brain_reset(IntPtr brain);

        [DllImport("taas_checkmate")]
        private static extern int brain_n_neurons(IntPtr brain);

        [DllImport("taas_checkmate")]
        private static extern int connectome_n_neurons(IntPtr conn);

        [DllImport("taas_checkmate")]
        private static extern int connectome_n_synapses(IntPtr conn);

        // ── Native error codes ────────────────────────────────────────────
        private const int OK = 0;
        private const int ERR_MAGIC = -2;
        private const int ERR_MODEL = -3;
        private const int ERR_PARAM = -5;

        private IntPtr _brain;
        private IntPtr _conn;
        private readonly int _n;

        private GCHandle _rpPin, _ciPin, _valPin, _ntPin;   // connectome pins
        private readonly float[] _iBuf;                     // external input scratch
        private readonly float[] _fScratch;                 // float state scratch
        private readonly int[]   _iScratch;                 // spike scratch
        private bool _disposed;

        public int NeuronCount => _n;
        public bool HasConnectome => _conn != IntPtr.Zero;

        /// <summary>Create a brain with `n` neurons of the given model.</summary>
        public TaasEngine(int n, BrainModel model)
        {
            if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
            _n = n;
            _brain = brain_new(n, ModelName(model));
            if (_brain == IntPtr.Zero)
                throw new InvalidOperationException($"brain_new failed for model {model}");

            _iBuf = new float[n];
            _fScratch = new float[n];
            _iScratch = new int[n];
        }

        private static string ModelName(BrainModel m) => m switch
        {
            BrainModel.Lif        => "lif",
            BrainModel.Izhikevich => "izhikevich",
            BrainModel.Rate       => "rate",
            BrainModel.Gco        => "gco",
            BrainModel.Hh         => "hh",
            _ => throw new ArgumentOutOfRangeException(nameof(m)),
        };

        // ── Connectome ────────────────────────────────────────────────────

        /// <summary>
        /// Load a CSR connectome. Ownership transfers to the engine (freed on
        /// Dispose). Passing an empty graph (n_synapses == 0) is allowed.
        /// </summary>
        public void LoadConnectome(int[] rowPtr, int[] colIndices, float[] values,
            float[] ntTypes = null)
        {
            EnsureNotDisposed();
            if (rowPtr == null || rowPtr.Length != _n + 1)
                throw new ArgumentException("rowPtr must have n+1 entries");

            FreeConnectome();

            ntTypes ??= new float[_n];
            _rpPin  = GCHandle.Alloc(rowPtr,  GCHandleType.Pinned);
            _ciPin  = GCHandle.Alloc(colIndices, GCHandleType.Pinned);
            _valPin = GCHandle.Alloc(values,   GCHandleType.Pinned);
            _ntPin  = GCHandle.Alloc(ntTypes,  GCHandleType.Pinned);

            _conn = connectome_load(
                Ptr(_rpPin), _n,
                Ptr(_ciPin), Ptr(_valPin),
                Ptr(_ntPin), colIndices.Length);

            if (_conn == IntPtr.Zero)
            {
                // Native rejected the CSR arrays — release the pins we just took.
                FreeConnectome();
                throw new InvalidOperationException("connectome_load rejected the CSR arrays");
            }
        }

        /// <summary>Load an isolated population (no synapses).</summary>
        public void LoadEmptyConnectome()
        {
            var rowPtr = new int[_n + 1]; // all zeros → no edges
            LoadConnectome(rowPtr, new int[0], new float[0]);
        }

        // ── Stepping ──────────────────────────────────────────────────────

        /// <summary>Full pipeline: input → neuron → push → propagate. Returns spike count.</summary>
        public int Step(float[] externalI, float dtMs)
        {
            EnsureNotDisposed();
            if (_conn == IntPtr.Zero)
                throw new InvalidOperationException("Step requires a connectome; use StepNeuronsOnly");

            if (externalI == null || externalI.Length < _n)
                throw new ArgumentException($"externalI must have at least {_n} entries");

            using var pin = Pin(externalI);
            return brain_step(_brain, _conn, pin.AddrOfPinnedObject(), dtMs);
        }

        /// <summary>Batch: run `nSteps` full pipeline steps, returning total spike count.</summary>
        public int StepBatch(float[] externalI, float dtMs, int nSteps)
        {
            EnsureNotDisposed();
            if (_conn == IntPtr.Zero)
                throw new InvalidOperationException("StepBatch requires a connectome");

            using var pin = Pin(externalI);
            return brain_step_n(_brain, _conn, pin.AddrOfPinnedObject(), dtMs, nSteps);
        }

        /// <summary>Compose the pipeline manually with StepFlags. Returns spike count.</summary>
        public int StepEx(float[] externalI, float dtMs, StepFlags flags)
        {
            EnsureNotDisposed();
            using var pin = externalI == null ? default : Pin(externalI);
            return brain_step_ex(_brain, _conn, Ptr(pin), dtMs, (int)flags);
        }

        /// <summary>Neuron step without a connectome (isolated population).</summary>
        public int StepNeuronsOnly(float[] externalI, float dtMs)
        {
            EnsureNotDisposed();
            using var pin = Pin(externalI);
            return brain_step_ex(_brain, IntPtr.Zero, pin.AddrOfPinnedObject(), dtMs,
                (int)(StepFlags.SetInput | StepFlags.Neuron | StepFlags.Push));
        }

        // ── State readers ─────────────────────────────────────────────────

        public void CopyVoltage(float[] dest)
        {
            EnsureNotDisposed();
            CheckDest(dest);
            using var pin = Pin(dest);
            brain_get_v(_brain, pin.AddrOfPinnedObject(), _n);
        }

        public void CopyRecovery(float[] dest)
        {
            EnsureNotDisposed();
            CheckDest(dest);
            using var pin = Pin(dest);
            brain_get_u(_brain, pin.AddrOfPinnedObject(), _n);
        }

        public void CopySpikes(int[] dest)
        {
            EnsureNotDisposed();
            CheckDest(dest);
            using var pin = Pin(dest);
            brain_get_spikes(_brain, pin.AddrOfPinnedObject(), _n);
        }

        public void CopyInputs(float[] dest)
        {
            EnsureNotDisposed();
            CheckDest(dest);
            using var pin = Pin(dest);
            brain_get_I_buf(_brain, pin.AddrOfPinnedObject(), _n);
        }

        // ── State writers ─────────────────────────────────────────────────

        public void WriteVoltage(float[] src)
        {
            EnsureNotDisposed();
            using var pin = Pin(src);
            brain_write_v(_brain, pin.AddrOfPinnedObject(), _n);
        }

        public void WriteInputs(float[] src)
        {
            EnsureNotDisposed();
            using var pin = Pin(src);
            brain_write_I_buf(_brain, pin.AddrOfPinnedObject(), _n);
        }

        public void InjectCurrent(int neuronIdx, float current)
        {
            EnsureNotDisposed();
            brain_inject_current(_brain, neuronIdx, current);
        }

        // ── Params ────────────────────────────────────────────────────────

        /// <summary>Override LIF params (null array = leave unchanged).</summary>
        public void SetLifParams(float[] tauM = null, float[] vRest = null,
            float[] vThresh = null, float[] vReset = null, float tauRef = 0f, float dtMs = 0f)
        {
            EnsureNotDisposed();
            using var p1 = tauM    == null ? default : Pin(tauM);
            using var p2 = vRest   == null ? default : Pin(vRest);
            using var p3 = vThresh == null ? default : Pin(vThresh);
            using var p4 = vReset  == null ? default : Pin(vReset);
            brain_set_lif_params(_brain, Ptr(p1), Ptr(p2),
                Ptr(p3), Ptr(p4), _n, tauRef, dtMs);
        }

        /// <summary>Override Izhikevich params (null array = leave unchanged).</summary>
        public void SetIzhikevichParams(float[] a = null, float[] b = null,
            float[] c = null, float[] d = null, float tauRef = 0f, float dtMs = 0f)
        {
            EnsureNotDisposed();
            using var p1 = a == null ? default : Pin(a);
            using var p2 = b == null ? default : Pin(b);
            using var p3 = c == null ? default : Pin(c);
            using var p4 = d == null ? default : Pin(d);
            brain_set_izhikevich_params(_brain, Ptr(p1), Ptr(p2),
                Ptr(p3), Ptr(p4), _n, tauRef, dtMs);
        }

        // ── Lifecycle ─────────────────────────────────────────────────────

        public void Reset()
        {
            EnsureNotDisposed();
            brain_reset(_brain);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            FreeConnectome();
            if (_brain != IntPtr.Zero)
            {
                brain_free(_brain);
                _brain = IntPtr.Zero;
            }
        }

        private void FreeConnectome()
        {
            if (_conn != IntPtr.Zero)
            {
                connectome_free(_conn);
                _conn = IntPtr.Zero;
            }
            if (_rpPin.IsAllocated)  _rpPin.Free();
            if (_ciPin.IsAllocated)  _ciPin.Free();
            if (_valPin.IsAllocated) _valPin.Free();
            if (_ntPin.IsAllocated)  _ntPin.Free();
        }

        // ── Helpers ───────────────────────────────────────────────────────

        private static GCHandle Pin<T>(T[] arr)
        {
            return GCHandle.Alloc(arr, GCHandleType.Pinned);
        }

        /// <summary>
        /// Safe pinned pointer: returns IntPtr.Zero for a non-allocated handle.
        /// Calling AddrOfPinnedObject() on a default GCHandle throws, so every
        /// optional-array call site must go through this helper.
        /// </summary>
        private static IntPtr Ptr(GCHandle h)
        {
            return h.IsAllocated ? h.AddrOfPinnedObject() : IntPtr.Zero;
        }

        private void CheckDest(Array dest)
        {
            if (dest == null || dest.Length < _n)
                throw new ArgumentException($"dest must have at least {_n} entries");
        }

        private void EnsureNotDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(TaasEngine));
        }
    }
}
