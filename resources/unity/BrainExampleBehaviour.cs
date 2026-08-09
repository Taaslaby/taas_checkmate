// ═══════════════════════════════════════════════════════════════════════
// BrainExampleBehaviour.cs — minimal Unity MonoBehaviour driving TaasEngine.
//
// Demonstrates:
//   - brain + connectome creation (ring of LIF neurons)
//   - per-FixedUpdate stepping with external input
//   - spike-driven animation (neuron "activation" cubes)
//   - leak-free Dispose on destroy
//
// Setup:
//   1. Import TaasEngine.cs into your Assets folder.
//   2. Put taas_checkmate.dll into Assets/Plugins/.
//   3. Attach this behaviour to an empty GameObject with 32 small cubes as
//      children (one per neuron), or set nNeurons to match your children.
// ═══════════════════════════════════════════════════════════════════════

using System.Collections.Generic;
using UnityEngine;

namespace Taas.Examples
{
    public sealed class BrainExampleBehaviour : MonoBehaviour
    {
        [Header("Engine")]
        [Tooltip("Number of LIF neurons (ring connectome).")]
        public int nNeurons = 32;

        [Tooltip("Simulation dt in ms per FixedUpdate.")]
        public float dtMs = 0.5f;

        [Tooltip("Constant external drive added to every neuron.")]
        public float drive = 18f;

        [Header("Visualization")]
        public Transform neuronParent;      // children = one cube per neuron
        public Color activeColor = Color.red;
        public Color idleColor = Color.gray;
        public float spikeFlashSeconds = 0.05f;

        private TaasEngine _engine;
        private float[] _input;
        private int[] _spikes;
        private float[] _voltage;
        private float[] _flash;                              // allocated in Awake
        private List<Renderer> _renderers = new List<Renderer>();

        private void Awake()
        {
            if (neuronParent == null)
                neuronParent = transform;

            // Build the ring connectome: neuron i → (i+1) % n, weight 1.0.
            var rowPtr = new int[nNeurons + 1];
            var colIdx = new int[nNeurons];
            var values = new float[nNeurons];
            var ntTypes = new float[nNeurons];
            for (int i = 0; i < nNeurons; i++)
            {
                colIdx[i] = (i + 1) % nNeurons;
                values[i] = 0.8f;          // mild recurrent drive
                ntTypes[i] = 1.0f;         // excitatory
                rowPtr[i + 1] = i + 1;
            }

            _engine = new TaasEngine(nNeurons, BrainModel.Lif);
            _engine.LoadConnectome(rowPtr, colIdx, values, ntTypes);

            _input = new float[nNeurons];
            _spikes = new int[nNeurons];
            _voltage = new float[nNeurons];
            _flash = new float[nNeurons];

            // Collect renderers of child cubes.
            foreach (Transform child in neuronParent)
            {
                var r = child.GetComponentInChildren<Renderer>();
                if (r != null) _renderers.Add(r);
            }
        }

        private void FixedUpdate()
        {
            if (_engine == null) return;

            // Drive: constant baseline + small sinusoidal modulation so the
            // network shows interesting dynamics (some neurons fire, others not).
            float time = Time.fixedTime;
            for (int i = 0; i < nNeurons; i++)
            {
                _input[i] = drive + 2f * Mathf.Sin(time * 5f + i * 0.3f);
            }

            _engine.Step(_input, dtMs);
            _engine.CopySpikes(_spikes);
            _engine.CopyVoltage(_voltage);

            // Flash neurons that spiked this tick.
            float dt = Time.fixedDeltaTime;
            for (int i = 0; i < nNeurons; i++)
            {
                if (_spikes[i] > 0) _flash[i] = spikeFlashSeconds;
                else _flash[i] = Mathf.Max(0f, _flash[i] - dt);
            }

            for (int i = 0; i < _renderers.Count && i < nNeurons; i++)
            {
                Color c = _flash[i] > 0f
                    ? Color.Lerp(activeColor, idleColor, _flash[i] / spikeFlashSeconds)
                    : idleColor;
                _renderers[i].material.color = c;
            }
        }

        private void OnDestroy()
        {
            _engine?.Dispose();
            _engine = null;
        }

        private void OnGUI()
        {
            GUILayout.Label($"Neurons: {nNeurons}   Spikes last tick: {CountSpikes()}");
        }

        private int CountSpikes()
        {
            if (_spikes == null) return 0;
            int n = 0;
            for (int i = 0; i < _spikes.Length; i++) if (_spikes[i] > 0) n++;
            return n;
        }
    }
}
