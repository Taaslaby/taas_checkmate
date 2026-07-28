#!/usr/bin/env python3
"""Generate mouse connectome via Sardna and compare across species."""
import sys
sys.path.insert(0, '/home/taaslab/Проекты/asm_engines/sardaanaCM')
from src.connectome_gpt.sardna import SarDNA


def generate_neurons(model, species_id, n=20, temp=0.8, topk=8):
    tokens = model.generate(max_len=n + 4, temperature=temp, top_k=topk, species_id=species_id)
    neurons = []
    for d, g, c, b in tokens:
        if 4 <= d <= 13 and 4 <= g <= 13 and 4 <= c <= 13 and 4 <= b <= 13:
            neurons.append({
                'depth': (d - 4) / 9.0,
                'degree': (g - 4) / 9.0,
                'comm': (c - 4) / 9.0,
                'btwn': (b - 4) / 9.0,
            })
    return neurons[:n]


def main():
    # Load Sardna
    model = SarDNA.load_trained(
        '/home/taaslab/Проекты/asm_engines/sardaanaCM/models/sardna/sarDNA_v2.pt',
        config=None, device='cpu'
    )
    print(f'Loaded: d_model={model.config.d_model}, n_species={model.config.n_species}')
    print(f'Species IDs: 0..{model.config.n_species - 1}')

    species_names = {
        0: 'FlyWire (дрозофила)',
        1: 'MANC (VNC мухи)',
        2: 'C. elegans гермафродит',
        3: 'C. elegans самец',
        4: 'Mouse brain (мышь)',
        5: 'Rhesus brain (макака)',
        6: 'Rhesus cortex (кора макаки)',
    }

    print('\n' + '=' * 65)
    print('  🧬 Sardna Species Comparison — Mouse Brain Generation')
    print('=' * 65)

    for sid in range(7):
        name = species_names.get(sid, f'Species {sid}')
        neurons = generate_neurons(model, sid, n=10)

        if not neurons:
            print(f'\n  [{sid}] {name}: ❌ no output')
            continue

        depths = [n['depth'] for n in neurons]
        degrees = [n['degree'] for n in neurons]
        comms = [n['comm'] for n in neurons]
        btw = [n['btwn'] for n in neurons]

        mu_d = sum(depths) / len(depths)
        mu_g = sum(degrees) / len(degrees)
        mu_c = sum(comms) / len(comms)
        mu_b = sum(btw) / len(btw)

        bar_d = '█' * int(mu_d * 20)
        bar_g = '█' * int(mu_g * 20)

        print(f'\n  [{sid}] {name}')
        print(f'        depth={mu_d:.3f} {bar_d}')
        print(f'        degree={mu_g:.3f} {bar_g}')
        print(f'        comm={mu_c:.3f}  btwn={mu_b:.3f}')

    # Mouse brain detailed
    print('\n' + '=' * 65)
    print('  🖱️  MOUSE BRAIN (species_id=4) — DETAILED')
    print('=' * 65)

    mouse_neurons = generate_neurons(model, 4, n=30)
    print(f'\n  {"#":>3} | {"Depth":>6} | {"Degree":>7} | {"Comm":>5} | {"Btwn":>5}')
    print(f'  {"-"*3}─┼─{"-"*6}─┼─{"-"*7}─┼─{"-"*5}─┼─{"-"*5}')
    for i, n in enumerate(mouse_neurons[:15]):
        print(f'  {i:3d} | {n["depth"]:6.3f} | {n["degree"]:7.3f} | {n["comm"]:5.3f} | {n["btwn"]:5.3f}')
    if len(mouse_neurons) > 15:
        print(f'  {"...":3} | {"...":6} | {"...":7} | {"...":5} | {"...":5}')

    # Stats summary
    print('\n' + '=' * 65)
    print('  📊 SPECIES SUMMARY')
    print('=' * 65)
    print(f'  {"Species":<25} | {"Depth":>6} | {"Degree":>7} | {"Comm":>5} | {"Btwn":>5}')
    print(f'  {"-"*25}─┼─{"-"*6}─┼─{"-"*7}─┼─{"-"*5}─┼─{"-"*5}')

    for sid in range(7):
        name = species_names.get(sid, f'Species {sid}')
        neurons = generate_neurons(model, sid, n=30)
        if not neurons:
            continue
        depths = [n['depth'] for n in neurons]
        degrees = [n['degree'] for n in neurons]
        comms = [n['comm'] for n in neurons]
        btw = [n['btwn'] for n in neurons]
        print(f'  {name:<25} | {sum(depths)/len(depths):6.3f} | {sum(degrees)/len(degrees):7.3f} '
              f'| {sum(comms)/len(comms):5.3f} | {sum(btw)/len(btw):5.3f}')

    print('\n  ✅ Done!')


if __name__ == '__main__':
    main()
