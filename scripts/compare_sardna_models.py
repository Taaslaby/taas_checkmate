#!/usr/bin/env python3
"""Compare sarDNA v2 vs CM2 — правильные конфиги из оригиналов."""
import sys
sys.path.insert(0, '/home/taaslab/Проекты/asm_engines/sardaanaCM')
import torch

S = '/home/taaslab/Проекты/asm_engines/sardaanaCM'

print('=' * 70)
print('  SARDNA SHOWDOWN: sarDNA v2 vs CM2 (original configs)')
print('=' * 70)

# 1. sarDNA v2
from src.connectome_gpt.sardna import SarDNA
dna = SarDNA.load_trained(f'{S}/models/sardna/sarDNA_v2.pt', config=None, device='cpu')
dna_params = sum(p.numel() for p in dna.parameters())
print(f'\n[sarDNA v2]')
print(f'  {dna_params/1e6:.2f}M params | {dna.config.n_layers}Lx{dna.config.d_model}dx{dna.config.n_heads}H')
print(f'  d_ff={dna.config.d_ff}, max_seq_len={dna.config.max_seq_len}')
print(f'  species: {dna.config.n_species}')

# 2. CM2 with CORRECT config from sardana_2/config.json
from src.connectome_gpt.model import SardaanaCM2, Config

# Original CM2 config from sardana_2/config.json
cm2_cfg = Config(
    d_model=384,
    n_heads=8,
    n_layers=8,
    d_ff=1536,
    dropout=0.1,
    max_seq_len=64,
)

# CM2 checkpoint has 2249 types, 658 neuropils
cm2 = SardaanaCM2(num_types=2249, num_neuropils=658, config=cm2_cfg)
ckpt = torch.load(f'{S}/models/cm2/cm2.pt', map_location='cpu', weights_only=True)

# Find state dict
if 'model_state_dict' in ckpt:
    sd = ckpt['model_state_dict']
elif 'state_dict' in ckpt:
    sd = ckpt['state_dict']
else:
    sd = ckpt

# Clean keys
clean_sd = {}
for k, v in sd.items():
    clean_k = k.replace('model.', '') if k.startswith('model.') else k
    clean_sd[clean_k] = v

missing, unexpected = cm2.load_state_dict(clean_sd, strict=False)
cm2.eval()
cm2_params = sum(p.numel() for p in cm2.parameters())
print(f'\n[CM2]')
print(f'  {cm2_params/1e6:.2f}M params | {cm2_cfg.n_layers}Lx{cm2_cfg.d_model}dx{cm2_cfg.n_heads}H')
print(f'  d_ff={cm2_cfg.d_ff}, max_seq_len={cm2_cfg.max_seq_len}')
print(f'  types: 2249, neuropils: 658')
print(f'  missing keys: {len(missing)}, unexpected: {len(unexpected)}')

print(f'\n  CM2 / sarDNA ratio: {cm2_params/dna_params:.1f}x')

# sarDNA generation for a few key species
species = {0:'FlyWire', 4:'Mouse', 5:'Rhesus'}
print(f'\n--- sarDNA v2 generation ---')
print(f'{"Sp":>2} | {"Species":<12} | {"N":>4} | {"Depth":>6} | {"Degree":>7} | {"Comm":>5} | {"Btwn":>5}')
print(f'{"-"*2}─┼─{"-"*12}─┼─{"-"*4}─┼─{"-"*6}─┼─{"-"*7}─┼─{"-"*5}─┼─{"-"*5}')
for sid, sname in species.items():
    toks = dna.generate(max_len=30, temperature=1.1, top_k=8, species_id=sid)
    nrn = [(d,g,c,b) for d,g,c,b in toks if 4<=d<=13 and 4<=g<=13 and 4<=c<=13 and 4<=b<=13]
    n = len(nrn)
    if n:
        ds = [(d-4)/9. for d,_,_,_ in nrn]
        gs = [(g-4)/9. for _,g,_,_ in nrn]
        cs = [(c-4)/9. for _,_,c,_ in nrn]
        bs = [(b-4)/9. for _,_,_,b in nrn]
        print(f'{sid:2d} | {sname:<12} | {n:4d} | {sum(ds)/n:6.3f} | {sum(gs)/n:7.3f} | {sum(cs)/n:5.3f} | {sum(bs)/n:5.3f}')
    else:
        print(f'{sid:2d} | {sname:<12} | {"-":>4}')

# CM2 generation
print(f'\n--- CM2 generation (greedy) ---')
with torch.no_grad():
    # Start with zero tokens
    t = torch.zeros(1, 1, dtype=torch.long)
    n = torch.zeros(1, 1, dtype=torch.long)
    d = torch.zeros(1, 1, dtype=torch.float32)
    types_out, npils_out, depths_out = [], [], []
    for step in range(20):
        lt, ln, ld = cm2(t, n, d)
        nt = lt[0,-1].argmax().item()
        nn = ln[0,-1].argmax().item()
        nd = ld[0,-1].item()
        types_out.append(nt)
        npils_out.append(nn)
        depths_out.append(nd)
        t = torch.cat([t, torch.tensor([[nt]])], dim=1)
        n = torch.cat([n, torch.tensor([[nn]])], dim=1)
        d = torch.cat([d, torch.tensor([[nd]])], dim=1)

# Try to decode types/neuropils if possible
try:
    from src.connectome_gpt.tokenizer import INDEX_MAP
    type_names = {v: k for k, v in INDEX_MAP['TYPE'].items()}
    npil_names = {v: k for k, v in INDEX_MAP['NEUROPIL'].items()}
    print(f'  {"#":>3} | {"Type":<22} | {"Neuropil":<22} | {"Depth":>6}')
    print(f'  {"-"*3}─┼─{"-"*22}─┼─{"-"*22}─┼─{"-"*6}')
    for i in range(min(10, len(types_out))):
        tn = type_names.get(types_out[i], f'type_{types_out[i]}')
        nn = npil_names.get(npils_out[i], f'npl_{npils_out[i]}')
        print(f'  {i:3d} | {tn:<22} | {nn:<22} | {depths_out[i]:6.3f}')
except:
    print(f'  First 10 types: {types_out[:10]}')
    print(f'  First 10 neuropils: {npils_out[:10]}')

print(f'\n  SUMMARY:')
print(f'  sarDNA v2: {dna_params/1e6:.2f}M — structural (4 heads, 7 species)')
print(f'  CM2:       {cm2_params/1e6:.2f}M — biological (3 heads, FlyWire only)')
print(f'  Sardna-Ultra: CM2 arch ({cm2_cfg.n_layers}Lx{cm2_cfg.d_model}d) + sarDNA task (4 heads, 7 species)')
print(f'  = {cm2_params/1e6:.2f}M params — понимает И структуру И биологию')
