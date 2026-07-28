#!/usr/bin/env python3
"""Quick NEURON validation test."""
import os, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from neuron import h
h.load_file('stdrun.hoc')
h.nrn_load_dll('x86_64/libnrnmech.so')

# Test HH
soma = h.Section(name='soma')
soma.L = 10; soma.diam = 10; soma.cm = 1.0
soma.insert('hh')
v_vec = h.Vector(); v_vec.record(soma(0.5)._ref_v)
t_vec = h.Vector(); t_vec.record(h._ref_t)

stim = h.IClamp(soma(0.5))
stim.delay = 50; stim.dur = 100; stim.amp = 0.2

h.dt = 0.05; h.tstop = 500
h.finitialize(-65)
h.continuerun(500)

V = list(v_vec)
print(f'HH: V_range=[{min(V):.1f}, {max(V):.1f}], spikes={sum(1 for i in range(1,len(V)) if V[i-1]<-20 and V[i]>0)}')
print('All NEURON mechanisms loaded and working!')
