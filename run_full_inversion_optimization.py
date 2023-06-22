from OpenMiChroM.ChromDynamics import MiChroM # OpenMiChroM simulation module
from OpenMiChroM.Optimization import AdamTraining # Adam optimization module

# modules to load and plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import h5py
import os
import sys
import time

def load_map(hic_file):
    hic_map = np.loadtxt(hic_file)
    num_beads = hic_map.shape[0]
    print("number of beads: ", num_beads)
    return hic_map

def initialize_optimization(hic_file, eta=0.02, cutoff=1e-4):
    #initialize optimization
    opt = AdamTraining(mu=2.0, rc = 2.0, eta=eta, it=1)
    opt.getPars(HiC=hic_file, norm=False, cutoff=cutoff)
    num_beads = opt.phi_exp.shape[0]
    _force_field_initializer(num_beads=num_beads)
    return opt

def _force_field_initializer(num_beads, init_ff='zeros'):
    #check if input dir exists, otherwise make it
    if not os.path.isdir('input'): 
        os.mkdir('input')
        print("Made `input` directory to store files")
    #generate sequence file
    with open('./input/seq.txt', 'w') as fseq:
        for ii in range(num_beads):
            fseq.write(f'{ii+1} Bead{ii+1}\n')
    seq = np.loadtxt('./input/seq.txt', dtype=str)[:,1]
    
    if init_ff=='zeros': init_ff = np.zeros((num_beads, num_beads))
    assert (len(init_ff.shape)==2 and init_ff.shape[0]==num_beads and init_ff.shape[1]==num_beads), "init_ff should be an array of dim (num_beads x num_beads)"
    lamb = pd.DataFrame(init_ff, columns=seq, index=seq)
    lamb.to_csv("input/lambda_0", index=False)

def initialize_simulation(name='opt_sim', platform='opencl', collapse=True, nblocks_collapse=10, blocksize_collapse=100000):
    #simulate
    sim = MiChroM(name=name,temperature=1.0, time_step=0.01, collision_rate=0.1)
    sim.setup(platform=platform)
    sim.saveFolder('output/')
    mychro = sim.createSpringSpiral(ChromSeq="input/seq.txt", isRing=True)
    sim.loadStructure(mychro, center=True)

    #add potentials
    sim.addFENEBonds(kfb=30.0)
    sim.addAngles(ka=2.0)
    sim.addRepulsiveSoftCore(Ecut=4.0)
    sim.addCustomTypes(mu=3.22, rc = 1.78, TypesTable='input/lambda_0')
    # sim.addFlatBottomHarmonic(kr=0.1, n_rad=10.0)
    sim.addCylindricalConfinement(r_conf=3.0, z_conf=10.0, kr=0.5)

    if collapse:
        print('Running collapse simulation')
        for _ in range(nblocks_collapse): 
            sim.runSimBlock(blocksize_collapse, increment=False)
        sim.saveStructure(mode = 'pdb')
    return sim

def collapse_under_confinement(sim, nblocks=100, blocksize=5000, kr=0.01, n_rad=7.0):
    if not sim.forcesApplied:
        sim.addFlatBottomHarmonic(kr=kr, n_rad=n_rad)
    else:
        sim.addAdditionalForce(sim.addFlatBottomHarmonic, kr=kr, n_rad=n_rad)

    #collapse run
    for _ in range(nblocks): sim.runSimBlock(blocksize, increment=False)
    sim.saveStructure(mode = 'pdb')
    #remove confinement
    sim.removeForce("FlatBottomHarmonic")

def temper_polymer(sim, T_high=300, blocksize=10000, dT=30):
    T_actual = sim.integrator.getTemperature()._value
    n_temper = int(T_high-T_actual)//dT + 1
    
    for jj in range(n_temper):
        sim.integrator.setTemperature(T_high-dT*jj)
        sim.runSimBlock(blocksize, increment=False)
        print("Temperature:", sim.integrator.getTemperature())
        
    sim.integrator.setTemperature(T_actual)
    sim.runSimBlock(blocksize, increment=False)
    print("Final temperature:", sim.integrator.getTemperature())

def train(sim, opt, n_steps, n_replicas=10, n_blocks=5000, blocksize=1000, temper_between_replicas=True):
    sim.initStorage(filename=sim.name)

    for iter_step in range(1,n_steps):
        print(f"Iteration {iter_step}")
        opt.reset_Pi()
        for replica in range(n_replicas):
            if temper_between_replicas: temper_polymer(sim)
            
            for _ in range(n_blocks):
                sim.runSimBlock(blocksize, increment=True) 
                pos = sim.getPositions()
                opt.probCalc(pos)
                if replica==0: 
                    sim.saveStructure(mode = 'pdb')
                    sim.saveStructure()
        
        #compute new lambdas and store into csv
        lamb_new = opt.getLamb(Lambdas=f"input/lambda_{iter_step-1}")
        lamb_new.to_csv(f"input/lambda_{iter_step}", index=False)
        error = opt.error
        with open('output/error.txt','a+') as ferr: ferr.write(f"{iter_step} {error}\n")
        print("ERROR: ", error)

        with h5py.File(os.path.join(sim.folder,f"Pi_{iter_step}.h5"), 'w') as hf:
            hf.create_dataset("Pi", data=opt.Pi/opt.NFrames)

        sim.removeForce("CustomTypes")
        sim.addAdditionalForce(sim.addCustomTypes, TypesTable=f"input/lambda_{iter_step}")
    
    sim.storage[0].close()

def main():
    hic_file = sys.argv[1]
    opt = initialize_optimization(hic_file, eta=0.02, cutoff=2e-3)
    sim = initialize_simulation(name='Wang2015_delSMC', platform='CUDA',)
    train(sim, opt, n_steps=20, n_replicas=10, n_blocks=5000, blocksize=1000, temper_between_replicas=True)

if __name__ == "__main__": main()