# Screening monolayer terminal groups with 3 additional terminal groups for consideration

## Installation/Set-up

#### Download and install anaconda
Note: I did this in the /ccs/proj/ directory on Titan

Note: Titan has been decomissioned as of [August 01, 2019](https://www.olcf.ornl.gov/2019-olcf-system-changes/#titan-eos-decommission); some of these commands might not work on other systems without editing.
#### Create a new environment (3.5 is preferred)
`>> conda create --name myconda python=3.5`

#### Activate the environment
`>> source activate myconda`

Note: You may have to first append to your path the directory where anaconda
is located, e.g.

`>> export PATH=/ccs/proj/xxx000/anaconda/titan/bin:$PATH`

#### Clone and install atools
commit caa89f99a07c84be4c457378138105100eaa9dfb

`>> git clone https://github.com/PTC-CMC/atools.git`

`>> pip install .`

#### Clone and install mbuild
commit fa2bc651823d8c0a93cac8721e0abf10a7b5e168

`>> git clone https://github.com/mosdef-hub/mbuild.git`

`>> pip install .`

#### Clone and install foyer
commit 1aa97bbebed22c94ad8d9d68486fbdbe7a3bd6d7

`>> git clone https://github.com/mosdef-hub/foyer.git`

`>> pip install .`

#### Install signac-flow
`>> conda install signac-flow=0.5.4 -c glotzer`

#### Install dependencies
Note: If mBuild and Foyer are installed via conda or pip, these dependencies
should be installed automatically.
```
>> conda config --add channels omnia mosdef
>> conda install lxml requests networkx mdtraj oset parmed openmm plyplus
>> pip install mdanalysis
```

#### Clone the terminal_group_mixed_original_16_new_3 repository
`>> git clone https://github.com/PTC-CMC/terminal_group_mixed_original_16_new_3.git`

#### Initialize the project
Note: All flow commands must be performed from the project root directory.

Note: The `-n 3 -c 17 1` signifies that three statepoints will be created for each
parameter state, with an alkane backbone of 17 carbons, each with a different random seed (incrementing from 1)

`>> python src/init.py -n 3 -c 17 1`

----------
## Signac workflow

#### Initialize/construct systems
This will submit jobs in bundles of 6 statepoints to be executed on
a single node. Although each node contains 16 processors, memory issues
limit the number of simultaneous systems that can be initialized.

`>> python src/project.py submit -o initialize_system --bundle 6 --nn 1 -w 0.5`

#### Run minimization in LAMMPS to fix overlaps
`>> python src/project.py submit -o fix_overlaps --bundle 400 --nn 400 -w 1`

#### Convert last frame of LAMMPS trajectory to a GROMACS structure file
`>> python src/project.py submit -o lmp_to_gmx --bundle 48 --nn 3 -w 0.5`

#### Create TPR file for GROMACS energy minimization
`>> python src/project.py submit -o em_grompp --bundle 400 --nn 400 -w 0.5`

#### Run GROMACS energy minimization
`>> python src/project.py submit -o mdrun_em --bundle 400 --nn 400 -w 0.5`

#### Create TPR file for GROMACS NVT equilibration
`>> python src/project.py submit -o nvt_equil_grompp --bundle 400 --nn 400 -w 0.5`

#### Run GROMACS NVT equilibration
`>> python src/project.py submit -o nvt_mdrun --bundle 400 --nn 400 -w 2`

#### Create TPR file for GROMACS compression
`>> python src/project.py submit -o compress_grompp --bundle 400 --nn 400 -w 0.5`

#### Run GROMACS compression
`>> python src/project.py submit -o mdrun_compress --bundle 400 --nn 400 -w 1`

#### Create TPR file for GROMACS shear at a normal load of 5nN
`>> python src/project.py submit -o shear_5nN_grompp --bundle 400 --nn 400 -w 0.5`

#### Run GROMACS shear at a normal load of 5nN

Note: Shear was originally performed for 5ns and then extended another 5ns. The MDP files have been updated to include the full 10ns now.

`>> python src/project.py submit -o mdrun_shear_5nN --bundle 400 --nn 400 -w 4`

#### Create TPR file for GROMACS shear at a normal load of 15nN
`>> python src/project.py submit -o shear_15nN_grompp --bundle 400 --nn 400 -w 0.5`

#### Run GROMACS shear at a normal load of 15nN
`>> python src/project.py submit -o mdrun_shear_15nN --bundle 400 --nn 400 -w 4`

#### Create TPR file for GROMACS shear at a normal load of 25nN
`>> python src/project.py submit -o shear_25nN_grompp --bundle 400 --nn 400 -w 0.5`

#### Run GROMACS shear at a normal load of 25nN
`>> python src/project.py submit -o mdrun_shear_25nN --bundle 400 --nn 400 -w 4`

NOTE: If the simulations do not complete in the hours provided, concatenating the TRR files is necessary.

This is easier to do through the `trjcat` command from command line `GROMACS`.

`gmx_mpi trjcat -f shear_5nN*.trr -o shear_5nN_combined.trr`

`gmx_mpi trjcat -f shear_15nN*.trr -o shear_15nN_combined.trr`

`gmx_mpi trjcat -f shear_25nN*.trr -o shear_25nN_combined.trr`

----------
## Post-processing/Analysis

#### Calculate friction forces for each shear trajectory
`>> python src/analysis.py submit -o calc_friction_system --bundle 48 --nn 3 -w 1`

#### Log COF
`>> python src/analysis.py submit -o calc_cof --bundle 48 --nn 3 -w 1`
