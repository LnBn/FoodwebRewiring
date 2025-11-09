# FoodwebRewiring
Julia scripts implementing various dynamical mechanisms for foodweb rewiring 


## Package requirements

Ensure the folder named 'Rewiring' containing Project.toml and Manifest.toml is in the same directory in the script. Uncomment the line "Pkg.instantiate()" on line 3 in the script. The necessary packages should auto-install when the script is run.


## Configuration

The script is (or will be) capable of simulating many different rewiring scenarios. Most parameters are listed at the top of the script and can be tweaked there. I will continue revising this for convenience. 

## Output
As run, the script should produce a picture of the initial graph, a picture of the final graph, a picture of the population timeseries, and some summary/log files of the simulation. These will be refined and more will be added.

![image](timeseries.png)

## Current issues

- USE_RANDOM_INITIAL needs to be set to 'true' - eventually there will be an option to automatically find stable equlibria (where all species persist without adaptation/rewiring) to use as ICs.
- Currently invasions don't work because they cause issues with the ODE solver due to resizing of the system. ENABLE_INVASION should be set to false.
- I still need to figure out "reasonable" parameter choices - e.g. choices that recover the findings of Gilljam and Kondoh in each scenario, and how to balance those mechanisms. Currently adapting/rewiring seems very destructive. Also not clear yet what the most 'important' parameters are to induce qualitative changes in the behaviour of the system.
