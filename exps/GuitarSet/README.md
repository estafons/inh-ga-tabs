## Contextual-based Classification

This directory tree is expected in order to store your data and results:

```
.
├── constants.ini
├── data
│   ├── annos
│   ├── audio
│   └── train
└── results
```


## Performing Tests on GuitarSet dataset

Download from GuitarSet dataset (https://zenodo.org/record/3371780):
1) ```annotations.zip``` and extract all ```.jams``` files to ```data/annos```.
2) ```audio_mono-mic.zip```, ``` audio_mono-pickup_mix.zip``` and extract all ```.wav``` files to ```data/audio```.

Specify the folders on the configuration file (currently ```constants.ini```) where annotations and input audio is stored. 

The track names from the subset from GuitarSet that was considered monophonic (more than 60ms between onsets), is stored as a txt file and uploaded (```names.txt```). Specify the location where the ```names.txt``` file is located or another subset of your choice. 

To reproduce all results, run: 
```
python automate_scripts.py
```

Confusion Matrices containting the overall accuracy are produced and stored in .png form in ```./results``` dir.

For more controlable experiments run the command below with the dsirable arguments or configure the model by tweekinig values of ```constants.ini``` accordingly:

```
python GuitarSetTest.py constants.ini . {-plot} {-verbose} {-run_genetic_alg} --dataset {mix, mic} --train_mode {1Fret,2FretA,2FretB,3Fret}
```

In order to plot guitar inharmonic irregularity, run:
```
python n_tonos.py constants.ini . {-compute} {-train}
```


### Training for GuitarSet
Methods for training on the guitarset dataset on isolated note instances can be found in ```GuitarTrain.py``` script. A folder structure as <#midi_note>──<#string> is expected where cropped note instances are stored for the specified midi_note and string number (strings are numbered 0,1,2,3,4,5 as E,A,G,D,B,e). Running the ```GuitarSetTrainWrapper``` method will print the betas computed and return a ```StringBetas``` object where they are stored. Also the user can specify the frets she wishes to train on on the ```constants.ini``` file at constant ***train_frets***.
