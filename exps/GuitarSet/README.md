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
<!-- 3) what about training samples -->

Specify the folders on the configuration file (currently ```constants.ini```) where annotations and input audio is stored. 

The track names from the subset from GuitarSet that was considered monophonic (more than 60ms between onsets), is stored as a txt file and uploaded (```names.txt```). Specify the location where the ```names.txt``` file is located or another subset of your choice. 

Then run:
```
python GuitarSetTest.py constants.ini . {-plot} {-verbose} {-run_genetic_alg} --dataset {mix, mic} --train_mode {1Fret,2FretA,2FretB,3Fret}
```

Run:
```
python n_tonos.py constants.ini . {-compute} {-train}
```
in order to plot guitar irregularity.

<!-- Then run the function ```TestGuitarSet``` from script ```GuitarSetTest.py``` and a confusion matrix will be saved at the location as specified in the ```constants.ini``` file. -->

### Training for GuitarSet
Methods for training on the guitarset dataset on isolated note instances can be found in ```GuitarTrain.py``` script. A folder structure as <#midi_note>──<#string> is expected where cropped note instances are stored for the specified midi_note and string number (strings are numbered 0,1,2,3,4,5 as E,A,G,D,B,e). Running the ```GuitarSetTrainWrapper``` method will print the betas computed and return a ```StringBetas``` object where they are stored. Also the user can specify the frets she wishes to train on on the ```constants.ini``` file at constant ***train_frets***.


# Calling HjerrildTest or GuitarSetTest
For now call as python HjerrildTest.py /absolute/path/to file/constants.ini /absolute/path/to/data/folder/
containing training data, annotations and audio files as displayed above.

# Specifiying the mode for computing partials
For now 3 methods are available for computing partials. ```compute_partials, compute_partials_with_order, and compute_partials_with_order_strict```. The first one is currently the stable and mostly tested. To choose compute_partials function user specifies it at the ToolBox object instantiation before the inharmonic analysis processes, in which cases user inputs arguements as list [no_of_partials,fundamental_init/2, constants]. In the later two cases, the user specifies at the same place the function to employ with arguements as [fundamental_init/2, constants, StringBetaObj]. Keep in mind for the compute_partials_with_order and compute_partials_with_order_strict user must first call the method ```compute_partial_orders(StrBetaObj, constants)```to initialize the max partial orders that can be used. More will be uploaded soon on each function's purpose. Currently lines 55-56 on hjerrildTest and line 74 on GuitarSetTest