## Audio-Based String Classifier


Download the zip file ```MATLAB_hjerrild_icassp19_guitar_string_fret_and_plucking_estimator.zip``` from the link below

https://vbn.aau.dk/en/publications/estimation-of-guitar-string-fret-and-plucking-position-using-para

and extract the content of dir ```hjerrild_ICASSP2019_guitar_dataset``` to ```./data/tran'```.


To reproduce all results, run: 
```
python automate_script.py
```

Confusion Matrices containting the overall accuracy are produced and stored in .png form in ```./results``` dir.

For more controlable experiments run the command below with the dsirable arguments or configure the model by tweekinig values of ```constants.ini``` accordingly:

```
python HjerrildTest.py constants.ini . {-plot} --guitar {martin, firebrand} --train_mode {1Fret, 2FretA, 2FretB, 3Fret}
```

The results will be stored to dir ```./results```.

