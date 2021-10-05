Download the zip file ```MATLAB_hjerrild_icassp19_guitar_string_fret_and_plucking_estimator``` from the link below

https://vbn.aau.dk/en/publications/estimation-of-guitar-string-fret-and-plucking-position-using-para

and extract the content of dir ```hjerrild_ICASSP2019_guitar_dataset``` to ```./data/tran'```.


Configure the model by tweekinig values of ```constants.ini``` accordingly and then run:

```
python HjerrildTest.py constants.ini . {-plot} --guitar {martin, firebrand} --train_mode {1Fret, 2FretA, 2FretB, 3Fret}
```

The results will be stored to dir ```./results```.
