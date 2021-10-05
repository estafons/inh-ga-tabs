This is a dataset consisting of guitar recordings used in the paper:
"A PARAMETRIC APPROACH TO EXTRACTION OF STRING, FRET AND PLUCKING
POSITION ALONG THE GUITAR STRING" submitted to ICASSP 2019.

The authors are: Jacob Møller and Mads Græsbøll Christensen
Audio Analysis Lab, CREATE, Aalborg University.

e-mail: {jmhh, mgc}@create.aau.dk


The structure of the directories is as follows:
1) ROOT DIRECTORY
In the root directory each directory contains recordings of every combination of string and fret for the given guitar. The name of each directory explains the guitar type and has a numerical index that can be used to distuingish all recordings.

2) NEXT LEVEL DIRECTORIES
If you are located inside any of the directories mentioned in "1)", there is six directories with the name of a given string (string 1-string 6). Inside each of these directories you will find 13 recordings representing each fret on this given string.
Note that the string names are in opposite order such that string 1 represents the low E-string and string 6 represents the high E-string.   


The data is recorded as line signals with sampling rate 44.1 kHz as line signals with a Roland Quad-Capture USB 2.0 sound card.

Along with the dataset containing the recordings of electric
and acoustic guitar, a script follows which was used for testing in the paper.

If you find this code or data set useful, please cite the paper:

------------------------------------------------------------------------
LaTeX citation format:

 @inproceedings{hjerrild_icassp2019,
  title={Estimation of guitar string, fret and plucking position using parametric pitch estimation},
  author={Hjerrild, Jacob M{\o}ller and Christensen, {Mads Gr\ae sb\o ll}},
  booktitle= {Proc.\ IEEE Int.\ Conf.\ Acoust., Speech, Signal Process.},
  year={2019}
}
 ------------------------------------------------------------------------



