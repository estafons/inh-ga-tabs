import math


def betafunc(comb, StringBetasObj, constants):
    fret = constants.train_frets[0]
    beta = StringBetasObj.betas_array[comb[0]][fret] * 2**(comb[1]/6)
    return beta

#----------------beta func example for exp model
def expfunc(comb, StringBetasObj, constants):
    fret1, fret2 = constants.train_frets[0], constants.train_frets[-1]
    b2, b1 = StringBetasObj.betas_array[comb[0]][fret2], StringBetasObj.betas_array[comb[0]][fret1]
    a = 6 * (math.log2(b2) - math.log2(b1)) / (fret2-fret1)
    beta = StringBetasObj.betas_array[comb[0]][fret1] * 2**(a * comb[1]/6)
    return beta

def linfunc(comb, StringBetasObj, constants):
    fret1, fret2 = constants.train_frets[0], constants.train_frets[-1]
    b2, b1 = StringBetasObj.betas_array[comb[0]][fret2], StringBetasObj.betas_array[comb[0]][fret1]
    a = 6 * (math.log2(b2) - math.log2(b1)) - fret2 + fret1
    beta = StringBetasObj.betas_array[comb[0]][fret1] * 2**((comb[1]+ a)/6 )
    return beta

def aphfunc(comb, StringBetasObj, constants):
    fret1, fret2, fret3 = constants.train_frets[0], constants.train_frets[1], constants.train_frets[2]
    b3, b2, b1 = (StringBetasObj.betas_array[comb[0]][fret3], 
                    StringBetasObj.betas_array[comb[0]][fret2], 
                            StringBetasObj.betas_array[comb[0]][fret1])
    a = 6 * (math.log2(b3) - math.log2(b2)) / (fret3 - fret2)
    b = 6 * (math.log2(b3) - math.log2(b1)) - a * (fret3 - fret1)
    beta = StringBetasObj.betas_array[comb[0]][fret1] * 2**((a*comb[1] +b)/6)
    return beta
