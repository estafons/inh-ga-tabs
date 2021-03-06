import subprocess
from pathlib import Path


if __name__ == '__main__':

  
    script_path = Path(Path.cwd(), 'GuitarSetTest.py')
    constants_path = Path(Path.cwd(), 'constants.ini')
    workspace_path = Path(Path.cwd() )
    for dataset in ['mix', 'mic']:
        for train_mode in ['1Fret', '2FretA', '2FretB', '3Fret']:
            cmd_list = ['python', script_path, constants_path, workspace_path, '--dataset', dataset,'--train_mode', train_mode, '-run_genetic_alg']
            proc = subprocess.Popen(cmd_list, stdin=None, stdout=None, stderr=None)#, shell=True)    
            