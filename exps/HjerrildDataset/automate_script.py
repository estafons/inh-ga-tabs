import subprocess
from pathlib import Path


if __name__ == '__main__':

  
    script_path = Path(Path.cwd(), 'HjerrildTest.py')
    constants_path = Path(Path.cwd(),'constants.ini')
    workspace_path = Path(Path.cwd())
    for guitar in ['martin', 'firebrand']:
        for train_mode in ['1Fret', '2FretA', '2FretB', '3Fret']:
            cmd_list = ['python', str(script_path), str(constants_path), str(workspace_path), '--guitar', guitar,'--train_mode', train_mode]
            proc = subprocess.Popen(cmd_list, stdin=None, stdout=None, stderr=None)                                        