import subprocess
from pathlib import Path


if __name__ == '__main__':

    
    script_path = Path(Path.cwd(), 'HjerrildDataset','HjerrildTest.py')
    constants_path = Path(Path.cwd(), 'HjerrildDataset', 'constants.ini')
    workspace_path = Path(Path.cwd(), 'HjerrildDataset')
    for guitar in ['martin', 'firebrand']:
        for train_mode in ['3Fret', '2FretA', '2FretB', '3Fret']:
            subprocess.run(['python', script_path, constants_path, 
                                    workspace_path, '--guitar', guitar,
                                        '--train_mode', train_mode])

    script_path = Path(Path.cwd(), 'GuitarSet','GuitarSetTest.py')
    constants_path = Path(Path.cwd(), 'GuitarSet', 'constants.ini')
    workspace_path = Path(Path.cwd(), 'GuitarSet')
    for dataset in ['mix', 'mic']:
        for train_mode in ['1Fret', '2FretA', '2FretB', '3Fret']:
            subprocess.run(['python', script_path, constants_path, 
                                workspace_path, '--dataset', dataset,
                                    '--train_mode', train_mode])