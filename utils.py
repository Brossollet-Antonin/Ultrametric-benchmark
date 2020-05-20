import getpass
import inspect

def get_project_paths():
    paths = {}
    if getpass.getuser() == 'slebst':
        paths['root'] = '/Users/slebst/ultrametric_benchmark/Ultrametric-benchmark/'
    elif getpass.getuser() == 'sl4744':
        paths['root'] = '/rigel/theory/users/sl4744/projects/Ultrametric-benchmark/'
    elif getpass.getuser() == 'ab4877':
        paths['root'] = '/rigel/theory/users/ab4877/Ultrametric-benchmark/'
    elif getpass.getuser() == 'Antonin':
        paths['root'] = 'C:/Users/Antonin/Documents/Documents/ENS 2A/Stage M1/Code/github/Ultrametric-benchmark/'
    paths['plots'] = paths['root'] + 'plots/'
    paths['simus'] = paths['root'] + 'Results/'
    paths['nb'] = paths['root'] + 'notebooks/'
    paths['jobs'] = paths['root'] + 'cluster_job/'
    paths['misc'] = paths['root'] + 'misc/'
    
    return paths

def verbose(message, verb_lvl, lvl=1):
    if verb_lvl >= lvl:
        print(message)

#####################################

def get_lbl_distr(shuffled_sequence, min_range, max_range, n_classes):
    """
    Inputs:
    shuffled_sequence: list
    min_range: int
    max_range: int

    Returns:
    histogram of the labels distribution from min_range to max_range
    """
    hist_tuple = np.histogram(
        shuffled_sequence[min_range:max_range],
        bins = n_classes
        )

    return hist_tuple[0]