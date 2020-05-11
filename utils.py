import getpass

def get_project_paths():
    paths = {}
    if 'slebastard' not in getpass.getuser():
        paths['root'] = 'C:/Users/Antonin/Documents/Documents/ENS 2A/Stage M1/Code/github/Ultrametric-benchmark/'
    else:
        paths['root'] = '/home/slebastard-adc/Documents/Projects/ultrametric_benchmark/Ultrametric-Benchmark/'
    paths['plots'] = paths['root'] + 'plots/'
    paths['simus'] = paths['root'] + 'Results/'
    paths['nb'] = paths['root'] + 'notebooks/'
    paths['jobs'] = paths['root'] + 'cluster_job/'
    
    return paths