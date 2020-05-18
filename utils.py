import getpass

def get_project_paths():
    paths = {}
    if getpass.getuser() == 'slebst':
        paths['root'] = '/home/slebst/Documents/Workspace/Jobs/Columbia/ultrametric_benchmark/Ultrametric-Benchmark/'
    elif getpass.getuser() == 'sl4744':
        paths['root'] = '/rigel/theory/users/sl4744/projects/Ultrametric-benchmark'
    else:
        paths['root'] = '/home/slebst/Documents/Workspace/Jobs/Columbia/ultrametric_benchmark/Ultrametric-Benchmark/'
    paths['plots'] = paths['root'] + 'plots/'
    paths['simus'] = paths['root'] + 'Results/'
    paths['nb'] = paths['root'] + 'notebooks/'
    paths['jobs'] = paths['root'] + 'cluster_job/'
    paths['misc'] = paths['root'] + 'misc/'
    
    return paths