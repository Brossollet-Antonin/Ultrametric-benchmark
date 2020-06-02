import matplotlib
import getpass
import inspect
import ast

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

def get_simus_directory():
    paths = get_project_paths()
    with open(paths['simus']+'simu_mapping_compact.txt', 'r', encoding='utf-8') as filenames:
        filenames_dct_txt = filenames.read().replace('\n', '')
        
    sim_directory = ast.literal_eval(filenames_dct_txt)
    return sim_directory

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

### Figure handling methods

def format_paper(fig_width=13.2, fig_height=9, size=15, line_width=1.5,
                axis_line_width=1.0, tick_size=12, tick_label_size=20,
                label_pad=4, legend_loc='lower right'):
    def cm2inch(x): return x/2.54
    fig_height = cm2inch(fig_height)
    fig_width = cm2inch(fig_width)
    rcParams = matplotlib.rcParams

    rcParams["figure.figsize"] = [fig_width, fig_height]   #default is [6.4, 4.8]
    rcParams["font.sans-serif"] = "Tahoma"
    rcParams["font.size"] = size
    rcParams["legend.fontsize"] = size
    rcParams["legend.frameon"] = False
    rcParams["legend.loc"] = legend_loc
    rcParams["axes.labelsize"] = size
    rcParams["xtick.labelsize"] = tick_label_size
    rcParams["ytick.labelsize"] = tick_label_size
    rcParams["xtick.major.size"] = tick_size
    rcParams["ytick.major.size"] = tick_size
    rcParams["axes.titlesize"] = 0 # no title for paper
    rcParams["axes.labelpad"] = label_pad  # default is 4.0
    rcParams["axes.linewidth"] = axis_line_width
    rcParams["lines.linewidth"] = line_width
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"
    rcParams["lines.antialiased"] = True
    rcParams["savefig.dpi"] = 320


def add_letter_figure(ax, letter, fontsize=15):
    ax.text(-0.1, 1.15, letter, transform=ax.transAxes, fontsize=fontsize
            , fontweight='bold', va='top', ha='right')

##################################