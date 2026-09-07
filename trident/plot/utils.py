# trident.plot.utils.py

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def activate_plot_settings():
    if is_notebook():        
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        mpl.rcParams["figure.figsize"] = [6, 4]
        mpl.rcParams["savefig.dpi"] = 200
    else:
        mpl.rcParams["savefig.dpi"] = 400
        
    sns.set_theme(style="white", rc={"grid.color": ".6", "grid.linestyle": ":"})
    sns.set_context("paper", font_scale=1.)
    
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["figure.autolayout"] = False
    mpl.rcParams["legend.frameon"] = True