import sys
sys.path.append('..')

from src.plotter import plot_time_series, plot_bar_chart, animation_bar_chart


csv_path = 'history.csv'

plot_time_series(csv_path, 'regret', 'Regret', cumsum=True, save=True)

xticklabels = ['Uniform', 'Stubborn 1', 'Stubborn 2', 'Stubborn 3',
               'No Action', 'WOFOST Randomized', 'WOFOST SP', 'WOFOST BP',
               'WOFOST Complement', 'WOFOST Min', 'WOFOST Max', 'WOFOST Max SP', 'WOFOST Max BP']
plot_bar_chart(csv_path, xticklabels=xticklabels, save=True)
animation_bar_chart(csv_path, xticklabels=xticklabels, save=True)
