import matplotlib.pyplot as plt
import numpy as np

# Beispiel-Dictionaries mit Häufigkeitsverteilungen aus mehreren Experimenten
experiment1 = {'A': 10, 'B': 15, 'C': 5, 'D': 20, 'E': 10, 'F': 5}
experiment2 = {'A': 5, 'B': 10, 'C': 15, 'D': 10}
experiment3 = {'A': 20, 'B': 5, 'C': 10, 'D': 15}
experiment4 = {'A': 10, 'B': 20, 'C': 10, 'D': 5, 'E': 15}

# Kombinieren der Dictionaries in eine Liste
experiments = [experiment1, experiment2, experiment3, experiment4]

# Alle eindeutigen Schlüssel extrahieren
all_keys = set()
for experiment in experiments:
    all_keys.update(experiment.keys())

# Schlüssel sortieren, um eine konsistente Reihenfolge beizubehalten
all_keys = sorted(all_keys)

# Daten für das Plotten vorbereiten
data = {key: [0] * len(experiments) for key in all_keys}
for i, experiment in enumerate(experiments):
    for key in all_keys:
        data[key][i] = experiment.get(key, 0)

# Breite der Balken
bar_width = 0.18

# Positionen der Balken auf der x-Achse
r = np.arange(len(all_keys))

# Plotten der Balken für jedes Experiment
for i in range(len(experiments)):
    values = [data[key][i] for key in all_keys]
    plt.bar(r + i * (bar_width+0.02), values,
            width=bar_width, label=f'Experiment {i+1}')

# Achsenbeschriftungen und Titel hinzufügen
plt.xlabel('Ausprägung')
plt.ylabel('Anzahl')
plt.title('Histogramm der Häufigkeitsverteilungen')
plt.xticks(r + bar_width * (len(experiments) - 1) / 2, all_keys)
plt.legend()

# Plot anzeigen
plt.show()
