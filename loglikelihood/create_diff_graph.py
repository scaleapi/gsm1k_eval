import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

include = set()

diffs = {}
with open('scale_diff.csv') as file:
    data = csv.reader(file)
    for row in data:
        diffs[row[0]] = row[1]

averages = {}
with open('averages.tsv') as file:
    data = csv.reader(file, delimiter='\t')
    for row in data:
        averages[row[0]] = row[1]

# set seaborn
sns.set()

# remove items in diffs are aren't in averages
not_diffs = {k: v for k, v in diffs.items() if k not in averages}
print(not_diffs)
diffs = {k: v for k, v in diffs.items() if k in averages}
# and the other way
averages = {k: v for k, v in averages.items() if k in diffs}
diffs = sorted(diffs.items())
averages = sorted(averages.items())
for avg, diff in zip(averages, diffs):
    print(avg[0], diff[0], avg[1], diff[1])

diffs_x = [float(x[1]) for x in diffs]
averages_y = [float(x[1]) for x in averages]

# make models different colors, pick a hue scheme
colors = plt.cm.get_cmap("tab20").colors
model_colors = {
    "CodeLlama": colors[2],
    "Llama": colors[3],
    "tral": colors[0], # mix/mistral
    "Smaug": colors[5],
    "Xwin-Math": colors[6],
    "Yi": colors[7],
    "dbrx": colors[8],
    "deepseek": colors[9],
    "falcon": colors[11],
    "gemma": colors[10],
    "gpt-neox": colors[12],
    "gpt2-xl": colors[13],
    "grok": colors[14],
    "llemma": colors[15],
    "math-shepherd": colors[16],
    "phi": colors[1],
    "Phi-3": colors[17],
    "pythia": colors[18],
    "vicuna": colors[19]
}

def get_model_color(model_name):
    for key in model_colors:
        if key in model_name:
            return key, model_colors[key]
    print('No color found for', model_name)
    return None

from collections import defaultdict

colors_to_counts = defaultdict(int)

symbols = ['o', '^', 'p', 'P', 'X', 'D', 'd', 'v', '<', '>', 's', '*', 'h', 'H']

code_store = defaultdict(dict)
for i in range(len(diffs)):
    key, model_color = get_model_color(diffs[i][0])
    # model_color += colors_to_counts[key]

    plt.scatter(diffs_x[i], averages_y[i], color=model_color, edgecolor="black", linewidth=0.5, marker=symbols[colors_to_counts[key]])
    colors_to_counts[key] += 1
    code_store[diffs[i][0]]["rgb"] = model_color
    code_store[diffs[i][0]]["marker"] = symbols[colors_to_counts[key]]
    # make the markers smaller

plt.rcParams["legend.markerscale"] = 0.7

# write color_store to json pretty
with open('colors.json', 'w') as file:
    json.dump(code_store, file, indent=4)

# create axis labels, title
plt.xlabel('Overfit on GSM8k')
plt.ylabel('Average Character-Normalized Loglikelihood')

# added dotted trendline
z = np.polyfit(diffs_x, averages_y, 1)
p = np.poly1d(z)
# line style is solid line
plt.plot(diffs_x, p(diffs_x), "-")

plt.grid(True, linestyle="solid", linewidth=0.5, alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=12)

plt.savefig('ll_graph_nolegend.png', dpi=300, bbox_inches="tight")

# y axis should go from -0.4 to -1.0
plt.ylim(-1, -0.4)

# create legend for model names on the bottom right
plt.legend(
            [x[0] for x in diffs],
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            fontsize=4,
            title="Model",
            title_fontsize="medium"
            )

plt.savefig('ll_graph.png', dpi=300, bbox_inches="tight")