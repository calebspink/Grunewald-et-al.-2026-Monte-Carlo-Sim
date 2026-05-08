# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 18:46:59 2025

@author: Caleb Spink (with help from Claude)

Notes: 
1. After deciding upon using the median sequence length (31), I tried running a sim with each sequence length.
   No significant difference. The current script reflects this.
   
2. Entropy was exploratory and is calculated upon running the script.

3. To use, replace input data with your own. If there is demand, I could convert this to a console application.

If you have any questions, please email css0121@auburn.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import mannwhitneyu

#Input data
ow_entropy = [0.691761499, 0.691896659,	0.677494404, 0.67652596, 0.692981882, 0.690593986, 0.713089883, 0.689009238, 0.692552532, 0.677494404, 0.691064569,	0.673011667, 0.693147181, 0.686961577,	0.673011667, 0.688138814, 0.693147181, 0.691761499,	0.679193266, 0.689009238, 0.693147181, 0.681624434,	0.655481774, 0.693053351, 0.66248165, 0.674297959, 0.690467809, 0.693147181, 0.688138814, 0.689856499]
LL_conditional_probabilities = [0,	0.2,	0.142857143,	0.538461538,	0.111111111,	0.083333333,		0.111111111,			0,	0.461538462,	0,		0,	0,	0,	0.181818182,	0,	0.125, 0, 0.055555556, 0, 0, 0, 0.142857143, 0.666666667, 0.135135135, 0, 0.083333333, 0, 0.157894737, 0.176470588, 0.058823529]
LR_conditional_probabilities = [1,	0.8,	0.857142857,	0.461538462,	0.888888889,	0.916666667,		0.666666667,			1,	0.538461538,	1,		1,	1,	1,	0.818181818,	1,	0.875,	1,	0.944444444,	1,	1,	1,		0.857142857,	0.333333333,		0.864864865,		1,	0.916666667,	1,	0.842105263,	0.823529412,	0.941176471]
RR_conditional_probabilities = [0.15,	0.157894737,	0.368421053,	0.25,	0.111111111,	0.2,		0.125,			0.166666667,	0.466666667,	0.3,		0.125,	0.444444444,	0,	0.4,	0.25,	0.272727273,	0,	0.105263158,	0.3,	0.166666667,	0,		0.333333333,	0.25,		0.111111111,		0.380952381,	0.378378378,	0.142857143,	0.15,	0.318181818,	0.210526316]
RL_conditional_probabilities = [0.85,	0.842105263,	0.631578947,	0.75,	0.888888889,	0.8,		0.875,			0.833333333,	0.533333333,	0.7,		0.875,	0.555555556,	1,	0.6,	0.75,	0.727272727,	1,	0.894736842,	0.7,	0.833333333,	1,		0.666666667,	0.75,		0.888888889,		0.619047619,	0.621621622,	0.857142857,	0.85,	0.681818182,	0.789473684]


#Calculate entropy
def calculate_entropy(sequence):
    counts = Counter(sequence)
    total = len(sequence)
    entropy = 0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log(p)
    return entropy

#Calculate conditional probabilities
def calculate_conditional_probabilities(sequence):
    transitions = {
        'L->R': 0, 'L->L': 0,
        'R->R': 0, 'R->L': 0
    }
    
    for i in range(len(sequence) - 1):
        current = sequence[i]
        next_state = sequence[i + 1]
        key = f"{current}->{next_state}"
        transitions[key] += 1
    
    l_total = transitions['L->R'] + transitions['L->L']
    r_total = transitions['R->R'] + transitions['R->L']
    
    conditional_probs = {}
    
    if l_total > 0:
        conditional_probs['P(R|L)'] = transitions['L->R'] / l_total
        conditional_probs['P(L|L)'] = transitions['L->L'] / l_total
    else:
        conditional_probs['P(R|L)'] = 0
        conditional_probs['P(L|L)'] = 0
    
    if r_total > 0:
        conditional_probs['P(R|R)'] = transitions['R->R'] / r_total
        conditional_probs['P(L|R)'] = transitions['R->L'] / r_total
    else:
        conditional_probs['P(R|R)'] = 0
        conditional_probs['P(L|R)'] = 0
    
    return transitions, conditional_probs

num_data = 100000
dataset_sizes = [38, 40, 34, 22, 55, 28, 20, 11, 29, 17, 62, 15, 16, 27, 15, 20, 10, 38, 36, 11, 6, 33, 11, 74, 69, 62, 41, 40, 40, 37]

#Store results
all_entropies = []
all_cond_probs = {'P(R|L)': [], 'P(L|L)': [], 'P(R|R)': [], 'P(L|R)': []}

for size in dataset_sizes:  
    for i in range(num_data):
        dataset = np.random.choice(['R', 'L'], size=size, p=[0.5, 0.5])
        ent = calculate_entropy(dataset)
        trans, cond = calculate_conditional_probabilities(dataset)
        
        all_entropies.append(ent)
        for key in all_cond_probs.keys():
            all_cond_probs[key].append(cond[key])

#Convert to numpy arrays
all_entropies = np.array(all_entropies)

print("\n" + "="*50)
print("DESCRIPTIVE STATISTICS FOR ENTROPY")
print("="*50)
print(f"Mean:               {np.mean(all_entropies):.6f}")
print(f"Median:             {np.median(all_entropies):.6f}")
print(f"Standard Deviation: {np.std(all_entropies):.6f}")
print(f"Variance:           {np.var(all_entropies):.6f}")
print(f"Minimum:            {np.min(all_entropies):.6f}")
print(f"Maximum:            {np.max(all_entropies):.6f}")
print(f"25th Percentile:    {np.percentile(all_entropies, 25):.6f}")
print(f"75th Percentile:    {np.percentile(all_entropies, 75):.6f}")
print(f"Range:              {np.max(all_entropies) - np.min(all_entropies):.6f}")
print(f"Skewness:           {((all_entropies - np.mean(all_entropies))**3).mean() / (np.std(all_entropies)**3):.6f}")

print("\n" + "="*50)
print("DESCRIPTIVE STATISTICS FOR CONDITIONAL PROBABILITIES")
print("="*50)
for prob_name, prob_values in all_cond_probs.items():
    prob_array = np.array(prob_values)
    print(f"\n{prob_name}:")
    print(f"  Mean:               {np.mean(prob_array):.6f}")
    print(f"  Median:             {np.median(prob_array):.6f}")
    print(f"  Standard Deviation: {np.std(prob_array):.6f}")
    print(f"  Minimum:            {np.min(prob_array):.6f}")
    print(f"  Maximum:            {np.max(prob_array):.6f}")

#Mann-Whitney U Tests
print("\n" + "="*50)
print("MANN-WHITNEY U TESTS")
print("="*50)

print("\nEntropy:")
u_stat, p_value = mannwhitneyu(all_entropies, ow_entropy, alternative='two-sided')
print(f"  U-statistic: {u_stat:.2f}")
print(f"  P-value: {p_value:.6f}")
print(f"  Significant at α=0.025: {'Yes' if p_value < 0.025 else 'No'}")

input_probs = {
    'P(L|L)': LL_conditional_probabilities,
    'P(L|R)': LR_conditional_probabilities,
    'P(R|L)': RL_conditional_probabilities,
    'P(R|R)': RR_conditional_probabilities
}

for prob_name in ['P(L|L)', 'P(L|R)', 'P(R|L)', 'P(R|R)']:
    print(f"\n{prob_name}:")
    u_stat, p_value = mannwhitneyu(all_cond_probs[prob_name], input_probs[prob_name], alternative='two-sided')
    print(f"  U-statistic: {u_stat:.2f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Significant at α=0.025: {'Yes' if p_value < 0.025 else 'No'}")

#Create entropy visualization with overlay
fig1, ax = plt.subplots(1, 1, figsize=(12, 7))

e_bins = np.arange(0, 0.725, 0.01)

#Simulation histogram
counts, bins, patches = ax.hist(all_entropies, bins=e_bins, edgecolor='black', alpha=0.6, 
        label='Simulation', color='#1D72F2', weights=np.ones(len(all_entropies)) / len(all_entropies))

#Overlay input data histogram
counts2, bins2, patches2 = ax.hist(ow_entropy, bins=e_bins, edgecolor='blue', alpha=0.4, 
        label='Ophelia', color='#DD550C', weights=np.ones(len(ow_entropy)) / len(ow_entropy))

#Info lines
ax.axvline(np.mean(all_entropies), color='red', linestyle='--', 
           linewidth=2, label=f'Simulation Mean: {np.mean(all_entropies):.4f}')
ax.axvline(0.693147, color='green', linestyle='-',
           linewidth=2, label='Theoretical Value: 0.693147')
ax.axvline(np.mean(ow_entropy), color='purple', linestyle='--',
           linewidth=2.5, label=f'Ophelia Mean: {np.mean(ow_entropy):.4f}')

ax.set_xlabel('Entropy', fontsize=12)
ax.set_ylabel('Relative Frequency', fontsize=12)
ax.set_title('Entropy: Simulation vs Ophelia', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(False)
plt.tight_layout()
plt.show()

#Create conditional probabilities visualization
fig2, axes = plt.subplots(2, 2, figsize=(14, 11))
fig2.suptitle('Conditional Probabilities: Simulation vs Ophelia', fontsize=16, fontweight='bold')

prob_names = ['P(L|L)', 'P(L|R)', 'P(R|L)', 'P(R|R)']
positions = [(0, 0), (0, 1), (1, 1), (1, 0)]

#Define bins with 0.05 width
rf_bins = np.arange(0, 1.05, 0.05)

for prob_name, (row, col) in zip(prob_names, positions):
    prob_array = np.array(all_cond_probs[prob_name])
    input_data = input_probs[prob_name]
    
    #Simulation histogram
    axes[row, col].hist(prob_array, bins=rf_bins, edgecolor='black', alpha=0.6, 
                       label='Simulation', color='#1D72F2', density=False, weights=np.ones(len(prob_array))/len(prob_array))
    
    #Overlay OW histogram
    axes[row, col].hist(input_data, bins=rf_bins, edgecolor='black', alpha=0.4, 
                       label='Ophelia', color='#DD550C', density=False, weights=np.ones(len(input_data))/len(input_data))
    
    #Info lines for relative frequency
    axes[row, col].axvline(np.mean(prob_array), color='red', linestyle='--', 
                          linewidth=2, label=f'Sim Mean: {np.mean(prob_array):.4f}')
    axes[row, col].axvline(0.5, color='green', linestyle='-',
                           linewidth=2, label='Theoretical: 0.5')
    axes[row, col].axvline(np.mean(input_data), color='purple', linestyle='--',
                           linewidth=2.5, label=f'Ophelia Mean: {np.mean(input_data):.4f}')
    
    axes[row, col].set_xlabel('Probability', fontsize=11)
    axes[row, col].set_ylabel('Relative Frequency', fontsize=11)
    axes[row, col].set_title(f'{prob_name}', fontsize=12, fontweight='bold')
    axes[row, col].legend(fontsize=9)
    axes[row, col].grid(False)
    axes[row, col].set_ylim([0, 0.5])
    
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Expected entropy for random 50/50 sequences: ~0.693 (using natural log)")
print(f"Observed mean entropy: {np.mean(all_entropies):.6f}")
print(f"Input data mean entropy: {np.mean(ow_entropy):.6f}")
print(f"\nExpected conditional probabilities: ~0.5 for all transitions")
print(f"Observed mean P(R|L): {np.mean(all_cond_probs['P(R|L)']):.6f}")
print(f"Input data mean P(R|L): {np.mean(input_probs['P(R|L)']):.6f}")