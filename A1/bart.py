import matplotlib.pyplot as plt

# Data for F18 from the table
methods = ['balanced weights bias', 'balanced weights', 'normalized weights', 'dynamic weights', 'adaptive_weights']
best_reached = [4.72, 4.72, 4.86, 4.86, 4.86]
mean_reached = [2.67, 2.67, 2.63, 2.63, 2.63]
# median_reached = [2.4, 2.4, 2.4, 2.4, 2.4]

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Bar width
bar_width = 0.25
index = range(len(methods))

# Plot the bars
bar1 = ax.bar(index, best_reached, bar_width, label='Best Reached')
bar2 = ax.bar([i + bar_width for i in index], mean_reached, bar_width, label='Mean Reached')
# bar3 = ax.bar([i + 2 * bar_width for i in index], median_reached, bar_width, label='Median Reached')

# Set the labels and title
ax.set_xlabel('Methods')
ax.set_ylabel('Values')
ax.set_title('Performance of Different Methods on F18')
ax.set_xticks([i + bar_width for i in index])
ax.set_xticklabels(methods, ha='right')
ax.legend()
plt.show()
#​⬤