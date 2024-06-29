import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 4))

x = [2, 3, 4]
width_column = 0.3
x_merge = [i - width_column/2 for i in x]
x_ours = [i + width_column/2 for i in x]

y_merge_dev = [19.5, 19.6, 19.6]
y_merge_test = [21.0, 20.5, 20.8]

y_ours_dev = [19.6, 18.8, 19.1]
y_ours_test = [20.0, 20.3, 21.0]

ax1 = fig.add_subplot(1,2,1)
plt.bar(x_merge, y_merge_dev, width = width_column, label = 'Merge')
plt.bar(x_ours, y_ours_dev, width = width_column, label = 'Ours')
plt.ylim(18.0, 20.0)

ax1 = fig.add_subplot(1,2,2)
plt.bar(x_merge, y_merge_test, width = width_column, label = 'Merge')
plt.bar(x_ours, y_ours_test, width = width_column, label = 'Ours')
# plt.yticks([18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0]) 
plt.ylim(18.0, 22.0)


plt.show()
plt.savefig('layers.png')