from bing_loader import BingLoader
import matplotlib.pyplot as plt

loader = BingLoader('/home/krakapwa/data/bing')

sample = loader[8]

print(sample['label/nodes'])

# poly and nodes are in (i, j) form
plt.imshow(sample['image'])
plt.scatter(sample['label/nodes'][:,0], sample['label/nodes'][:,1])
plt.show()
