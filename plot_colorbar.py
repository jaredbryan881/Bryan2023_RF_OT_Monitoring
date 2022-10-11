import matplotlib.pyplot as plt
import numpy as np

a = np.array([[0,1]])
plt.figure(figsize=(1, 10))
img = plt.imshow(a, cmap="inferno")
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.1, 0.8, 0.8])
plt.colorbar(orientation="vertical", cax=cax)
plt.savefig("colorbar_inferno.pdf")


a = np.array([[0,1]])
plt.figure(figsize=(1, 10))
img = plt.imshow(a, cmap="coolwarm")
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.1, 0.8, 0.8])
plt.colorbar(orientation="vertical", cax=cax)
plt.savefig("colorbar_coolwarm.pdf")

a = np.array([[0,1]])
plt.figure(figsize=(1, 10))
img = plt.imshow(a, cmap="twilight")
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.1, 0.8, 0.8])
plt.colorbar(orientation="vertical", cax=cax)
plt.savefig("colorbar_twilight.pdf")