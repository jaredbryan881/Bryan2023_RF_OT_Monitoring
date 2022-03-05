import matplotlib.pyplot as plt
import numpy as np

def main():
	savefig=False
	
	a = np.array([[-2,2]])
	plt.figure(figsize=(0.2, 4.3))
	img = plt.imshow(a, cmap="coolwarm")
	plt.gca().set_visible(False)
	cax = plt.axes([0.1, 0.2, 0.8, 0.6])
	cbar = plt.colorbar(orientation="vertical", cax=cax)
	if savefig:
		plt.savefig("colorbar.pdf")
	plt.show()

if __name__=="__main__":
	main()