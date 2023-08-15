from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from qreservoir.datasets.Dataset import Dataset, TrainTestSplit
from sklearn.model_selection import train_test_split


class Circles(Dataset):
    def __init__(self, noise: float = 0.5, size: int = 200) -> None:
        self.X, self.y = make_circles(noise=noise, factor=0.3, n_samples=size)

    def get_train_test(self, test_size: float = 0.3) -> TrainTestSplit:
        return train_test_split(self.X, self.y, test_size=test_size, random_state=42)

    def show(self) -> None:
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5

        # just plot the dataset first
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        plt.title("Input data")

        # Plot the training points
        plt.scatter(
            self.X[:, 0], self.X[:, 1], c=self.y, cmap=cm_bright, edgecolors="k"
        )

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()


if __name__ == "__main__":
    d = Circles(noise=0.5, size=1000)
    d.show()
