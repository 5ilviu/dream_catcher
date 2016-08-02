
from scipy import misc
import matplotlib.pyplot as plt


def get_img(image='./res/cs_1_1.jpg'):
    f = misc.imread(image)
    return f.reshape(24300) / f.max()


def show_img(f):
    plt.show()
    plt.imshow(f)

# print show_img(get_img())

# from sklearn.metrics import confusion_matrix
# y_true = [2, 0, 2, 2, 0, 1]
# y_pred = [0, 0, 2, 2, 0, 2]
# print confusion_matrix(y_true, y_pred)
