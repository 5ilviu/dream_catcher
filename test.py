from numpy.random.mtrand import permutation
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import import_data

if __name__ == '__main__':
    global visual
    dataset = import_data.get_dataset()
    data = dataset[0]
    label = dataset[1]
    # data = np.resize(data, (50000, 784))
    # perm = permutation(50000)
    # data = data[perm]
    perm = permutation(len(data))
    data = data[perm]
    label = label[perm]
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(data, label)

    testset = import_data.get_dataset('test')
    predict = []

    bingo = 0
    for i in range(0, len(testset[0])):
        predict.append(clf.predict(testset[0][i].reshape(1, -1)))
        if predict[i] == testset[1][i]:
            bingo += 1
    print bingo
    print 'procente: {}%'.format((bingo*1./len(testset[0])*100))

    from sklearn.metrics import confusion_matrix
    print confusion_matrix(testset[1], predict)
