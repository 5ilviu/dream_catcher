from numpy.random.mtrand import permutation
from sklearn.linear_model import SGDClassifier

import dirty_importer
import import_data

if __name__ == '__main__':
    dataset = dirty_importer.get_dataset()
    data = dataset[0]
    label = dataset[1]
    perm = permutation(len(data))
    data = data[perm]
    label = label[perm]
    clf = SGDClassifier(loss="log", penalty="l1")
    clf.fit(data, label)

    testset = import_data.get_dataset('test')
    data = testset[0]
    label = testset[1]
    perm = permutation(len(data))
    data = data[perm]
    label = label[perm]
    predict = []

    bingo = 0
    for i in range(0, len(data)):
        predict.append(clf.predict(data[i].reshape(1, -1)))
        if predict[i] == label[i]:
            bingo += 1
    print bingo
    print 'correct: {}%'.format((bingo*1./len(data)*100))

    from sklearn.metrics import confusion_matrix
    print confusion_matrix(label, predict)
