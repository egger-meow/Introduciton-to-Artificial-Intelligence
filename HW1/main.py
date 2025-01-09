import dataset
import model
import detection
import matplotlib.pyplot as plt
def acc(a,b):

    l = len(a)    
    count = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            count += 1
    return float(count/l)
# Part 1: Implement loadImages function in dataset.py and test the following code.
print('Loading images')
train_data = dataset.load_images('data/train')
print(f'The number of training samples loaded: {len(train_data)}')
test_data = dataset.load_images('data/test')
print(f'The number of test samples loaded: {len(test_data)}')

print('Show the first and last images of training dataset')
fig, ax = plt.subplots(1, 2)
ax[0].axis('off')
ax[0].set_title('Car')
ax[0].imshow(train_data[1][0], cmap='gray')
ax[1].axis('off')
ax[1].set_title('Non car')
ax[1].imshow(train_data[-1][0], cmap='gray')
plt.show()

# Part 2: Build and train 3 kinds of classifiers: KNN, Random Forest and Adaboost.
# Part 3: Modify difference values at parameter n_neighbors of KNeighborsClassifier, n_estimators 
# of RandomForestClassifier and AdaBoostClassifier, and find better results.
# car_clf = model.CarClassifier(
#     model_name="RF", # KNN, RF (Random Forest) or AB (AdaBoost)
#     train_data=train_data,
#     test_data=test_data
# )
truth = []
truthCount = []
f = open('GroundTruth.txt', mode = 'r')
for i in range(50):
    l = f.readline()
    c = l.count('1')   
    truthCount.append(c)
    truth.append(l.split(' '))



car_clf = model.CarClassifier(
    model_name="KNN", # KNN, RF (Random Forest) or AB (AdaBoost)
    train_data=train_data,
    test_data=test_data
)

car_clf.train()
car_clf.eval()

# Part 4: Implement detect function in detection.py and test the following code.



occ1 = detection.detect('data/detect/detectData.txt', car_clf)

acc1 = []
f = open('ML_Models_pred.txt', mode = 'r')
for i in range(50):
    l = f.readline()
    acc1.append(l.split(' '))
acc1 = list(map(lambda a,b:acc(a,b),acc1,truth))
f.close()



car_clf = model.CarClassifier(
    model_name="RF", # KNN, RF (Random Forest) or AB (AdaBoost)
    train_data=train_data,
    test_data=test_data
)

car_clf.train()
car_clf.eval()
occ2 = detection.detect('data/detect/detectData.txt', car_clf)
acc2 = []
f = open('ML_Models_pred.txt', mode = 'r')
for i in range(50):
    l = f.readline()
    acc2.append(l.split(' '))
acc2 = list(map(lambda a,b:acc(a,b),acc2,truth))
f.close()

car_clf = model.CarClassifier(
    model_name="AB", # KNN, RF (Random Forest) or AB (AdaBoost)
    train_data=train_data,
    test_data=test_data
)

car_clf.train()
car_clf.eval()
occ3 = detection.detect('data/detect/detectData.txt', car_clf)
acc3 = []
f = open('ML_Models_pred.txt', mode = 'r')
for i in range(50):
    l = f.readline()
    acc3.append(l.split(' '))
acc3 = list(map(lambda a,b:acc(a,b),acc3,truth))
f.close()

import matplotlib.pyplot as plt



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


ax1.plot(truthCount, label = 'Ground Truth', linestyle='-')
ax1.plot(occ1, label = 'KNN', linestyle='-')
ax1.plot(occ2, label = 'Random Forest', linestyle='-')
ax1.plot(occ3, label = 'AdaBoost', linestyle='-')
ax1.set_ylabel('#cars')
ax1.set_xlabel('Time Slot')
ax1.set_title('Parking Slots Occupation')
ax1.legend()



ax2.plot(acc1, label = 'KNN', linestyle='-')
ax2.plot(acc2, label = 'Random Forest', linestyle='-')
ax2.plot(acc3, label = 'AdaBoost', linestyle='-')
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Time Slot')
ax2.set_title('Accuracy of the Models')
ax2.legend()




plt.tight_layout()
plt.show()
    