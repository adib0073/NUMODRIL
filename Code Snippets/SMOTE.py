print("Before SMOTE, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before SMOTE, counts of label '0': {} \n".format(sum(y_train == 0))) 

# import SMOTE module from imblearn library 
# pip install imblearn (if you don't have imblearn in your system) 
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
x_train_res, y_train_res = sm.fit_sample(x_train.reshape(x_train.shape[0], -1), y_train) 

print('After SMOTE, the shape of train_X: {}'.format(x_train_res.shape)) 
print('After SMOTE, the shape of train_y: {} \n'.format(y_train_res.shape)) 

print("After SMOTE, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After SMOTE, counts of label '0': {}".format(sum(y_train_res == 0))) 
