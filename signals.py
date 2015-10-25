from sklearn import svm
import numpy as np

kernels = [('poly', 1), ('poly', 2), ('poly', 3), ('poly', 4), ('sigmoid', 0), ('rbf', 0)]
degree=3
gamma=0.0
coef0=0.0
tol=0.001
C=1.0

x = [map(float, line.strip().split(' ')) for line in open('./Train/X_train.txt', 'r').readlines()]

y = map(int, [line.strip() for line in open('./Train/Y_train.txt', 'r').readlines()])

xValidate = [map(float, line.strip().split(' ')) for line in open('./Test/X_test.txt', 'r').readlines()]

yValidate = map(int, [line.strip() for line in open('./Test/Y_test.txt', 'r').readlines()])


for kernel, degree in kernels:

	print '----------Kernel : ', kernel, '----------\n'
	motionPredictor = None
	if degree != 0:
		print 'Degree : ', degree
		motionPredictor = svm.SVC(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C).fit(x, y)
	else:
		motionPredictor = svm.SVC(kernel=kernel, gamma=gamma, coef0=coef0, tol=tol, C=C).fit(x, y)

	motionCorrect, motionIncorrect , motionFalsePositive, motionFalseNegative, totalmotion = 0,0,0,0,0
	for xVal, yVal in zip(xValidate, yValidate):
		predicted = motionPredictor.predict(xVal)[0]
		if predicted == yVal:
			motionCorrect += 1
		else:
			motionIncorrect += 1
		totalmotion += 1

	print '\nMotion Analysis : \n'
	print 'Correct : {0:.2f} ({1:.2f} %)\nIncorrect : {2:.2f} ({3:.2f} %)\n'.format(motionCorrect, 100.0 * float(motionCorrect) / float(totalmotion), motionIncorrect, 100.0 * float(motionIncorrect) / float(totalmotion))
