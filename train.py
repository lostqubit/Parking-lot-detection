import argparse
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from preprocessing import processing
from mAlexNet import create_placeholders, initialize_parameters, forward_propagation, compute_cost
from utils import random_mini_batches


def set_hparams():

	epochs=18
	batch_size=64
	learning_rate= 0.01
	momentum = 0.9
	weight_decay=0.0005

	hparams = {"epochs":epochs, "batch_size":batch_size, "lr":learning_rate, "momentum":momentum, "weight_decay":weight_decay}

	return hparams


def main():

	tf.reset_default_graph()
	tf.set_random_seed(1)
	seed=3
	
	df = pd.read_csv('train_set.csv')
	X_train = df['path_to_image'].get_values()
	Y_train = df.drop(columns=['path_to_image']).get_values()
	m = len(X_train)
	n_h = 224 
	n_w = 224
	n_c = 3

	hparams= set_hparams()
	costs=[]
	epoch_costs=[]

	X,Y = create_placeholders(n_h,n_w,n_c)

	parameters = initialize_parameters()

	output = forward_propagation(X,parameters)

	cost = compute_cost(output,Y)

	global_step = tf.Variable(0, trainable=False)
	starter_learning_rate = hparams['lr']
	decayed_learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           19885, 0.5, staircase=True)

	optimizer = tf.train.MomentumOptimizer(learning_rate=decayed_learning_rate,momentum=0.9,use_nesterov=False).minimize(cost,global_step=global_step)

	init = tf.global_variables_initializer()

	saver = tf.train.Saver()


	with tf.Session() as sess:

		sess.run(init)

		for epoch in range(hparams['epochs']):
			minibatch_cost = 0

			minibatch_size = hparams['batch_size']
			num_minibatches = int(m/minibatch_size)
			seed=seed+1
			minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed)

			for minibatch in minibatches:
				(minibatch_X,minibatch_Y) = minibatch
				minibatch_X = processing(minibatch_X)

				temp_cost, _ = sess.run([cost,optimizer],feed_dict={X:minibatch_X,Y:minibatch_Y})
				minibatch_cost += temp_cost / num_minibatches

				if global_step%1 == 0:
					costs.append(minibatch_cost)
					print('Cost after iteration %i: %f' % (global_step,minibatch_cost))

			if epoch%1 ==0:
				epoch_costs.append(minibatch_cost)
				print('Cost after epoch %i: %f' % (epoch, minibatch_cost))
				save_path = saver.save(sess, 'model.ckpt',global_step=global_step)
				print('Model saved in path: %s' % save_path)

		


		plot_cost(costs)

		#predict_op = tf.argmax(output, 1)
		#correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
		# Calculate accuracy on the test set
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		#print(accuracy)
		#train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
		#test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
		#print("Train Accuracy:", train_accuracy)
		#print("Test Accuracy:", test_accuracy)


if __name__ == '__main__':
  main()














