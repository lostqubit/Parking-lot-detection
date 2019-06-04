import tensorflow as tf

def create_placeholders(in_height,in_width,in_channels):

	X = tf.placeholder(tf.float32, [None, in_height, in_width, in_channels])
	Y = tf.placeholder(tf.float32, [None, 2])
	
	return X, Y

def initialize_parameters():
	
	tf.set_random_seed(1)
	W1 = tf.get_variable("W1", [11,11,3,16], initializer=tf.contrib.layers.xavier_initializer(seed=0),regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005/2.0))
	W2 = tf.get_variable("W2", [5,5,16,20], initializer=tf.contrib.layers.xavier_initializer(seed=0),regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005/2.0))
	W3 = tf.get_variable("W3", [3,3,20,30], initializer=tf.contrib.layers.xavier_initializer(seed=0),regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005/2.0))
	parameters={"W1":W1,"W2":W2,"W3":W3}

	return parameters


def forward_propagation(X,parameters):

	W1 = parameters['W1']
	W2 = parameters['W2']
	W3 = parameters['W3']

	# conv1
	Z1 = tf.nn.conv2d(X, filter=W1, strides=[1,4,4,1], padding='SAME')
	A1 = tf.nn.relu(Z1)
	A1_norm = tf.nn.local_response_normalization(A1, depth_radius=5, bias=2, alpha=0.0001, beta=0.75)
	P1 = tf.nn.max_pool(A1_norm, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

	# conv2
	Z2 = tf.nn.conv2d(P1, filter=W2, strides=[1,1,1,1], padding='SAME')
	A2 = tf.nn.relu(Z2)
	A2_norm = tf.nn.local_response_normalization(A2, depth_radius=5, bias=2, alpha=0.0001, beta=0.75)
	P2 = tf.nn.max_pool(A2_norm, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

	# conv3
	Z3 = tf.nn.conv2d(P2, filter=W3, strides=[1,1,1,1], padding='SAME')
	A3 = tf.nn.relu(Z3)
	P3 = tf.nn.max_pool(A3, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

	# Flatten
	P3 = tf.contrib.layers.flatten(P3)

	# fully connected layer
	Z4 = tf.contrib.layers.fully_connected(P3,48,weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005/2.0),activation_fn=None)
	A4 = tf.nn.relu(Z4)

	# Output layer
	Z5 = tf.contrib.layers.fully_connected(A4,2,weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005/2.0),activation_fn=None)

	return Z5

def compute_cost(Z5,Y):

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z5,labels=Y))
	l2_loss = tf.losses.get_regularization_loss()
	cost += l2_loss
	return cost

