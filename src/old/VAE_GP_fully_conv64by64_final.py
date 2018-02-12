import tensorflow as tf
import numpy as np
from DataHandler import DataHandler, plot_data_tc
# if on VM comment out the following line:
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

def trans_conv2d(x, W, padding='SAME', special_shape=False):
	x_shape = tf.shape(x)
	w_shape = tf.shape(W)
	if special_shape:
		out_shape = tf.stack([x_shape[0], 7, 7, w_shape[2]])
	else:
		out_shape = tf.stack([x_shape[0], 2*x_shape[1], 2*x_shape[2], w_shape[2]])
	return tf.nn.conv2d_transpose(x, W, out_shape, strides=[1,2,2,1], padding=padding)

def vae_encode(x):
	x = tf.reshape(x, [-1,64,64,1])
	w_in_1 = weight_variable([3,3,1,16])
	b_in_1 = bias_variable([16])
	encode_1 = tf.nn.relu(conv2d(x,w_in_1) + b_in_1) #now 32x32x16

	w_in_2 = weight_variable([3,3,16,32])
	b_in_2 = bias_variable([32])
	encode_2 = tf.nn.relu(conv2d(encode_1,w_in_2)+b_in_2) #now 16x16x32

	w_in_3 = weight_variable([3,3,32,64])
	b_in_3 = bias_variable([64])
	encode_3 = tf.nn.relu(conv2d(encode_2,w_in_3)+b_in_3) #now 8x8x64

	w_in_4 = weight_variable([3,3,64,128])
	b_in_4 = bias_variable([128])
	encode_4 = tf.nn.relu(conv2d(encode_3,w_in_4)+b_in_4) #now 4x4x128

	w_in_5 = weight_variable([3,3,128,256])
	b_in_5 = bias_variable([256])
	encode_5 = tf.nn.relu(conv2d(encode_4,w_in_5)+b_in_5) #now 2x2x256

	w_in_6 = weight_variable([2,2,256,512])
	b_in_6 = bias_variable([512])
	encode_6 = tf.nn.relu(conv2d(encode_5,w_in_6)+b_in_6) #now 1x1x256
	flat_6 = tf.layers.flatten(encode_6)
	
	#latent space mean
	w_mean = weight_variable([512,100])
	b_mean = bias_variable([100])
	encode_mean = tf.matmul(flat_6, w_mean)+b_mean
	# latent space log variance 
	w_var = weight_variable([512,100])
	b_var = bias_variable([100])
	encode_log_var = tf.matmul(flat_6, w_var)+b_var #1x1x20
	return encode_mean, encode_log_var

def vae_sample(mean, log_variance, kernel_mat, sequence_lengths):
	# keep a vector tell you the time length of all sequences in the batch
	num_samples = sum(sequence_lengths)
	mean = tf.reshape(mean, [-1, 100])
	log_variance = tf.reshape(log_variance, [-1, 100])
	var = tf.exp(log_variance)
	noise = tf.random_normal([num_samples,100])
	sample = mean + tf.sqrt(var)*noise
	return sample
	# kl = 0.5*tf.reduce_sum(1.0+tf.log(var)-tf.square(mean)-var, 1)

def calc_gp_kl(mean, log_variance, kernel_mat):
	#split the matricies so each row is a sequence: ie 100x100 --> 500x20 
	# ideally would have a better way to do this =(
	var = tf.exp(log_variance)
	mean_t_kl = break_mat(mean)
	var_t_kl = break_mat(var)
	log_var_t_kl = break_mat(log_variance)

	gp_kl_full = []
	# get the inverse and determinant of the kernel matrix
	inv_k = tf.matrix_inverse(kernel_mat)
	det_k = tf.matrix_determinant(kernel_mat)
	for m, v, log_v in zip(mean_t_kl, var_t_kl, log_var_t_kl):
		gp_kl_full.append(gp_kl_div(m, v, inv_k, kernel_mat, det_k, log_v))
	gp_kl = tf.stack(gp_kl_full)
	gp_kl_sum = tf.reduce_sum(gp_kl)
	# gp_kl = tf.reshape(gp_kl, [5,100])
	# gp_kl_sum = tf.reduce_sum(tf.reduce_sum(gp_kl, 1),0)
	return gp_kl_sum, gp_kl

def break_mat(mat):
	mat_T = tf.transpose(mat)
	split0_mat, split1_mat, split2_mat, split3_mat, split4_mat = tf.split(mat_T, num_or_size_splits=5, axis=1)
	mat_GP = tf.concat([split0_mat,split1_mat,split2_mat,split3_mat,split4_mat], 0)
	mat_unpack = tf.unstack(mat_GP)
	return mat_unpack

def gp_kl_div(mean, var, inv_K, kernel_mat, det_k, log_var):
	mean = tf.reshape(mean, [20,1])
	var = tf.reshape(var, [20])
	log_var = tf.reshape(log_var, [20])
	diag_var = tf.diag(var)	
	negative_mean = tf.scalar_mul(-1.0,mean)
	p_1 = tf.trace(tf.matmul(inv_K,diag_var)) 
	p_2 = tf.log(det_k) - tf.reduce_sum(log_var)
	p_3a = tf.matmul(inv_K,negative_mean)
	p_3 = tf.matmul(tf.transpose(negative_mean), p_3a) #removed squeeze
	p_4 = p_1 - 20.0 + p_2 + p_3
	p_5 = tf.scalar_mul(0.5, p_4)
	return p_5
	
def vae_decode(X):
	X = tf.reshape(X, [-1,100]) 
	w1 = weight_variable([100, 512])
	b1 = bias_variable([512])
	out_1 = tf.nn.relu(tf.matmul(X,w1) + b1)
	out_1_3d = tf.reshape(out_1, [-1,1,1,512]) #1x1x512

	w_deconv_0 = weight_variable([2,2,256,512])
	b_deconv_0 = bias_variable([256])
	out_0_deconv = tf.nn.relu(trans_conv2d(out_1_3d, w_deconv_0) + b_deconv_0) #2x2x256

	w_deconv_1 = weight_variable([3,3,128,256])
	b_deconv_1 = bias_variable([128])
	out_1_deconv = tf.nn.relu(trans_conv2d(out_0_deconv, w_deconv_1) + b_deconv_1) #4x4x128

	w_deconv_2 = weight_variable([3,3,64,128])
	b_deconv_2 = bias_variable([64])
	out_2_deconv = tf.nn.relu(trans_conv2d(out_1_deconv, w_deconv_2) + b_deconv_2) #8x8x64

	w_deconv_3 = weight_variable([3,3,32,64])
	b_deconv_3 = bias_variable([32])
	out_3_deconv = tf.nn.relu(trans_conv2d(out_2_deconv, w_deconv_3) + b_deconv_3) #16x16x32

	w_deconv_4 = weight_variable([3,3,16,32])
	b_deconv_4 = bias_variable([16])
	out_4_deconv = tf.nn.relu(trans_conv2d(out_3_deconv, w_deconv_4) + b_deconv_4) #32x32x16

	w_deconv_5 = weight_variable([3,3,1,16])
	b_deconv_5 = bias_variable([1])
	decode = tf.layers.flatten(tf.sigmoid(trans_conv2d(out_4_deconv, w_deconv_5) + b_deconv_5)) #64x64x1
	return decode

def kernel_matrix(T, time_char):
	k_mat = np.zeros((T, T), dtype=np.float32)
	def kernel_function(t1,t2, time_char):
		noise = 1e-3
		if t1 != t2:
			noise = 0.0
		signal = 1.0-noise
		return signal*np.exp(-np.power((float(t1)-float(t2)),2)/(2.0*np.power(time_char,2))) + noise
	ax_1, ax_2 = k_mat.shape
	for i in range(ax_1):
		for j in range(ax_2):
			k_mat[i,j] = kernel_function(i,j, time_char)
	return k_mat
 
def main():
	# data path
	#on local comp
	data_path = "/Users/ethanevans/Documents/Externship/Data/"
	# on VM:
	# data_path = "/home/ethan/Data/"
	data_file = "mnist_test_seq.npy"
	# batch size - IMPORTANT this is the number of sequences! Not the number of images
	batch_size = 5 #each sequence here is 20 steps --> 100 images 
	#load the data 
	MovingMnist = DataHandler(data_path, data_file, batch_size=batch_size)
	# MovingMnist.make_shuffled_dataset() # comment out for sequence training

	#define the kernel matrix for the GP prior. Current all latent variables will have the same K
	T = 20
	time_char = 1.0 
	sequence_lengths = [20,20,20,20,20]
	K = kernel_matrix(T, time_char)

	# make the model
	x = tf.placeholder(tf.float32, [None,4096], name="x")
	latent_mean, latent_log_variance = vae_encode(x)
	latent_mean = tf.identity(latent_mean, name='latent_mean')
	latent_log_variance = tf.identity(latent_log_variance, name='latent_variance')

	latent_sample = vae_sample(latent_mean, latent_log_variance, K, sequence_lengths)
	sum_gp_kl, gp_kl = calc_gp_kl(latent_mean, latent_log_variance, K)
	latent_sample = tf.identity(latent_sample, name='latent_sample')
	sum_gp_kl = tf.identity(sum_gp_kl, name='gp_kl_sum')

	x_decode = vae_decode(latent_sample)
	x_decode = tf.identity(x_decode, name='x_decode')

	# define the loss function (KL-div + reconst error)
	sum_recon_loss = -tf.reduce_sum(x*tf.log(1e-10+x_decode) + (1.0-x)*tf.log(1.0-x_decode+1e-10),name='sum_recon_loss')
	beta = 1.0
	loss = tf.add(sum_recon_loss, tf.scalar_mul(beta, sum_gp_kl), name='loss') 

	train_step = tf.train.AdamOptimizer(2e-4).minimize(loss)

	logs_path = './VAE_GP_graphs_and_models/'
	model_path = logs_path + 'GP_model'
	write = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

	steps = 1000000
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#### THE NEXT CODE IS FOR DOWNSIZING THE INPUT IMAGES TO 28X28 ON THE TRAINING DATA
		# large_data = MovingMnist.datasets['mixed_train'][0][:,:,:,:]
		# MovingMnist.datasets['mixed_train'][0] = data.eval(feed_dict={test:large_data})
		# MovingMnist.datasets['mixed_train'][0] = np.reshape(MovingMnist.datasets['mixed_train'][0], (-1,8000,28,28))
		
		### uncomment the next line to train on just 2 batch of 20 to show that the net works
		MovingMnist.datasets['train'][0] = MovingMnist.datasets['train'][0][:20,:1,:,:]
		# print(MovingMnist.datasets['mixed_train'][0].shape)
		# plt.imshow(MovingMnist.datasets['mixed_train'][0][0,0,:,:])
		# plt.show()

		#make a saver to save the graph 
		saver = tf.train.Saver()

		#uncomment the next line if working with mixed training set (and comment out the 'train' line)
		# MovingMnist.make_discrete(MovingMnist.datasets['mixed_train'][0])
		MovingMnist.make_discrete(MovingMnist.datasets['train'][0])
		MovingMnist.make_discrete(MovingMnist.datasets['test'][0])

		for i in range(steps):
			batch = MovingMnist.data_batch('train')
			batch = np.reshape(batch,[-1,4096])
			# batch = MovingMnist.data_batch('mixed_train') # 20x5x64x64
			# batch = MovingMnist.data_batch('train')
			# batch = np.reshape(batch,[-1,4096]) # 100x64x64
			train_step.run(feed_dict={x:batch})
			if i % 50 == 0:
				# print(i)
				print(loss.eval(feed_dict={x:batch}))
				print(sum_gp_kl.eval(feed_dict={x:batch}))
				print(sum_recon_loss.eval(feed_dict={x:batch}))
			# if i % 2000 == 0:
				# saver.save(sess, model_path, global_step=i)
				# print(gp_kl.eval(feed_dict={x:batch}))
			# if i % 300000 == 0:
			# 	batch = MovingMnist.data_batch('test')
			# 	batch = np.reshape(batch, [-1,4096])
			# 	img = np.reshape(batch[0], [64,64])
			# 	plt.imshow(img)
			# 	plt.show()
			# 	test = x_decode.eval(feed_dict={x:batch})
			# 	test = tf.reshape(test, [-1,64,64])
			# 	for i in range(20):
			# 		plt.imshow(test.eval()[i])
			# 		plt.show()

			

if __name__ == '__main__':
	main()
