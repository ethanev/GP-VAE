from sklearn.externals import joblib
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from DataHandler import DataHandler, plot_data_tc

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
	return encode_mean

def approx_kernels(sequences, latent_size, batch_size):
	'''
	 produces a list of latent dim kernels and the 'noise' for the reparam trick 
	 when done on a single z latent over the sequence time. 
	 input:
			sequences : a tf placeholder to which one passes a np array of the batch wit time steps 
				ex:[[5.0,6.0,9.0,10.0],[1.0,2.0,5.0,10.4]]
			latent_size : the size of the latent space, a number of a tf.shape()[x] works
	 Output:
	 		full_approx_kernel: A tensor where each row is a linearized kernel matrix 
	 						dim: ((latent_size*batch_size) x time_length^2)
	 		full_chol_noise: A tensor where each row is the multiplication of L (from cholesky(kernel_matrix)) with a vector of random normals
	'''
	sequences = tf.unstack(sequences, num=batch_size)
	time_chars = tf.Variable(tf.constant(1.0, shape=[latent_size,1]), name='approx_time_chars')
	split_time_chars = tf.split(time_chars, num_or_size_splits=latent_size)
	
	full_approx_kernel = []
	full_chol_noise = []
	for sequence in sequences: 
		#break the time_chars and iterate to calc indiv kls
		approx_kernel, chol_noise = build_kernels(sequence, split_time_chars)
		full_approx_kernel.append(approx_kernel)
		full_chol_noise.append(chol_noise)
	full_approx_kernel = tf.concat(full_approx_kernel, 0)
	full_chol_noise = tf.concat(full_chol_noise, 0)
	return full_approx_kernel, full_chol_noise

def prior_kernels(sequence, latent_size, batch_size):
	'''
	 Input: (see before for latent_size and batch_size) 
	 			sequence: a single sequence (nd.array) of [1,2,3,...20] since mnist is 20 steps
	 Output:
				full_prior_kernel : each row is the prior kernel for a singe latent (latent_size*batch_size x time_length^2) and 
					each batch has the SAME matrix 
	'''
	prior_time_chars = tf.Variable(tf.constant(1.0, shape=[latent_size,1]), name='prior_time_chars')
	split_prior_time_char = tf.split(prior_time_chars, num_or_size_splits=latent_size)
	prior_kernel, _ = build_kernels(sequence, split_prior_time_char)
	full_prior_kernel = tf.reshape(tf.tile(prior_kernel, [batch_size,1]), [latent_size*batch_size, -1])
	return full_prior_kernel 

def build_kernels(sequence, split_time_chars):
	'''
	 Inputs:
	 			sequence: similar idea to before (either a tensor or ndarray is fine)
	 			split_time_chars: a list of 1d tensors where each tensor is the time characteristic of the similarly indexed kernel matrix
	 Output:
				latent_kernels : tensor of dim latent_size x time_length^2 (each row is the full kernel matrix of the latent)
	 			latent_noise : tensor of dim latent_Size x time_length (used in reparam trick to sample a value)
	'''
	latent_kernels = []
	latent_noise = []
	for char in split_time_chars:
		K, chol_noise = tf_kernel(sequence, char)
		latent_kernels.append(K)
		latent_noise.append(chol_noise)
	latent_noise = tf.concat(latent_noise,0)
	latent_kernels = tf.concat(latent_kernels,0)
	return latent_kernels, latent_noise

def tf_kernel(sequence, char):
	'''
	Inputs: 
				sequence: see above (it looks lie [1,2,3....20])
				char: a singe 1d tensor that is a time characteristic
	Outputs:
				K: linearized kernel matrix
				noise: L*noise 				 
	'''
	noise = 1e-3
	signal = 1-noise
	sequence_tall = tf.reshape(sequence, [-1, 1])
	sequence_tall = sequence_tall[:,tf.newaxis,:]
	sequence_long = tf.reshape(sequence, [1,-1])
	sequence_long = sequence_long[tf.newaxis,:,:]
	diff = tf.squeeze((sequence_tall - sequence_long))
	diff = tf.cast(diff, tf.float32)
	inner = -tf.pow(diff, 2) / (2.0*tf.pow(char,2))
	noise_mat = tf.eye(tf.shape(diff)[0]) * noise
	K = signal * tf.exp(inner) + noise_mat
	L = tf.cholesky(K)
	random = tf.random_normal([tf.shape(L)[0],1])
	noise = tf.matmul(L, random)
	K = tf.reshape(K, [1,-1])
	noise = tf.reshape(noise, [1,-1])
	return K, noise

def gp_vae_sample(mean, noise_chol_full_time, batch_size):
	'''
	 input:
	 	mean: the output of the encoder, dim:[(batch_size*time_length), latent_size]		NOTE: time_length = 20
	 	noise_chol_full_time : full matrix where is row comes from L*noise (each row is time based, thus needed to be rearranged to match with the means) 
	 		dim: [(batch_size*latent_size),time_length]
	 Outputs:
		time_based_sample: tensor, shape = [batch_size*time_length, latent_size]
	'''
	noise_chol_full_latent = tf.transpose(noise_chol_full_time)
	batched_chol_full_latent = tf.split(noise_chol_full_latent, num_or_size_splits=batch_size, axis=1)
	noise_chol_full_latent = tf.concat(batched_chol_full_latent, 0)
	latent_based_sample = mean + noise_chol_full_latent
	return latent_based_sample

def calc_gp_kl(mean, approx_linear_kernel, prior_kernel, batch_size):
	'''
	 split the mean so each row is a sequence: ie 100x100 batch_size*time_lengh x latent_size --> 500x20 Latent_size*batch_size X time_length
	 ideally would have a better way to do this =(
	 INPUT:
			mean : a tensor of -1xlatent_size where the -1 is batch_size * time_length of each seq
			approx_linear_kernel : a tensor of -1x(time_length)^2 is each row is a linearized kernel matrix
				and the -1 corresponds to batch_size * latent_size
	 		prior_kernel : a tensor of -1x(time_length)^2 is each row is a linearized kernel matrix
				and the -1 corresponds to batch_size * latent_size
	 Outputs:
	 		gp_kl_sum : the collaped KL term after summing up all elements
	 		gp_kl : the tensor version of the kl prior to summing over all examples (used for debugging)
	'''
	approx_linear_kernel = tf.unstack(approx_linear_kernel) # now its a list of 1xT^2 tensors
	prior_kernel = tf.unstack(prior_kernel) # now its a list of 1xT^2 tensors 
	mean_t_kl = break_mat(mean, batch_size) 

	gp_kl_full = []
	for m, K, p_K in zip(mean_t_kl, approx_linear_kernel, prior_kernel):
		gp_kl_full.append(gp_kl_div(m, K, p_K))
	gp_kl = tf.stack(gp_kl_full)
	gp_kl_sum = tf.reduce_sum(gp_kl)
	return gp_kl_sum, gp_kl

def break_mat(mat, batch_size):
	mat_T = tf.transpose(mat)
	split_mat = tf.split(mat_T, num_or_size_splits=batch_size, axis=1)
	mat_GP = tf.concat(split_mat, 0)
	mat_unpack = tf.unstack(mat_GP)
	return mat_unpack

def gp_kl_div(mean, approx_linear_kernel, prior_linear_kernel):
	mean = tf.reshape(mean, [20,1])
	mean = tf.cast(mean, tf.float64)
	approx_kernel = tf.reshape(approx_linear_kernel, [20,20])
	approx_kernel = tf.cast(approx_kernel, tf.float64)
	prior_kernel = tf.reshape(prior_linear_kernel, [20,20])
	prior_kernel = tf.cast(prior_kernel, tf.float64)
	inv_prior_K = tf.matrix_inverse(prior_kernel)
	log_det_approx_K = tf.linalg.logdet(approx_kernel)
	log_det_prior_K = tf.linalg.logdet(prior_kernel)
	negative_mean = tf.scalar_mul(-1.0,mean)
	p_1 = tf.trace(tf.matmul(inv_prior_K, approx_kernel)) 
	p_2 = log_det_prior_K - log_det_approx_K
	p_3a = tf.matmul(inv_prior_K,negative_mean)
	p_3 = tf.matmul(tf.transpose(negative_mean), p_3a) 
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

	# TO RUN THIS MAKE SURE YOU MAKE THE FOLLOWING DIRECTORY IN THE CURRENT LOCATION:
	#      Full_GP_models_fixed
	# ALTER THE PATHS RIGHT BELOW THIS FOR WHERE THE DATA IS!
	# IF YOU WANT NICE FIGURES: CHANGE .PNG TO .SVG and alter the format

	# data path
	#on local comp
	#data_path = "/Users/ethanevans/Documents/Externship/Data/"
	# on VM:
	data_path = "/home/ethan/Data/"
	data_file = "mnist_test_seq.npy"
	# batch size - IMPORTANT this is the number of sequences! Not the number of images
	batch_size = 5 #each sequence here is 20 steps --> 100 images 
	#load the data 
	MovingMnist = DataHandler(data_path, data_file, batch_size=batch_size)
	# MovingMnist.make_shuffled_dataset() # comment out for sequence training

	prior_length = 20
	prior_sequence = np.array([float(i)+1 for i in range(prior_length)], dtype=np.float32)
	test_sequences = np.array([[float(i)+1 for i in range(prior_length)] for i in range(batch_size)], dtype=np.float32)
	latent_size = 100

	# make the model
	x = tf.placeholder(tf.float32, [None,4096], name="x")
	sequences = tf.placeholder(tf.float32, [None, 20], name="sequences")
	# time = tf.placeholder(tf.float32, [None,T], name='time')
	latent_mean = vae_encode(x)
	latent_mean = tf.identity(latent_mean, name='latent_mean')

	prior_kernel = prior_kernels(prior_sequence, latent_size, batch_size)
	prior_kernel = tf.identity(prior_kernel, name='prior_kernels')

	approx_kernel, chol_noise = approx_kernels(test_sequences, latent_size, batch_size) #change back to passing a tf.placeholder for sequence
	approx_kernel = tf.identity(approx_kernel, name='approx_kernels')
	chol_noise = tf.identity(chol_noise, name='chol_noise')

	latent_sample = gp_vae_sample(latent_mean, chol_noise, batch_size)
	sum_gp_kl, gp_kl = calc_gp_kl(latent_mean, approx_kernel, prior_kernel, batch_size)
	latent_sample = tf.identity(latent_sample, name='latent_sample')
	sum_gp_kl = tf.identity(sum_gp_kl, name='gp_kl_sum')

	x_decode = vae_decode(latent_sample)
	x_decode = tf.identity(x_decode, name='x_decode')

	# define the loss function (KL-div + reconst error)
	sum_recon_loss = -tf.reduce_sum(x*tf.log(1e-10+x_decode) + (1.0-x)*tf.log(1.0-x_decode+1e-10),name='sum_recon_loss')
	sum_recon_loss = tf.cast(sum_recon_loss, tf.float64)
	beta = 1.0
	loss = tf.add(sum_recon_loss, tf.scalar_mul(beta, sum_gp_kl), name='loss') 

	train_step = tf.train.AdamOptimizer(2e-4).minimize(loss)

	logs_path = './Full_GP_models_fixed/'
	model_path = logs_path + 'GP_model'
	write = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

	steps = 5000000
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#### THE NEXT CODE IS FOR DOWNSIZING THE INPUT IMAGES TO 28X28 ON THE TRAINING DATA
		# large_data = MovingMnist.datasets['mixed_train'][0][:,:,:,:]
		# MovingMnist.datasets['mixed_train'][0] = data.eval(feed_dict={test:large_data})
		# MovingMnist.datasets['mixed_train'][0] = np.reshape(MovingMnist.datasets['mixed_train'][0], (-1,8000,28,28))
		
		### uncomment the next line to train on just 2 batch of 20 to show that the net works
		#MovingMnist.datasets['train'][0] = MovingMnist.datasets['train'][0][:20,:5,:,:]
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
			test_sequences = np.reshape(test_sequences, [-1,20])
			# batch = MovingMnist.data_batch('mixed_train') # 20x5x64x64
			# batch = MovingMnist.data_batch('train')
			# batch = np.reshape(batch,[-1,4096]) # 100x64x64
			train_step.run(feed_dict={x:batch, sequences:test_sequences})
			if i % 500 == 0:
				# print(i)
				print(loss.eval(feed_dict={x:batch}))
				print(sum_gp_kl.eval(feed_dict={x:batch}))
				print(sum_recon_loss.eval(feed_dict={x:batch}))
			if i % 25000 == 0:
				saver.save(sess, model_path, global_step=i)
				# print(gp_kl.eval(feed_dict={x:batch}))
			# if i % 300000 == 0:
				# large_image = np.zeros((64, 64*20))
				# batch = MovingMnist.data_batch('test')
				# batch = np.reshape(batch, [-1,4096])
				# # img = np.reshape(batch[0], [64,64])
				# # plt.imshow(img)
				# # plt.show()
				# test = x_decode.eval(feed_dict={x:batch})
				# test = np.reshape(test, [-1,64,64])
				# for j in range(20):
				# 	large_image[:, j * 64: (j + 1) * 64] = test[j,:,:]
				# plt.imshow(large_image)
				# plt.savefig('VAE_GP_%i.png'%(i))

			

if __name__ == '__main__':
	main()