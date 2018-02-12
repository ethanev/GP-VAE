from sklearn.externals import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from DataHandler import SyntheticDataHandler, DataHandler

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

def vae_encode(x, latent_size):
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
	w_mean = weight_variable([512,latent_size])
	b_mean = bias_variable([latent_size])
	encode_mean = tf.matmul(flat_6, w_mean)+b_mean
	return encode_mean

def approx_kernels(sequences, sequence_sizes, latent_size, batch_size, number_samples):
	'''
	 produces a list of latent dim kernels and the 'noise' for the reparam trick 
	 when done on a single z latent over the sequence time. 
	 input:
			sequences : a tf placeholder to which one passes a np array of the batch with time steps 
				ex:[[5.0,6.0,9.0,10.0],[1.0,2.0,5.0,10.4]]
			sequence_sizes : tf placeholder to which is pass a np array of shape [batch_size] where each value is the length of that seqeuence
			latent_size : the size of the latent space, a number of a tf.shape()[x] works
	'''
	sequences = tf.split(sequences, num_or_size_splits=batch_size)

	time_chars = tf.Variable(tf.constant(1.0, shape=[latent_size,1]), name='approx_time_chars')
	# time_chars = tf.constant([9.0,3.0], shape=[latent_size,1])
	split_time_chars = tf.split(time_chars, num_or_size_splits=latent_size)

	split_sequence_length = tf.split(sequence_sizes, num_or_size_splits=batch_size)
	pad_length = tf.reduce_max(sequence_sizes)
	pad_length_sq = tf.pow(pad_length, 2)
	full_approx_kernel = []
	full_chol_noise = []
	for sequence, size in zip(sequences, split_sequence_length): 
		sequence = tf.slice(sequence,[0,0], [1,tf.reshape(size,())])
		approx_kernel, chol_noise = build_kernels(sequence, split_time_chars, number_samples)
		padding_kernel = [[0, 0], [0, tf.reshape(tf.cast(pad_length_sq-size*size, tf.int32),())]]# pad_length-prior_kernel.shape[1]
		pad_approx_kernel = tf.pad(approx_kernel, padding_kernel, 'CONSTANT')
		
		chol_noise = tf.split(chol_noise, number_samples, axis=1)
		padding_chol = [[0, 0], [0, tf.reshape(tf.cast(pad_length-size, tf.int32),())]]
		padded_chol = []
		for sample in chol_noise:
			pad_chol_noise = tf.pad(sample, padding_chol, 'CONSTANT')
			padded_chol.append(pad_chol_noise)
		pad_chol_noise = tf.concat(padded_chol, axis=1)
		full_approx_kernel.append(pad_approx_kernel)
		full_chol_noise.append(pad_chol_noise)
	full_approx_kernel = tf.concat(full_approx_kernel, 0)
	full_chol_noise = tf.concat(full_chol_noise, 0)
	return full_approx_kernel, full_chol_noise, time_chars

def prior_kernels(sequences, sequence_sizes, latent_size, batch_size):
	'''
	 Input:
				sequences: TF placeholder [None, 15] matrix where 15 is the input dimension 
				Sequence_sizes: TF placeholder [batch_size] matrix where the components are the length of the sequences of the batch
				latent_size: number of latent states desired
				batch_size: number of sequences per batch
	 Output:
				full_prior_kernel : each row is the prior kernel for a singe latent (latent_size*batch_size x time_length^2) and 
					each batch has the SAME matrix 
	'''
	sequences = tf.split(sequences, num_or_size_splits=batch_size)

	# prior_time_chars = tf.Variable(tf.constant([9.0,3.0], shape=[latent_size,1]), name='prior_time_chars')
	prior_time_chars = tf.constant(1.0, shape=[latent_size,1])
	split_prior_time_char = tf.split(prior_time_chars, num_or_size_splits=latent_size) #the same values are used for each seq in the batch
	
	split_prior_sequence_length = tf.split(sequence_sizes, num_or_size_splits=batch_size)
	pad_length = tf.pow(tf.reduce_max(sequence_sizes),2) # used to set the number of padding 0s to add to each sequence

	full_prior_kernel = []
	for sequence, size in zip(sequences, split_prior_sequence_length):
		sequence = tf.slice(sequence,[0,0], [1,tf.reshape(size,())])
		prior_kernel, _ = build_kernels(sequence, split_prior_time_char)

		paddings = [[0, 0], [0, tf.reshape(tf.cast(pad_length-size*size, tf.int32),())]]# pad_length-prior_kernel.shape[1]
		prior_kernel = tf.pad(prior_kernel, paddings, 'CONSTANT')

		full_prior_kernel.append(prior_kernel)
	full_prior_kernel = tf.concat(full_prior_kernel, 0)
	return full_prior_kernel, prior_time_chars

def build_kernels(sequence, split_time_chars, number_samples=1):
	'''
	 Inputs:
	 Output:
				latent_kernels : tensor of dim latent_size x time_length^2 (each row is the full kernel matrix of the latent)
	 			latent_noise : tensor of dim latent_Size x time_length (used in reparam trick to sample a value)
	'''
	latent_kernels = []
	latent_noise = []
	for char in split_time_chars:
		K, chol_noise = tf_kernel(sequence, char, number_samples)
		latent_kernels.append(K)
		latent_noise.append(chol_noise)
	latent_noise = tf.concat(latent_noise,0)
	latent_kernels = tf.concat(latent_kernels,0)
	return latent_kernels, latent_noise

def tf_kernel(sequence, char, number_samples):
	'''
	 sequence is (typically) a placeholder tensor with the times that the data point are from, ie [1.0,2.0,4.0]
	 char is a of a single number corresponding to the time characteristic of the GP
	'''
	noise = 1e-3
	signal = 1-noise
	sequence_tall = tf.reshape(sequence, [-1, 1])
	sequence_tall = sequence_tall[:,tf.newaxis,:]
	sequence_long = tf.reshape(sequence, [1,-1])
	sequence_long = sequence_long[tf.newaxis,:,:]
	diff = tf.squeeze((sequence_tall - sequence_long))
	# diff = tf.cast(diff, tf.float32)
	inner = -tf.pow(diff, 2) / (2.0*tf.pow(char,2))
	noise_mat = tf.eye(tf.shape(diff)[0]) * noise
	K = signal * tf.exp(inner) + noise_mat
	L = tf.cholesky(K)
	random = tf.random_normal([tf.shape(L)[0],number_samples])
	# random = tf.ones([tf.shape(L)[0],number_samples]) #THIS IS JUST FOR DEBUGGIN!
	noise = tf.matmul(L, random)
	K = tf.reshape(K, [1,-1])
	noise = tf.transpose(noise)
	noise = tf.reshape(noise, [1,-1])
	return K, noise

def gp_vae_sample(mean, noise_chol_full_time, sequence_sizes, batch_size, number_samples, latent_size):
	'''
	 input:
	 	batch_means_seq :  [(latent_size*batch_size),time_length] ie each latent of each batch with time
	 	noise_chol_full_time : same as above but from noise * cholesky(K) where K is for each *row* (each latent over time)
	 Outputs:
		time_based_sample: tensor, shape = [batch_size*time_length, latent_size]
	'''
	batched_chol_full_latent = tf.split(noise_chol_full_time, num_or_size_splits=batch_size, axis=0)
	#now need to splice each of the matricies to extract only the times that are actually seen (all padded with 0s on the right)
	split_sequence_length = tf.split(sequence_sizes, num_or_size_splits=batch_size)
	split_means = tf.split(mean, num_or_size_splits=sequence_sizes)
	samples = []
	for sequence, size, mean in zip(batched_chol_full_latent, split_sequence_length, split_means):
		indiv_sample_chol = tf.split(sequence, num_or_size_splits=number_samples, axis=1)
		for chol in indiv_sample_chol:
			chol_noise = tf.slice(chol,[0,0], [latent_size,tf.reshape(size,())])
			noise_trans = tf.transpose(chol_noise)
			sample = mean + noise_trans
			samples.append(sample)
	samples_matrix = tf.concat(samples, 0)
	return samples_matrix

def calc_gp_kl(mean, sequence_sizes, approx_linear_kernel, prior_kernel, batch_size, latent_size):
	'''
	 split the mean so each row is a sequence: ie 100x100 --> 500x20 
	 ideally would have a better way to do this =(
	 INPUT:
			mean : a tensor of -1xlatent_size where the -1 is batch_size * time_length of each seq
			sequence_sizes : tensor of shape [batch_size] where each element is the length of a given sequence
			approx_linear_kernel : a tensor of -1x(time_length)^2 is each row is a linearized kernel matrix
				and the -1 corresponds to batch_size * latent_size
	 		prior_kernel : a tensor of -1x(time_length)^2 is each row is a linearized kernel matrix
				and the -1 corresponds to batch_size * latent_size
			batch_size : the number of sequences used in each training step
	 		latent_size : number of latent variables
	'''
	approx_linear_kernel = tf.unstack(approx_linear_kernel, num=batch_size*latent_size) # now its a list of 1xT^2 tensors
	# return approx_linear_kernel[2]
	prior_kernel = tf.unstack(prior_kernel, num=batch_size*latent_size) # now its a list of 1xT^2 tensors
	mean_t_kl = trans_break_mat(mean, sequence_sizes) # mean_time will be used for sampling 

	sequence_sizes = tf.reshape(tf.tile(tf.expand_dims(sequence_sizes, -1),  [1, latent_size]), [-1])
	split_sequence_length = tf.split(sequence_sizes, num_or_size_splits=batch_size*latent_size)
	
	gp_kl_full = []
	test = []
	i = 0
	for m, K, p_K, length in zip(mean_t_kl, approx_linear_kernel, prior_kernel, split_sequence_length):
		K = tf.slice(K, [0], [tf.reshape(tf.square(length),())])
		p_K = tf.slice(p_K, [0], [tf.reshape(tf.square(length),())])
		kl = gp_kl_div(m, K, p_K, length)
		gp_kl_full.append(kl)
	gp_kl = tf.stack(gp_kl_full)
	gp_kl_sum = tf.reduce_sum(gp_kl)
	return gp_kl_sum, gp_kl

def trans_break_mat(mat, batch_sequence_size):
	'''
	 takes in a matrix of (batch_sequence_lengths_sum) x latent_size and returns list of batch members of size latent_size x sequence_length
	'''
	mat_T = tf.transpose(mat)
	split_mat = tf.split(mat_T, num_or_size_splits=batch_sequence_size, axis=1)
	for ind, tensor in enumerate(split_mat):
		split_mat[ind] = tf.unstack(tensor)
	split_mat = [tensor for sublist in split_mat for tensor in sublist]
	return split_mat

def gp_kl_div(mean, approx_linear_kernel, prior_linear_kernel, length):
	mean = tf.reshape(mean, [-1,1])
	mean = tf.cast(mean, tf.float64)
	length = tf.cast(tf.reshape(length,()), tf.int32)
	approx_kernel = tf.reshape(approx_linear_kernel, [length,length])
	approx_kernel = tf.cast(approx_kernel, tf.float64)
	prior_kernel = tf.reshape(prior_linear_kernel, [length,length])
	prior_kernel = tf.cast(prior_kernel, tf.float64)
	inv_prior_K = tf.matrix_inverse(prior_kernel)
	log_det_approx_K = tf.linalg.logdet(approx_kernel)
	log_det_prior_K = tf.linalg.logdet(prior_kernel)
	negative_mean = tf.scalar_mul(-1.0,mean)
	p_1 = tf.trace(tf.matmul(inv_prior_K, approx_kernel)) 
	p_2 = log_det_prior_K - log_det_approx_K
	p_3a = tf.matmul(inv_prior_K,negative_mean)
	p_3 = tf.matmul(tf.transpose(negative_mean), p_3a) 
	p_4 = p_1 - tf.cast(length, tf.float64) + p_2 + p_3
	p_5 = tf.scalar_mul(0.5, p_4)
	return p_5
	
def vae_decode(X, latent_size):
	X = tf.reshape(X, [-1,latent_size]) 
	w1 = weight_variable([latent_size, 512])
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
 
def write_file(losses, step, model_type='train'):
	file = open('./Full_GP_models/%s_loss_file_%d.csv' %(model_type, step), 'w')
	for step in losses:
		for ele in step:
			file.write(ele + ',')
		file.write('\n')
	file.close()

def main():

	# TO RUN THIS MAKE SURE YOU MAKE THE FOLLOWING DIRECTORY IN THE CURRENT LOCATION:
	#      Full_GP_models
	# ALTER THE PATHS RIGHT BELOW THIS FOR WHERE THE DATA IS!
	# IF YOU WANT NICE FIGURES: CHANGE .PNG TO .SVG and alter the format
	data_path = "../Data/"
	data_file = "mnist_test_seq.npy"

	max_time = 20 #FOR THE MOVING MNIST!
	batch_size = 5 # number of sequences analyzed per training step

	MovingMnist = DataHandler(data_path, data_file, batch_size=batch_size, time_included=True)

	# make the model
	latent_size = 100
	number_samples = 1

	x = tf.placeholder(tf.float32, [None,4096], name="x")
	sequences = tf.placeholder(tf.float32, [batch_size, None], name="sequences")
	sequence_sizes_placeholder = tf.placeholder(tf.int32, [batch_size], name="sequence_lengths")
	split_x = tf.split(x, num_or_size_splits=sequence_sizes_placeholder)
	x_sample_num_tile = []
	for sequence in split_x:
		x_sample_num_tile.append(tf.tile(sequence, [number_samples,1]))
	x_sample_num_tile = tf.concat(x_sample_num_tile, axis=0)

	latent_mean = vae_encode(x, latent_size)
	latent_mean = tf.identity(latent_mean, name='latent_mean')

	prior_kernel, prior_time_chars = prior_kernels(sequences, sequence_sizes_placeholder, latent_size, batch_size)
	prior_kernel = tf.identity(prior_kernel, name='prior_kernels')
	# chol_noise = approx_kernels(sequences, sequence_sizes_placeholder, latent_size, batch_size, number_samples)
	approx_kernel, chol_noise, approx_time_chars = approx_kernels(sequences, sequence_sizes_placeholder, latent_size, batch_size, number_samples) #change back to passing a tf.placeholder for sequence
	approx_kernel = tf.identity(approx_kernel, name='approx_kernels') #shape is (batch*latent) x time
	chol_noise = tf.identity(chol_noise, name='chol_noise')

	latent_sample = gp_vae_sample(latent_mean, chol_noise, sequence_sizes_placeholder, batch_size, number_samples, latent_size)
	sum_gp_kl, gp_kl= calc_gp_kl(latent_mean, sequence_sizes_placeholder, approx_kernel, prior_kernel, batch_size, latent_size)
	# p_K= calc_gp_kl(latent_mean, sequence_sizes_placeholder, approx_kernel, prior_kernel, batch_size, latent_size)
	latent_sample = tf.identity(latent_sample, name='latent_sample')
	sum_gp_kl = tf.identity(sum_gp_kl, name='gp_kl_sum')

	x_decode = vae_decode(latent_sample, latent_size)
	x_decode = tf.identity(x_decode, name='x_decode')

	# define the loss function (KL-div + reconst error)
	reconstruction = -tf.reduce_sum(tf.multiply(x_sample_num_tile,tf.log(1e-10+x_decode)) + tf.multiply((1.0-x_sample_num_tile),tf.log(1.0-x_decode+1e-10)),name='sum_recon_loss', axis=1)
	reconstruction = tf.cast(reconstruction, tf.float64)
	split_recon = tf.split(reconstruction, num_or_size_splits=tf.scalar_mul(number_samples,sequence_sizes_placeholder), axis=0)
	for i, sequence in enumerate(split_recon):
		split_recon[i] = tf.transpose(tf.reshape(sequence, [number_samples,-1]))
	row_sample_recon = tf.concat(split_recon, axis=0)
	row_sample_recon_mean = tf.reduce_mean(row_sample_recon, 1)
	sum_recon_loss = tf.reduce_sum(row_sample_recon_mean, name='sum_recon_loss')

	beta = tf.placeholder(tf.float64, [])
	beta_value = 1.0
	loss = tf.add(sum_recon_loss, (beta*sum_gp_kl), name='loss') 
	train_step = tf.train.AdamOptimizer(2e-4).minimize(loss)
	logs_path = './Full_GP_models/'
	model_path = logs_path + 'gp_model'
	write = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

	steps = 5000000
	with tf.Session() as sess:
		MovingMnist.make_discrete(MovingMnist.datasets['train'][0])
		MovingMnist.make_discrete(MovingMnist.datasets['test'][0])
		sess.run(tf.global_variables_initializer())
		train_losses = [['Step', 'Reconstruction loss', 'KL loss']]
		test_losses = [['Step', 'Reconstruction loss', 'KL loss']]
		#make a saver to save the graph 
		saver = tf.train.Saver()
		for i in range(steps):
			batch_xs, batch_time_steps, batch_lengths = MovingMnist.data_batch('train')
			inputs = {x:batch_xs, sequences:batch_time_steps, sequence_sizes_placeholder:batch_lengths, beta:beta_value}
			train_step.run(feed_dict=inputs)
			if i % 500 == 0:
				step_losses = []
				step_losses.append(str(i))
				step_losses.append(str(sum_recon_loss.eval(feed_dict=inputs)))
				step_losses.append(str(sum_gp_kl.eval(feed_dict=inputs)))
				train_losses.append(step_losses)
			if i % 10000 == 0:
				batch_xs, batch_time_steps, batch_lengths = MovingMnist.data_batch('test')
				inputs = {x:batch_xs, sequences:batch_time_steps, sequence_sizes_placeholder:batch_lengths, beta:beta_value}

				large_image = np.zeros((64, 64*20))
				batch = np.reshape(batch_xs, [-1,64,64])
				for j in range(20):
					large_image[:, j * 64: (j + 1) * 64] = batch[j,:,:]
				plt.imshow(large_image)
				plt.savefig('./Full_GP_models/input_sequence_%d.png' %(i))
				plt.gcf().clear()

				test_step_losses = []
				test_step_losses.append(str(i))
				test_step_losses.append(str(sum_recon_loss.eval(feed_dict=inputs)))
				test_step_losses.append(str(sum_gp_kl.eval(feed_dict=inputs)))
				test_losses.append(test_step_losses)

				test = x_decode.eval(feed_dict=inputs)
				test = np.reshape(test, [-1,64,64])
				large_image = np.zeros((64, 64*20))
				for j in range(20):
					large_image[:, j * 64: (j + 1) * 64] = test[j,:,:]
				plt.imshow(large_image)
				plt.savefig('./Full_GP_models/output_sequence_%d.png' %(i))
				plt.gcf().clear()

			if i % 25000 == 0:
				saver.save(sess, model_path, global_step=i)
				write_file(train_losses, i)
				write_file(test_losses, i, type='test')

if __name__ == '__main__':
	main()