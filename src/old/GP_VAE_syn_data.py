import tensorflow as tf
import numpy as np
from DataHandler import DataHandler, plot_data_tc
from sklearn.externals import joblib
# if on VM comment out the following line:
import matplotlib.pyplot as plt


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def vae_encode(x, latent_size):
	x = tf.reshape(x, [-1,15])
	w_in_1 = weight_variable([15,8])
	b_in_1 = bias_variable([8])
	encode_1 = tf.nn.relu(tf.matmul(x,w_in_1) + b_in_1)

	#latent space mean
	w_mean = weight_variable([8,latent_size])
	b_mean = bias_variable([latent_size])
	encode_mean = tf.matmul(encode_1, w_mean)+b_mean
	return encode_mean

def approx_kernels(sequences, sequence_sizes, latent_size, batch_size):
	# produces a list of latent dim kernels and the 'noise' for the reparam trick 
	# when done on a single z latent over the sequence time. 
	# input:
	#		sequences : a tf placeholder to which one passes a np array of the batch with time steps 
	#			ex:[[5.0,6.0,9.0,10.0],[1.0,2.0,5.0,10.4]]
	#		latent_size : the size of the latent space, a number of a tf.shape()[x] works
	sequences = tf.split(sequences, num_or_size_splits=batch_size)

	time_chars = tf.Variable(tf.constant(1.0, shape=[latent_size,1]), name='approx_time_chars')
	split_time_chars = tf.split(time_chars, num_or_size_splits=latent_size)

	split_prior_sequence_length = tf.split(sequence_sizes, num_or_size_splits=batch_size)
	pad_length = tf.reduce_max(sequence_sizes)
	pad_length_sq = tf.pow(pad_length, 2)
	full_approx_kernel = []
	full_chol_noise = []
	for sequence, size in zip(sequences, split_prior_sequence_length): 
		sequence = tf.slice(sequence,[0,0], [1,tf.reshape(size,())])
		approx_kernel, chol_noise = build_kernels(sequence, split_time_chars)

		padding_kernel = [[0, 0], [0, tf.reshape(tf.cast(pad_length_sq-size*size, tf.int32),())]]# pad_length-prior_kernel.shape[1]
		approx_kernel = tf.pad(approx_kernel, padding_kernel, 'CONSTANT')
		padding_chol = [[0, 0], [0, tf.reshape(tf.cast(pad_length-size, tf.int32),())]]
		chol_noise = tf.pad(chol_noise, padding_chol, 'CONSTANT')

		full_approx_kernel.append(approx_kernel)
		full_chol_noise.append(chol_noise)
	full_approx_kernel = tf.concat(full_approx_kernel, 0)
	full_chol_noise = tf.concat(full_chol_noise, 0)
	return full_approx_kernel, full_chol_noise

def prior_kernels(sequences, sequence_sizes, latent_size, batch_size):
	# Input:
	#			sequences: TF placeholder [None, 15] matrix where 15 is the input dimension 
	#			Sequence_sizes: TF placeholder [batch_size] matrix where the components are the length of the sequences of the batch
	#			latent_size: number of latent states desired
	#			batch_size: number of sequences per batch
	# Output:
	#			full_prior_kernel : each row is the prior kernel for a singe latent (latent_size*batch_size x time_length^2) and 
	#				each batch has the SAME matrix 
	sequences = tf.split(sequences, num_or_size_splits=batch_size)

	prior_time_chars = tf.Variable(tf.constant(1.0, shape=[latent_size,1]), name='prior_time_chars')
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
	return full_prior_kernel

def build_kernels(sequence, split_time_chars):
	# Inputs
	# Output:
	#			latent_kernels : tensor of dim latent_size x time_length^2 (each row is the full kernel matrix of the latent)
	# 			latent_noise : tensor of dim latent_Size x time_length (used in reparam trick to sample a value)
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
	# sequence is (typically) a placeholder tensor with the times that the data point are from, ie [1.0,2.0,4.0]
	# char is a of a single number corresponding to the time characteristic of the GP
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

def gp_vae_sample(mean, noise_chol_full_time, sequence_sizes, batch_size):
	# input:
	# 	batch_means_seq :  [(latent_size*batch_size),time_length] ie each latent of each batch with time
	# 	noise_chol_full_time : same as above but from noise * cholesky(K) where K is for each *row* (each latent over time)
	# Outputs:
	#	time_based_sample: tensor, shape = [batch_size*time_length, latent_size]
	batched_chol_full_latent = tf.split(noise_chol_full_time, num_or_size_splits=batch_size, axis=0)
	#now need to splice each of the matricies to extract only the times that are actually seen (all padded with 0s on the right)
	split_sequence_length = tf.split(sequence_sizes, num_or_size_splits=batch_size)
	latent_based_chol_noise = []
	for sequence, size in zip(batched_chol_full_latent, split_sequence_length):
		sequence = tf.slice(sequence,[0,0], [2,tf.reshape(size,())])
		sequence_trans = tf.transpose(sequence)
		latent_based_chol_noise.append(sequence_trans)
	noise_chol_full_latent = tf.concat(latent_based_chol_noise, 0)
	latent_based_sample = mean + noise_chol_full_latent
	return latent_based_sample

def calc_gp_kl(mean, sequence_sizes, approx_linear_kernel, prior_kernel, batch_size, latent_size):
	# split the mean so each row is a sequence: ie 100x100 --> 500x20 
	# ideally would have a better way to do this =(
	# INPUT:
	#		mean : a tensor of -1xlatent_size where the -1 is batch_size * time_length of each seq
	#		approx_linear_kernel : a tensor of -1x(time_length)^2 is each row is a linearized kernel matrix
	#			and the -1 corresponds to batch_size * latent_size
	# 		prior_kernel : a tensor of -1x(time_length)^2 is each row is a linearized kernel matrix
	#			and the -1 corresponds to batch_size * latent_size
	approx_linear_kernel = tf.unstack(approx_linear_kernel, num=batch_size*latent_size) # now its a list of 1xT^2 tensors
	# return approx_linear_kernel[2]
	prior_kernel = tf.unstack(prior_kernel, num=batch_size*latent_size) # now its a list of 1xT^2 tensors 
	mean_t_kl = trans_break_mat(mean, sequence_sizes) # mean_time will be used for sampling 

	sequence_sizes = tf.reshape(tf.tile(tf.expand_dims(sequence_sizes, -1),  [1, 2]), [-1])
	split_sequence_length = tf.split(sequence_sizes, num_or_size_splits=batch_size*2)
	
	gp_kl_full = []
	i = 0
	for m, K, p_K, length in zip(mean_t_kl, approx_linear_kernel, prior_kernel, split_sequence_length):
		K = tf.slice(K, [0], [tf.reshape(tf.square(length),())])
		p_K = tf.slice(p_K, [0], [tf.reshape(tf.square(length),())])
		# length = tf.cast(tf.reshape(length,()), tf.int32)
		# p_K = tf.reshape(p_K, [length, length])
		# i+=1
		# if i == 1:
		# 	return p_K, length
		# p_K_inv = tf.matrix_inverse(p_K)
		# return gp_kl_div(m,K,p_K,length)
		gp_kl_full.append(gp_kl_div(m, K, p_K, length))
	gp_kl = tf.stack(gp_kl_full)
	gp_kl_sum = tf.reduce_sum(gp_kl)
	return gp_kl_sum, gp_kl

def trans_break_mat(mat, batch_sequence_size):
	# takes in a matrix of (batch_sequence_lengths_sum) x latent_size and returns list of batch members of size latent_size x sequence_length
	mat_T = tf.transpose(mat)
	split_mat = tf.split(mat_T, num_or_size_splits=batch_sequence_size, axis=1)
	for ind, tensor in enumerate(split_mat):
		split_mat[ind] = tf.unstack(tensor)
	split_mat = [tensor for sublist in split_mat for tensor in sublist]
	return split_mat

def gp_kl_div(mean, approx_linear_kernel, prior_linear_kernel, length):
	mean = tf.reshape(mean, [-1,1])
	# approx_linear_kernel = tf.reshape(approx_linear_kernel, [-1])
	# prior_linear_kernel = tf.reshape(prior_linear_kernel, [-1])
	length = tf.cast(tf.reshape(length,()), tf.int32)
	approx_kernel = tf.reshape(approx_linear_kernel, [length,length])
	prior_kernel = tf.reshape(prior_linear_kernel, [length,length])
	det_approx_kernel = tf.matrix_determinant(approx_kernel)
	inv_prior_K = tf.matrix_inverse(prior_kernel)
	det_prior_K = tf.matrix_determinant(prior_kernel)
	negative_mean = tf.scalar_mul(-1.0,mean)
	p_1 = tf.trace(tf.matmul(inv_prior_K, approx_kernel)) 
	p_2 = tf.log(det_prior_K+1e-15) - tf.log(det_approx_kernel+1e-15)
	p_3a = tf.matmul(inv_prior_K,negative_mean)
	p_3 = tf.matmul(tf.transpose(negative_mean), p_3a) 
	p_4 = p_1 - 20.0 + p_2 + p_3
	p_5 = tf.scalar_mul(0.5, p_4)
	return p_5
	
def vae_decode(x, latent_size):
	x = tf.reshape(x, [-1,latent_size]) 
	w1 = weight_variable([latent_size, 8])
	b1 = bias_variable([8])
	out_1 = tf.nn.relu(tf.matmul(x,w1) + b1)

	w_deconv_2 = weight_variable([8,15])
	b_deconv_2 = bias_variable([15])
	decode = tf.sigmoid(tf.matmul(out_1, w_deconv_2) + b_deconv_2)
	return decode
 
def main():
	# data path
	#on local comp
	#data_path = "/Users/ethanevans/Documents/Externship/Data/"
	# on VM:
	# data_path = "/home/ethan/Data/"
	data = joblib.load('./toy_data.pkl')
	#find indicies of timesteps
	small_data = [] #small data set of just first 5 sequences
	true_sequences = [] #the indecies for the time steps 
	sequence_sizes = [] #the sizes of each of the sequences in the data set
	max_time = 1000
	for i in range(5):
		# true_sequences.append(np.reshape(np.where(data['x'][i,0,:] > -1)[0],[1,-1])) # time indicies where the sequence is valid for the kernel matrix
		true_sequences.append(np.reshape(data['time'][np.where(data['x'][i,0,:] > -1)],[1,-1]))
		# build the actual data to be used after removing columns that are not data
		small_data.append(np.reshape(data['x'][i,:,:][np.where(data['x'][i,:,:] > -1)],[15,-1]).T) #new test is a list of 5, indiv_length X 15 matricies
		# build a list of the sizes for each sequecne to know where to split the data matrix in the VAE
		sequence_sizes.append(np.reshape(data['x'][i,:,:][np.where(data['x'][i,:,:] > -1)],[15,-1]).T.shape[0]) 
	small_data_mat = np.concatenate(small_data, axis=0)
	true_sequences_mat = []
	for seq in true_sequences:
		true_sequences_mat.append(np.pad(seq,((0,0),(0,max_time-seq.shape[1])), 'constant', constant_values=0))
	true_sequences_mat = np.concatenate(true_sequences_mat)

	# np.reshape(test['x'][0,:,:][np.where(test['x'][0,:,:] > -1)], [15,-1])
	# sequence_time_steps = np.where(test['x'][0,0,:] > -1)
	# test['x'][0,:,:][np.where(test['x'][0,0,:] > -1)]

	batch_size = 5 # number of sequences analyzed per training step
	#load the data 
	# MovingMnist = DataHandler(data_path, data_file, batch_size=batch_size)
	# MovingMnist.make_shuffled_dataset() # comment out for sequence training

	prior_sequences = true_sequences 
	latent_size = 2

	# make the model
	x = tf.placeholder(tf.float32, [None,15], name="x")
	sequences = tf.placeholder(tf.float32, [batch_size, None], name="sequences")
	sequence_sizes_placeholder = tf.placeholder(tf.int32, [batch_size], name="sequence_lengths")
	latent_mean = vae_encode(x, latent_size)
	latent_mean = tf.identity(latent_mean, name='latent_mean')

	prior_kernel = prior_kernels(sequences, sequence_sizes_placeholder, latent_size, batch_size)
	prior_kernel = tf.identity(prior_kernel, name='prior_kernels')

	approx_kernel, chol_noise = approx_kernels(sequences, sequence_sizes_placeholder, latent_size, batch_size) #change back to passing a tf.placeholder for sequence
	approx_kernel = tf.identity(approx_kernel, name='approx_kernels') #shape is (batch*latent) x time
	chol_noise = tf.identity(chol_noise, name='chol_noise')

	latent_sample = gp_vae_sample(latent_mean, chol_noise, sequence_sizes_placeholder, batch_size)
	sum_gp_kl, gp_kl = calc_gp_kl(latent_mean, sequence_sizes_placeholder, approx_kernel, prior_kernel, batch_size, latent_size)
	# p_K= calc_gp_kl(latent_mean, sequence_sizes_placeholder, approx_kernel, prior_kernel, batch_size, latent_size)
	latent_sample = tf.identity(latent_sample, name='latent_sample')
	sum_gp_kl = tf.identity(sum_gp_kl, name='gp_kl_sum')

	x_decode = vae_decode(latent_sample, latent_size)
	x_decode = tf.identity(x_decode, name='x_decode')

	# define the loss function (KL-div + reconst error)
	sum_recon_loss = -tf.reduce_sum(x*tf.log(1e-10+x_decode) + (1.0-x)*tf.log(1.0-x_decode+1e-10),name='sum_recon_loss')
	beta = 1.0
	loss = tf.add(sum_recon_loss, tf.scalar_mul(beta, sum_gp_kl), name='loss') 

	train_step = tf.train.AdamOptimizer(2e-4).minimize(loss)

	# logs_path = './VAE_GP_graphs_and_models/'
	# model_path = logs_path + 'GP_model'
	# write = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

	steps = 300000
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		inputs = {x:small_data_mat, sequences:true_sequences_mat, sequence_sizes_placeholder:sequence_sizes}
		# print(p_K.eval(feed_dict=inputs))
		# print(p_K.eval(feed_dict=inputs).shape)
		# print(length.eval(feed_dict=inputs))
	# 	#make a saver to save the graph 
	# 	saver = tf.train.Saver()
		for i in range(steps):
	# 		batch = MovingMnist.data_batch('train')
	# 		batch = np.reshape(batch,[-1,4096])
	# 		test_sequences = np.reshape(test_sequences, [-1,20])
	# 		# batch = MovingMnist.data_batch('mixed_train') # 20x5x64x64
	# 		# batch = MovingMnist.data_batch('train')
	# 		# batch = np.reshape(batch,[-1,4096]) # 100x64x64
			train_step.run(feed_dict=inputs)
			if i % 5000 == 0:
				plt.imshow(small_data_mat[0:514,:].T)
				plt.show()
				decoded = x_decode.eval(feed_dict=inputs).T
				single_seq = decoded[:,:sequence_sizes[0]]
				plt.imshow(single_seq)
				plt.show()
				print(i)
				print(sum_recon_loss.eval(feed_dict=inputs))
				print(sum_gp_kl.eval(feed_dict=inputs))
				# print(sum_recon_loss.eval(feed_dict={x:batch}))
			# if i % 25000 == 0:
				# saver.save(sess, model_path, global_step=i)
				# print(gp_kl.eval(feed_dict={x:batch}))
			# if i % 300000 == 0:
			
			# 	plt.imshow(img)
			# 	plt.show()
			# 	test = x_decode.eval(feed_dict={x:batch})
			# 	test = tf.reshape(test, [-1,64,64])
			# 	for i in range(20):
			# 		plt.imshow(test.eval()[i])
			# 		plt.show()

			

if __name__ == '__main__':
	main()
