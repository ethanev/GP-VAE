import tensorflow as tf
import numpy as np
from DataHandler import DataHandler, plot_data_tc
import matplotlib.pyplot as plt
import random
from scipy.stats import norm

def kernel_function(t1,t2):
	noise = 1e-3
	if t1 != t2:
		noise = 0.0
	signal = 1.0-noise
	time_char = 1.0
	return signal*np.exp(-np.power((float(t1)-float(t2)),2)/(2.0*np.power(time_char,2))) + noise

def kernel_matrix(sequence1, sequence2):
	k_mat = np.zeros((len(sequence1), len(sequence2)), dtype=np.float32)
	ax_1, ax_2 = k_mat.shape
	for i, ele1 in enumerate(sequence1):
		for j, ele2 in enumerate(sequence2):
			k_mat[i,j] = kernel_function(ele1,ele2)
	return k_mat

def prior_gp_sample(times, kernel_matrix):
	x_space = np.linspace(1,20,20)
	L = np.linalg.cholesky(kernel_matrix + 1e-15*np.eye(times))
	f_prior = np.dot(L, np.random.normal(size=(times,500)))
	f_prior_split = np.split(f_prior, 5, axis=1)
	f_prior_concat = np.concatenate(f_prior_split, axis=0)
	return f_prior_concat

def embedding_sampling(mean, log_var):
	sample_in_batch = 100
	var = np.exp(log_var)
	repeated_mean = np.reshape(np.tile(mean,sample_in_batch),[-1,mean.shape[0]])
	repeated_var = np.reshape(np.tile(var,sample_in_batch),[-1,var.shape[0]])
	random = np.random.normal(scale=1.0, size=(sample_in_batch,mean.shape[0]))
	return repeated_mean + np.multiply(random,repeated_var)

def sample_given_part_latent(z_d, z_d_times, full_seq_times, mean=False):
	#given a partial z_d use this knowledge of the means along with the GP to sample
	# the rest of the latents
	data_K = kernel_matrix(z_d_times, z_d_times)
	L = np.linalg.cholesky(data_K) # + 1e-15*np.eye(len(z_d_times)))
	length = len(full_seq_times)
	K_s = kernel_matrix(z_d_times, full_seq_times)
	Lk = np.linalg.solve(L, K_s)
	mu = np.dot(Lk.T, np.linalg.solve(L, z_d))
	K_ss = kernel_matrix(full_seq_times, full_seq_times)
	L = np.linalg.cholesky(K_ss+ 1e-15*np.eye(length)  - np.dot(Lk.T, Lk)) 
	f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(length,1)))
	f_post = f_post.reshape(1,length)
	if mean:
		return mu.reshape(1,-1)
	else:
		return f_post

def trans_split_latent(latent, num_splits):
	latent = latent.T
	return np.split(latent, num_splits, axis=1)

def drop_sample_VAE_prior(split_trans_matrix, droprate=0.5):
	batch = []
	all_ind = []
	for seq in split_trans_matrix:
		print(seq)
		num_to_drop = droprate * seq.shape[1]
		sequence_ind = list(range(seq.shape[1]))
		drop_inds = random.sample(sequence_ind, int(num_to_drop))
		for ind in sorted(drop_inds):
			seq[:,ind] = np.random.normal(scale=1.0, size=(seq.shape[0],))
		batch.append(seq.T)
		print(seq)
		keep_inds = [ele for ele in sequence_ind if ele not in drop_inds]
		shifted_ind = [ele+1 for ele in sorted(keep_inds)]
		all_ind.append(shifted_ind)
	batch = np.concatenate(batch)
	return batch, all_ind

def drop_part_of_sequences(split_trans_matrix, droprate=0.5):
	all_seqs = []
	all_ind = []
	for seq in split_trans_matrix:
		keep = []
		num_to_keep = seq.shape[1] - droprate * seq.shape[1]
		sequence_ind = list(range(seq.shape[1]))
		keep_inds = random.sample(sequence_ind, int(num_to_keep))
		for ind in sorted(keep_inds):
			keep.append(np.reshape(seq[:,ind],(-1,1)))
		sub_seq = np.concatenate(keep, axis=1)
		all_seqs.append(sub_seq)
		shifted_ind = [ele+1 for ele in sorted(keep_inds)]
		all_ind.append(shifted_ind)
	return all_seqs, all_ind

def post_gp_sample(t_s_matrix,sequence_times, full_seqeunce_times, mean=False):
	'''
	Note: This is not actually using the learned kernel matrix (learned time char)
			it just makes a stock prior kernel matrix (time char = 1.0)
			Need to alter it so you can pass in the correct kernel matrix (from learned time char) for each latent 
			similar to whats happening in 'single_batch_random_single_latent_fromGPapprox'
	'''
	batch = []
	for sequence,indicies in zip(t_s_matrix,sequence_times):
		sequence_sample = []
		for row in sequence:
			sequence_sample.append(sample_given_part_latent(row, indicies, full_seqeunce_times, mean))
		sequence_sample = np.concatenate(sequence_sample, axis=0).T
		batch.append(sequence_sample)
	batch = np.concatenate(batch,axis=0)
	return batch

def make_timeseries_plot(large_image, kept_indicies, full_sequence):
	plt.imshow(large_image, cmap='gray')
	xlabels = [str(ele) if ele in kept_indicies[0] else str(ele)+'*' for ele in full_sequence]
	x = list(range(large_image.shape[1]))
	plt.xticks(np.arange(min(x)+32, max(x)+32, 64), xlabels)
	ylabels = ['dropped', 'not dropped']
	y = list(range(large_image.shape[0]))
	plt.yticks(np.arange(min(y)+32, max(y)+32, 64), ylabels)
	plt.tight_layout()
	plt.show()

def random_single_latent(latent, latent_shape, randomize_location):
	'''
	Gives 100 samples of one image with the latent varried over all 100
	'''
	single_latent = latent[10,:] #row of length latent size, each entry is a single latent variable, 10 is just a random image in the 100 images
	full_single_latent = np.tile(single_latent, (latent_shape[0],1)) 
	random_noise = np.linspace(norm.ppf(0.00001), norm.ppf(0.99999),latent_shape[0])
	# random_noise = np.random.normal(scale=2.0, size=[latent_shape[0]])
	# random_noise = np.sort(random_noise)
	full_single_latent[:,randomize_location] = random_noise
	return full_single_latent

def single_batch_random_single_latent(latent, latent_shape, randomize_location, batch_size):
	'''
	Gives 20 samples of one image with the latent varried over all 20 
	'''
	single_latent = latent[12,:]
	full_single_latent = np.tile(single_latent, (20,1))
	random_noise = np.linspace(norm.ppf(0.000001), norm.ppf(0.999999),20)
	full_single_latent[:,randomize_location] = random_noise
	full_single_latent = np.tile(full_single_latent, (batch_size,1))
	return full_single_latent

def single_batch_random_single_latent_fromGPapprox(latent, latent_shape, gp_location, batch_size, gp_sample):
	single_latent = latent[12,:]  # 12 is the index of a given image's latents (choose any in 0-99)
	full_single_latent = np.tile(single_latent, (20,1))
	full_single_latent[:,gp_location] = gp_sample # alter the latents to be from the gp sample 
	full_single_latent = np.tile(full_single_latent, (batch_size,1))
	return full_single_latent

def main():
	data_path = "/Users/ethanevans/Documents/Externship/Data/"
	data_file = "mnist_test_seq.npy"
	# batch size - must be the same as when trained
	batch_size = 5 
	max_time = 20 #sequence time length 
	beta_value = 1.0
	latent_size = 100
	test_data_size = 1000
	MovingMnist = DataHandler(data_path, data_file, batch_size=batch_size, time_included=True)
	MovingMnist.make_discrete(MovingMnist.datasets['train'][0])
	MovingMnist.make_discrete(MovingMnist.datasets['test'][0])
	# MovingMnist._shuffle_data(MovingMnist.datasets['test'][0])

	#REBUILD the graph:
	meta = '/Users/ethanevans/Documents/Externship/Models/GP_model_noprior-125000.meta'
	sess = tf.Session()
	saver = tf.train.import_meta_graph(meta)
	saver.restore(sess, tf.train.latest_checkpoint('./'))
		
	#for either model these are the same 
	graph = tf.get_default_graph()
	x = graph.get_tensor_by_name('x:0')
	sequences = graph.get_tensor_by_name('sequences:0')
	sequence_sizes_placeholder = graph.get_tensor_by_name('sequence_lengths:0')
	beta = tf.placeholder(tf.float64, [])
	latent_mean = graph.get_tensor_by_name('latent_mean:0')		
	latent_sample = graph.get_tensor_by_name('latent_sample:0')
	approx_kernel = graph.get_tensor_by_name('approx_kernels:0')
	chol_noise = graph.get_tensor_by_name('chol_noise:0')
	x_decode = graph.get_tensor_by_name('x_decode:0')

	mean = False
	drop_rate = 0.5
	full_sequence =  [ele for ele in range(1,21)]
	large_image = np.zeros((128, 64*len(full_sequence)))
	K = kernel_matrix(full_sequence, full_sequence)

	### To show a test image sequence reconstruction
	# batch_xs, batch_time_steps, batch_lengths = MovingMnist.data_batch('test')
	# inputs = {x:batch_xs, sequences:batch_time_steps, sequence_sizes_placeholder:batch_lengths}
	# print('test image sequence')
	# test_images = sess.run(x_decode, inputs)
	# large_image = np.zeros((64, 64*20))
	# test_images = np.reshape(test_images, [-1,64,64])
	# for j in range(20):
	# 	large_image[:, j * 64: (j + 1) * 64] = test_images[j,:,:]
	# plt.imshow(large_image)
	# plt.show()

	### calculate the activations of the latents using a sample of 100 for each image. 
	# num_samples = 100
	# full_expt_list = []
	# for i in range(int(test_data_size/batch_size)):
	# 	batch_xs, batch_time_steps, batch_lengths = MovingMnist.data_batch('test')
	# 	inputs = {x:batch_xs, sequences:batch_time_steps, sequence_sizes_placeholder:batch_lengths}
	# 	batch_means = sess.run(latent_mean, inputs)
	# 	expectation = np.zeros((batch_size*max_time, latent_size))
	# 	for j in range(num_samples):
	# 		batch_noise_pt = sess.run(chol_noise, inputs)
	# 		batch_sample_pt = sess.run(latent_sample, {latent_mean:batch_means, chol_noise:batch_noise_pt, sequences:batch_time_steps, sequence_sizes_placeholder:batch_lengths})
	# 		expectation += batch_sample_pt
	# 	expectation = expectation / num_samples
	# 	full_expt_list.append(expectation)
	# full_exp_mat = np.concatenate(full_expt_list)
	# #now get variance of the latent state over the whole test data set:
	# latent_var = []
	# for i in range(latent_size):
	# 	latents = full_exp_mat[:,i]
	# 	latents = np.squeeze(latents)
	# 	latent_var.append(np.cov(latents))
	# xs = [i for i in range(1,101)]
	# plt.plot(xs, np.sort(latent_var), 'bo')
	# plt.show()


	#### FOR THE FULL GP MODEL (PRIOR AND LATENT SAMPLING)
	### for moving around in the latent space for all of the latents individually (from the 'for z in'...loop below) 
	### There are three functions:
	###      random_single_latent (make the whole batch of images (100) one image where its latent space for ONE latent is explored through inv. CDF of a N(0,1) gaussian)
	###		 single_batch_random_single_latent (same a above but the space is explored in only 20)
	###		 single_batch_random_single_latent_fromGPapprox (same idea but instead of each latent over time coming from N(0,1) its from the LEARNED GP kernel matrix)
	means = np.zeros((20,))
	test_approx_kernel = sess.run(approx_kernel, inputs)
	latent_sample_batch = sess.run(latent_sample, inputs)
	for z in range(latent_sample_batch.shape[1]):
		###the following 4 lines are for if you want to explored the latent via the GP Kernel matrix
		#single_kernel = test_approx_kernel[z,:]
		#single_kernel_mat = np.reshape(single_kernel, (20,20))
		#new_lat_z_vs_time = np.random.multivariate_normal(means, single_kernel_mat)
		#new_lat_z_vs_time = np.squeeze(new_lat_z_vs_time)
		# single_latent_space = single_batch_random_single_latent_fromGPapprox(latent_sample_batch, latent_sample_batch.shape, z, batch_size, new_lat_z_vs_time)
		# Or comment out the next line: (can also switch it to 'single_batch_random() if you want finer exploration')
		single_latent_space = single_batch_random_single_latent(latent_sample_batch, latent_sample_batch.shape, z, batch_size)
		test_images = sess.run(x_decode, {latent_sample:single_latent_space})
		large_image = np.zeros((64, 64*max_time))
		test_images = np.reshape(test_images, [-1,64,64])
		for j in range(20):
			large_image[:, j * 64: (j + 1) * 64] = test_images[j,:,:]
		plt.imshow(large_image)
		plt.savefig('./small_latent_%d_random.svg' %(z), format='svg', dpi=1200)
		plt.gcf().clear()



	##### NOTE STILL DEBUGGING THIS: not currently working (matrix inversion issues )
	full_gp = True
	### TO ANALYZE AN IMAGE AFTER DROPPING PART OF THE LATENT AND SAMPLING FOR THOSE POINTS 
	### FROM THE GP POSTERIOR AND THEN RECONSTRUCTING WITH PLOTTNG....USE THE FOLLOWING CODE
	print('test image sequence, with dropping, GP posterior')
	if full_gp:
		batch_xs, batch_time_steps, batch_lengths = MovingMnist.data_batch('test')
		# batch_xs, batch_time_steps, batch_lengths = MovingMnist.data_batch('test') #just to get a differnt image!
		inputs = {x:batch_xs, sequences:batch_time_steps, sequence_sizes_placeholder:batch_lengths}
		latent = sess.run(latent_mean, inputs)
		true_latent_sample = sess.run(latent_sample, inputs)
	else:
		# This actually wont be used due to how these dynamic time models were made and is just a remnant
		batch = MovingMnist.data_batch('test')
		batch = np.reshape(batch,[-1,4096])
		latent = sess.run(latent_mean, {x:batch})

	alt_lat = trans_split_latent(latent, batch_size)
	sub_seq_lat, kept_indicies = drop_part_of_sequences(alt_lat, droprate=drop_rate)

	latent_dropped = post_gp_sample(sub_seq_lat, kept_indicies, full_sequence, mean)
	reconst = sess.run(x_decode, {latent_sample:latent_dropped})
	images = np.reshape(reconst, [-1,64,64])
	for i in range(20):
		large_image[:64, i * 64: (i + 1) * 64] = images[i,:,:]

	print('test image sequence: without dropping')
	reconst = sess.run(x_decode, {latent_sample:true_latent_sample})
	images = np.reshape(reconst, [-1,64,64])
	for i in range(20):
		large_image[64:128, i * 64: (i + 1) * 64] = images[i,:,:]

	make_timeseries_plot(large_image, kept_indicies, full_sequence)

if __name__ == "__main__":
	main()