import tensorflow as tf
import numpy as np
from DataHandler import DataHandler, plot_data_tc
import matplotlib.pyplot as plt
import random

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
	L = np.linalg.cholesky(data_K + 0.00005*np.eye(len(z_d_times)))
	length = len(full_seq_times)
	K_s = kernel_matrix(z_d_times, full_seq_times)
	Lk = np.linalg.solve(L, K_s)
	mu = np.dot(Lk.T, np.linalg.solve(L, z_d))
	K_ss = kernel_matrix(full_seq_times, full_seq_times)
	L = np.linalg.cholesky(K_ss + 1e-6*np.eye(length) - np.dot(Lk.T, Lk))
	f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(length,1)))
	f_post = f_post.reshape(1,length)
	if mean:
		return mu.reshape(1,-1)
	else:

		return f_post

def trans_split_latent(latent, sequence_indexes):
	latent = latent.T
	return np.split(latent, sequence_indexes, axis=1)

def drop_sample_VAE_prior(split_trans_matrix, droprate=0.3):
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

def drop_part_of_sequences(split_trans_matrix, droprate=0.3):
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
	batch = []
	for sequence,indicies in zip(t_s_matrix,sequence_times):
		sequence_sample = []
		for row in sequence:
			sequence_sample.append(sample_given_part_latent(row, indicies, full_seqeunce_times, mean))
		sequence_sample = np.concatenate(sequence_sample, axis=0).T
		batch.append(sequence_sample)
	batch = np.concatenate(batch,axis=0)
	return batch

def make_timeseries_plot(large_image, kept_indicies, full_sequence, name):
	plt.imshow(large_image, cmap='gray')
	xlabels = [str(ele) if ele in kept_indicies[0] else str(ele)+'*' for ele in full_sequence]
	x = list(range(large_image.shape[1]))
	plt.xticks(np.arange(min(x)+32, max(x)+32, 64), xlabels)
	ylabels = ['dropped', 'not dropped']
	y = list(range(large_image.shape[0]))
	plt.yticks(np.arange(min(y)+32, max(y)+32, 64), ylabels)
	plt.tight_layout()
	plt.savefig(name,format='svg', dpi=1200)

def main():
	data_path = "/Users/ethanevans/Documents/Externship/Data/"
	data_file = "mnist_test_seq.npy"
	# batch size - must be the same as when trained! 
	batch_size = 5
	MovingMnist = DataHandler(data_path, data_file, batch_size=batch_size)
	MovingMnist.make_discrete(MovingMnist.datasets['train'][0])
	MovingMnist.make_discrete(MovingMnist.datasets['test'][0])
	
	### To plot a test image sequence:
	# batch = MovingMnist.data_batch('test')
	# batch = np.reshape(batch,[-1,64,64])
	# large_image = np.zeros((64,64*20))
	# for i in range(20):
	# 	large_image[:,i*64:(i+1)*64] = batch[i,:,:]
	# plt.imshow(large_image, cmap='gray')
	# plt.show()

	### To shuffle the test dataset 
	# MovingMnist._shuffle_data(MovingMnist.datasets['test'][0])

	#REBUILD the graph:
	meta = '/Users/ethanevans/Documents/Externship/Models/GP_model-500000.meta'
	sess = tf.Session()
	saver = tf.train.import_meta_graph(meta)
	saver.restore(sess, tf.train.latest_checkpoint('./'))
		
	#For the 'GP prior' and 'vanilla VAE' the following are the same
	graph = tf.get_default_graph()
	x = graph.get_tensor_by_name('x:0')
	latent_mean = graph.get_tensor_by_name('latent_mean:0')	
	latent_log_variance = graph.get_tensor_by_name('latent_variance:0')	
	latent_sample = graph.get_tensor_by_name('latent_sample:0')
	x_decode = graph.get_tensor_by_name('x_decode:0')


	mean = True #this is used for reconstruction after dropping, after solving for the means at the 
	# given dropped time points do you sample (False) to generate the Zs or do you just take the mean values (True)
	# the sampling procedure does not really give informative results but the mean = True does
 	drop_rate = 0.5 # what percent of the sequence is dropped?
	full_sequence =  [ele for ele in range(1,21)]
	large_image = np.zeros((128, 64*len(full_sequence)))
	K = kernel_matrix(full_sequence, full_sequence)

	### calculate and plot activations for either model
	# print('Calculating Activations')
	# test_data_size = 1000
	# num_samples = 100
	# max_time=20
	# latent_size=100
	# full_expt_list = []
	# for i in range(int(test_data_size/batch_size)):
	# 	batch_xs = MovingMnist.data_batch('test')
	# 	batch_xs = np.reshape(batch_xs, [-1,4096])
	# 	inputs = {x:batch_xs}
	# 	batch_means = sess.run(latent_mean, inputs)
	# 	batch_log_var = sess.run(latent_log_variance, inputs)
	# 	expectation = np.zeros((batch_size*max_time, latent_size))
	# 	for j in range(num_samples):
	# 		batch_sample_pt = sess.run(latent_sample, {latent_mean:batch_means, latent_log_variance:batch_log_var})
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

	### TO ANALYZE AN IMAGE AFTER DROPPING PART OF THE LATENT AND SAMPLING FOR THOSE POINTS 
	### FROM THE GP POSTERIOR AND THEN RECONSTRUCTING WITH PLOTTNG....USE THE FOLLOWING CODE
	print('test image sequence, with dropping, GP posterior')
	batch = MovingMnist.data_batch('test')
	batch = np.reshape(batch,[-1,4096])
	latent = sess.run(latent_mean, {x:batch})
	basic_seq_index = [20,40,60,80]
	alt_lat = trans_split_latent(latent, basic_seq_index)
	sub_seq_lat, kept_indicies = drop_part_of_sequences(alt_lat, droprate=drop_rate)
	
	latent_dropped = post_gp_sample(sub_seq_lat, kept_indicies, full_sequence, mean)
	reconst = sess.run(x_decode, {latent_sample:latent_dropped})
	images = np.reshape(reconst, [-1,64,64])
	for i in range(20):
		large_image[:64, i * 64: (i + 1) * 64] = images[i,:,:]
	reconst = sess.run(x_decode, {latent_sample:latent})
	images = np.reshape(reconst, [-1,64,64])
	for i in range(20):
		large_image[64:128, i * 64: (i + 1) * 64] = images[i,:,:]
	make_timeseries_plot(large_image, kept_indicies, full_sequence, './GP_prior_dropped_30percent_mean.svg')

	### SAME THING BUT WITH MEAN OFF
	# mean=False
	# latent_dropped = post_gp_sample(sub_seq_lat, kept_indicies, full_sequence, mean)
	# reconst = sess.run(x_decode, {latent_sample:latent_dropped})
	# images = np.reshape(reconst, [-1,64,64])
	# for i in range(20):
	# 	large_image[:64, i * 64: (i + 1) * 64] = images[i,:,:]
	# reconst = sess.run(x_decode, {latent_sample:latent})
	# images = np.reshape(reconst, [-1,64,64])
	# for i in range(20):
	# 	large_image[64:128, i * 64: (i + 1) * 64] = images[i,:,:]
	# make_timeseries_plot(large_image, kept_indicies, full_sequence, './GP_prior_dropped_30percent_notmean.svg')


	### TO ANALYZE THE VAE AFTER DROPPING PART OF THE LATENT AND SAMPLING FOR THOSE POINT FROM 
	### THE NORMAL VAE PRIOR, USE THE FOLLOWING CODE
	# batch = MovingMnist.data_batch('test')
	# batch = np.reshape(batch,[-1,4096])
	# latent = sess.run(latent_mean, {x:batch})
	# basic_seq_index = [20,40,60,80]
	# alt_lat = trans_split_latent(latent, basic_seq_index)
	# part_VAE_prior_latent, kept_indicies = drop_sample_VAE_prior(alt_lat, droprate=0.3)
	# reconst = sess.run(x_decode, {latent_sample:part_VAE_prior_latent})
	# images = np.reshape(reconst, [-1,64,64])
	# for i in range(20):
	# 	large_image[:64, i * 64: (i + 1) * 64] = images[i,:,:]
	# reconst = sess.run(x_decode, {x:batch})
	# images = np.reshape(reconst, [-1,64,64])
	# for i in range(20):
	# 	large_image[64:128, i * 64: (i + 1) * 64] = images[i,:,:]
	# make_timeseries_plot(large_image, kept_indicies, full_sequence, './VAE_dropped_30percent.svg')


	### TO LOOK AT A SINGLE TEST IMAGE WITHOUT GP SAMPLING USE THE FOLLOWING:
	# print('Test image sequence')
	# batch = MovingMnist.data_batch('test')
	# batch = np.reshape(batch,[-1,4096])
	# train_images = sess.run(x_decode, {x:batch})
	# train_images = np.reshape(train_images, [-1,64,64])
	# for i in range(10):
	# 	plt.imshow(train_images[i,:,:])
	# 	plt.show()

	### TO LOOK AT A SINGE TEST IMAGE THAT IS REPEATED FOR THE WHOLE BATCH WITH DIFFERENT 
	### RECONSTRUCTIONS (DUE TO DIFFERENT SAMPLES) USE THE FOLLOWING
	# print('Single test image repeated')
	# batch = MovingMnist.data_batch('test')
	# batch = np.reshape(batch,[-1,4096])
	# latent_mean = sess.run(latent_mean, {x:batch})
	# latent_log_var = sess.run(latent_log_variance, {x:batch})
	# first_mean = latent_mean[0]
	# first_log_var = latent_log_var[0]
	# repeated_img = embedding_sampling(first_mean, first_log_var)
	# print(repeated_img.shape)
	# out_images = sess.run(x_decode, {latent_sample:repeated_img})
	# test = np.reshape(out_images, [-1,64,64])
	# for i in range(10):
	# 	plt.imshow(test[i,:,:])
	# 	plt.show()

if __name__ == "__main__":
	main()