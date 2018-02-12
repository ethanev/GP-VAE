from sklearn.externals import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from DataHandler import DataHandler

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

def vae_sample(mean, log_variance, batch_size):
	# keep a vector tell you the time length of all sequences in the batch
	mean = tf.reshape(mean, [-1, 100])
	log_variance = tf.reshape(log_variance, [-1, 100])
	var = tf.exp(log_variance)
	noise = tf.random_normal([batch_size*20,100])
	sample = mean + tf.sqrt(var)*noise
	kl = 0.5*tf.reduce_sum(1.0+tf.log(1e-10+var)-tf.square(mean)-var, 1)
	return sample, kl

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
 
def write_file(losses, step, model_type='train'):
	file = open('./No_GP_models/%s_loss_file_%d.csv' %(model_type, step), 'w')
	for step in losses:
		for ele in step:
			file.write(ele + ',')
		file.write('\n')
	file.close()

def main():

	# TO RUN THIS MAKE SURE YOU MAKE THE FOLLOWING DIRECTORY IN THE CURRENT LOCATION:
	#      No_GP_models
	# ALTER THE PATHS RIGHT BELOW THIS FOR WHERE THE DATA IS!
	# IF YOU WANT NICE FIGURES: CHANGE .PNG TO .SVG and alter the format

	#on local comp
	# data_path = "/Users/ethanevans/Documents/Externship/Data/"
	# on VM:
	data_path = "/home/ethan/Data/"
	data_file = "mnist_test_seq.npy"
	# how many full sequences you want. 
	batch_size = 5
	MovingMnist = DataHandler(data_path, data_file, batch_size=batch_size)
	# MovingMnist.make_shuffled_dataset() # comment out for sequence training

	# make the model
	x = tf.placeholder(tf.float32, [None,4096], name='x')
	latent_mean, latent_log_variance = vae_encode(x)
	latent_mean = tf.identity(latent_mean, name='latent_mean')
	latent_log_variance = tf.identity(latent_log_variance, name='latent_variance')

	latent_sample, kl = vae_sample(latent_mean, latent_log_variance, batch_size)
	latent_sample = tf.identity(latent_sample, name='latent_sample')
	kl = tf.identity(kl, name='kl')

	x_decode = vae_decode(latent_sample)
	x_decode = tf.identity(x_decode, name='x_decode')

	# define the loss function (KL-div + reconst error)
	reconstruct_loss = -tf.reduce_sum(x*tf.log(1e-10+x_decode) + (1.0-x)*tf.log(1.0-x_decode+1e-10), 1) #vector of reconstruction losses where each entry in the loss on a single image
	summed_recon_loss = tf.reduce_sum(reconstruct_loss, name='sum_recon_loss') 
	summed_kl_loss = -tf.reduce_sum(kl, name='kl_loss')
	beta = 1.0
	loss = tf.add(summed_recon_loss, tf.scalar_mul(beta, summed_kl_loss), name='loss')
	train_step = tf.train.AdamOptimizer(2e-4).minimize(loss)

	steps = 5000000
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		#### THE NEXT CODE IS FOR DOWNSIZING THE INPUT IMAGES TO 28X28 ON THE TRAINING DATA
		# large_data = MovingMnist.datasets['mixed_train'][0][:,:,:,:]
		# MovingMnist.datasets['mixed_train'][0] = data.eval(feed_dict={test:large_data})
		# MovingMnist.datasets['mixed_train'][0] = np.reshape(MovingMnist.datasets['mixed_train'][0], (-1,8000,28,28))
		
		### uncomment the next line to train on just 2 batch of 20 to show that the net works (OVER FIT ON IT)
		# MovingMnist.datasets['mixed_train'][0] = MovingMnist.datasets['mixed_train'][0][:20,:1,:,:]
		# print(MovingMnist.datasets['mixed_train'][0].shape)
		# plt.imshow(MovingMnist.datasets['mixed_train'][0][0,0,:,:])
		# plt.show()

		# Make a saver to save the graph and session
		saver = tf.train.Saver()
		save_path = "./No_GP_models/VAE_plain"

		# Make the input data discrete (0 or 1)
		MovingMnist.make_discrete(MovingMnist.datasets['train'][0])
		MovingMnist.make_discrete(MovingMnist.datasets['test'][0])
		### If you want to work with the mixed training set and make it discrete use the following:
		# MovingMnist.make_discrete(MovingMnist.datasets['mixed_train'][0])
		train_losses = [['Step', 'Reconstruction loss', 'KL loss']]
		test_losses = [['Step', 'Reconstruction loss', 'KL loss']]
		for i in range(steps):
			# batch = MovingMnist.data_batch('mixed_train') # 20x5x64x64
			batch = MovingMnist.data_batch('train')
			batch = np.reshape(batch,[-1,4096]) # 100x64x64 --> 100,4096 (it reshapes back in the encode)
			train_step.run(feed_dict={x:batch})
			if i % 500 == 0:
				step_losses = []
				step_losses.append(str(i))
				step_losses.append(str(summed_recon_loss.eval(feed_dict={x:batch})))
				step_losses.append(str(summed_kl_loss.eval(feed_dict={x:batch})))
				train_losses.append(step_losses)
			if i % 10000 == 0:
				batch = MovingMnist.data_batch('test')
			
				large_image = np.zeros((64, 64*20))
				batch = np.reshape(batch, [-1,64,64])
				for j in range(20):
					large_image[:, j * 64: (j + 1) * 64] = batch[j,:,:]
				plt.imshow(large_image)
				plt.savefig('./No_GP_models/input_sequence_%d.png' %(i))
				plt.gcf().clear()

				batch = np.reshape(batch, [-1,4096])
				test = x_decode.eval(feed_dict={x:batch})
				test = np.reshape(test, [-1,64,64])

				test_step_losses = []
				test_step_losses.append(str(i))
				test_step_losses.append(str(summed_recon_loss.eval(feed_dict={x:batch})))
				test_step_losses.append(str(summed_kl_loss.eval(feed_dict={x:batch})))
				test_losses.append(test_step_losses)

				large_image = np.zeros((64, 64*20))
				for j in range(20):
					large_image[:, j * 64: (j + 1) * 64] = test[j,:,:]
				plt.imshow(large_image)
				plt.savefig('./No_GP_models/output_sequence_%d.png' %(i))
				plt.gcf().clear()

				noise = np.random.normal(size=(100,100))
				samples = x_decode.eval(feed_dict={latent_sample:noise})
				sample = np.reshape(samples, [-1,64,64])
				large_image = np.zeros((64, 64*20))
				for j in range(20):
					large_image[:, j * 64: (j + 1) * 64] = sample[j,:,:]
				plt.imshow(large_image)
				plt.savefig('./No_GP_models/prior_sequence_%d.png' %(i))
				plt.gcf().clear()
			if i % 50000 == 0:
				saver.save(sess, save_path, global_step=i)
				write_file(train_losses, i)
				write_file(test_losses, i, type='test')

if __name__ == '__main__':
	main()