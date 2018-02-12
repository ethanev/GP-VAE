import numpy as np
import matplotlib.pyplot as plt

class DataHandler(object):
	'''
	Use this for moving Mnist, If you are working with one of the models that learns the time characteristic, use time_included=True 
		(for the VAE_GPprior_diag_cov.py use time_included=False)
	'''
	def __init__(self, data_path, data_file, batch_size=100, train_fraction=0.8, randomize_data=False, time_included=False):
		self.time_included = time_included
		self.batch_size = batch_size
		self.randomize_data = randomize_data
		self.train_fraction = train_fraction
		self.data_path = data_path
		self.data_file = self.data_path + data_file
		self.valid_counter = 0
		self.train_counter = 0 
		self.test_counter = 0
		self.datasets = {}
		self._load_data()
		
	def data_batch(self, dataset):
		data, counter = self.datasets[dataset]
		# looks at a dataset specific counter to know where in the data set it is
		# if the counter + the batch size is greater than the size of the array
			# shuffle the dataset
			# reset the counter
		if counter + self.batch_size > data.shape[1]:
			# print('Reshuffling')
			self._shuffle_data(self.datasets[dataset][0])
			# data_test = self.datasets[dataset][0][:,0,:,:]
			# plot_data_tc(data_test)
			self.datasets[dataset][1] = 0
			counter = 0
		else:
			pass
			# print('counter value is: %d' %(counter))
		# pull the next batch and return 
		batch = data[:,counter:counter+self.batch_size,:,:]
		self.datasets[dataset][1] += self.batch_size
		if self.time_included:
			times = [float(i) for i in range(20)]
			times = np.asarray(times)
			times = np.reshape(np.tile(times, self.batch_size), [self.batch_size,-1])
			batch_lengths = self.batch_size*[20]
			data = np.swapaxes(batch,0,1)
			data = np.reshape(data, [-1,4096])
			# data = np.reshape(batch, [])
			return data, times, batch_lengths
		else:
			return np.swapaxes(batch,0,1)
	
	def make_shuffled_dataset(self):
		self.datasets['mixed_train'] = self.datasets['train']
		self.datasets['mixed_train'][0] = np.reshape(self.datasets['mixed_train'][0],[-1,64,64])
		np.random.shuffle(self.datasets['mixed_train'][0])
		self.datasets['mixed_train'][0] = np.reshape(self.datasets['mixed_train'][0],[20,8000,64,64])

	def make_cropped_dataset(self):
		self.datasets['cropped_train'] = self.datasets['train']
		self.datasets['cropped_train'][0] = self.datasets['cropped_train'][0][:,:,18:18+28, 18:18+28]
		# print(self.datasets['cropped_train'][0].shape)

	# def make_resized_dataset(self):
	# 	self.datasets['resized'] = self.datasets['train']
	# 	self.datasets['resized'] = resize(self.datasets['resized'][0][0,0,:,:], (28,28))
		
	def make_discrete(self, dataset):
		dataset[dataset >= 0.498] = 1
		dataset[dataset < 0.498] = 0

	def _load_data(self):
		self.raw_data = np.load(self.data_file)
		self._normalize_data()
		self._split_data()

	def _normalize_data(self):
		self.raw_data = (self.raw_data - 0) / 255

	def _shuffle_data(self, dataset):
		dataset = np.swapaxes(dataset,0,1) # switch to format where 1st dim is the data points
		np.random.shuffle(dataset)
		dataset = np.swapaxes(dataset,0,1) # switch back to where 1st dim is the 'time step'

	def _split_data(self):
		# shuffle the data and split into three datasets
		# currently this makes the test / validation sets split what is left after removing the train set
		if self.randomize_data:
			self._shuffle_data(self.raw_data)
		train_size = int(self.train_fraction * self.raw_data.shape[1])
		valid_size = int((self.raw_data.shape[1] - train_size) / 2)
		# perform the data splitting
		self.datasets['train'] = [self.raw_data[:,:train_size,:,:], self.train_counter]
		self.datasets['valid'] = [self.raw_data[:,train_size:train_size+valid_size,:,:], self.valid_counter]
		self.datasets['test']  = [self.raw_data[:,train_size+valid_size:,:,:], self.test_counter]

class SyntheticDataHandler():
	'''
	Use this class for all synthetic data trials! 
	'''
	def __init__(self, data, max_time, batch_size=5, train_fraction=0.9, randomize=False):
		self.randomize = randomize #ONLY USED FOR THE INITIAL DATA PREP. SHUFFLE DATA OR NO? (BEFORE SPLITTING TRAIN/TEST)
		self.train_fraction = train_fraction
		self.batch_size = batch_size
		self.max_time = max_time
		self.data = data
		self.datasets = {}
		self.databatches = {} #dictionary mapping 'test' or 'train' to lists of [xs, time_steps, lengths, counter]
		self._split_data()

	def data_batch(self, data_name):
		# assert data_name == 'test' or 'train'
		xs, time_steps, lengths, counter, x_index = self.databatches[data_name]
		size_of_data = len(lengths)
		if counter + self.batch_size > size_of_data:
			self._prep_dataset(self.datasets[data_name], data_name, randomize=True)
			xs, time_steps, lengths, counter, x_index = self.databatches[data_name]
		batch_lengths = lengths[counter:counter+self.batch_size]
		batch_time_steps = time_steps[counter:counter+self.batch_size, :]
		length_xs = sum(batch_lengths)
		batch_xs = xs[x_index:x_index+length_xs,:]
		self.databatches[data_name][3] += self.batch_size
		self.databatches[data_name][4] += length_xs
		return batch_xs, batch_time_steps, batch_lengths 
	
	def _randomize(self, dataset):
		np.random.shuffle(dataset)

	def _prep_dataset(self, dataset, name, randomize=False):
		'''
		input: a matrix of data Nx15x45 (15 is the size of a singe time step, 45 the number of time steps
		Returns a list of batches 
		'''
		self.databatches[name] = []
		true_sequences = []
		data = []
		sequence_sizes = []
		length = dataset.shape[0]
		if randomize:
			self._randomize(dataset)
		for i in range(length):
			# true_sequences.append(np.reshape(np.where(self.data['x'][i,0,:] > -1)[0],[1,-1])) # time indicies where the sequence is valid for the kernel matrix
			true_sequences.append(np.reshape(self.data['time'][np.where(dataset[i,0,:] > -1)],[1,-1]))
			# build the actual data to be used after removing columns that are not data
			data.append(np.reshape(dataset[i,:,:][np.where(dataset[i,:,:] > -1)],[15,-1]).T)#new test is a list of 5, indiv_length X 15 matricies
			# build a list of the sizes for each sequecne to know where to split the data matrix in the VAE
			sequence_sizes.append(np.reshape(dataset[i,:,:][np.where(dataset[i,:,:] > -1)],[15,-1]).T.shape[0]) 
		data_mat = np.concatenate(data, axis=0) #shape: total_times_in_all_batches x 15
		true_sequences_mat = []
		for seq in true_sequences:
			true_sequences_mat.append(np.pad(seq,((0,0),(0,self.max_time-seq.shape[1])), 'constant', constant_values=0))
		true_sequences_mat = np.concatenate(true_sequences_mat)
		counter = 0
		x_index = 0
		all_data = [data_mat, true_sequences_mat, sequence_sizes, counter, x_index]
		self.databatches[name].extend(all_data)

	def _split_data(self):
		if self.randomize:
			np.random.shuffle(self.data['x'])
		size = int(self.data['x'].shape[0] * self.train_fraction)
		self.datasets['train'] = self.data['x'][:size,:,:]
		self.datasets['test'] = self.data['x'][size:,:,:]
		self._prep_dataset(self.datasets['train'], 'train')
		self._prep_dataset(self.datasets['test'], 'test')
		
def plot_data_tc(indiv_data_traj):
	for i in range(indiv_data_traj.shape[0]):
		plt.imshow(indiv_data_traj[i,:,:])
		plt.show()

def main():
	# define data location
	data_path = "/Users/ethanevans/Documents/Externship/Data/"
	data_file = "mnist_test_seq.npy"
	# Instantiate data handler object and read the data
	MovingMnist = DataHandler(data_path, data_file, batch_size=2)
	MovingMnist.make_shuffled_dataset()



if __name__ == '__main__':
	main()
