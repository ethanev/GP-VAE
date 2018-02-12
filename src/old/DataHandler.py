import numpy as np
# If on VM comment out the following
import matplotlib.pyplot as plt
from skimage.transform import resize


class DataHandler(object):
	def __init__(self, data_path, data_file, batch_size=100, train_fraction=0.8, randomize_data=False):
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

		# data_test = self.data[:,0,:,:]
		# plot_data_tc(data_test)

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
	# for _ in range(20):
	# 	batch = MovingMnist.data_batch('test')
	# 	print(batch.shape)
	# test to read a data piece - tajubg akk 
	# data_test = MovingMnist.data[:,0,:,:]
	# plot_data_tc(data_test)


if __name__ == '__main__':
	main()
