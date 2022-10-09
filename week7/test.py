import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def mean():
	mean_taget = 10


	r = np.array(range(0,500))
	mean_r = r.mean()
	r = r/(mean_r/mean_taget) ### make data that have mean what we want
	mean_r = r.mean()

	print(mean_taget)
	print(mean_r)

def std():
	std_taget = 15


	r = np.array(range(0,500))
	std_r = r.std()
	r = r/(std_r/std_taget) ### make data that have mean what we want
	std_r = r.std()

	print(std_taget)
	print(std_r)

def combin():
	std_taget = 15
	mean_taget = 10


	r = np.array(range(0,500))
	std_r = r.std()
	mean_r = r.mean()
	r = r/( (mean_r/mean_taget) ) ### make data that have mean what we want
	std_r = r.std()
	mean_r = r.mean()

	print(std_taget)
	print(std_r)
	print(mean_taget)
	print(mean_r)

def genarate_data(desired_mean, desired_std_dev, num_samples):
	desired_mean = desired_mean
	desired_std_dev =desired_std_dev
	num_samples = 500

	samples = np.random.normal(size=num_samples)
	#samples = np.array(range(0,num_samples))
	actual_mean = np.mean(samples)
	actual_std = np.std(samples)
	
	zero_mean_samples = samples - (actual_mean)
	zero_mean_std = np.std(zero_mean_samples)
	scaled_samples = zero_mean_samples * (desired_std_dev/zero_mean_std) ###### key
	
	final_samples = scaled_samples + desired_mean
	final_mean = np.mean(final_samples)
	final_std = np.std(final_samples)
	
	return final_samples

# parameter
start = -100
end = 100
ker1, mu1 = genarate_data(-30, 5, end-start).std(), genarate_data(-30, 5, end-start).mean()
ker2, mu2 = genarate_data(30, 15, end-start).std(), genarate_data(30, 15, end-start).mean()
Prior_1 = 0.5
Prior_2 = 1-Prior_1

plt.plot(genarate_data(-30, 5, end-start), genarate_data(30, 15, end-start), "o")
plt.show()


