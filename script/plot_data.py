import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_pickle(pickle_file):
	ls = []
	with open(pickle_file,'rb') as fid:
		ls = pickle.load(fid)
	return ls


def load_pickles(pickle_file_list):
	ls = []
	for path in pickle_file_list:
		ls.append(load_pickle(path))
	return ls

paths = ['epochs.p', 'loss_loss.p', 'val_loss.p', 'bi_acc.p', 'val_bi_acc.p']
epochs, loss, val_loss, bi_acc, val_bi_acc = load_pickles(paths)

def show_loss():
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
	ax1.plot(epochs, loss, 'b-',epochs, val_loss, 'r-')
	ax1.set_title('Loss', fontsize=15)
	ax2.plot(epochs, bi_acc, 'b-', epochs, val_bi_acc, 'r-')
	ax2.legend(['Training', 'Validation'])
	ax2.set_title('Binary Accuracy (%)', fontsize=15)
	fig.savefig('loss_history')

show_loss()