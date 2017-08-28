import keras.callbacks

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

class AccCallback(keras.callbacks.Callback):

    def __init__(self,test_func, inputs, blank_label, batch_size, logs=False, name='net1'):
        self.test_func = test_func
        self.inputs = inputs
        self.blank = blank_label
        self.log_level = logs
        self.batch_size = batch_size
        self.history = { 'mean_ed': [], 'mean_norm_ed': [] }
        self.name = name

    def on_epoch_end(self, epoch, logs={}):
        directory = self.name + '/logs/'
        log_file = None
        if self.log_level == True:
            if not os.path.exists(directory):
                os.makedirs(directory)
            log_file = file(directory+'epoch_'+str(epoch)+'.log','w+')

        count = 0
        while count <= self.inputs['the_input'].shape[0]:
            if count==0:
                func_out = self.test_func([self.inputs['the_input'][count:count + self.batch_size]])[0]
            elif count + self.batch_size <= self.inputs['the_input'].shape[0]:
                func_out = np.append(func_out,
                        self.test_func([self.inputs['the_input'][count:count + self.batch_size]])[0],
                        axis=0)
            else:
                func_out = np.append(func_out,
                        self.test_func([self.inputs['the_input'][count:]])[0],
                        axis=0)
            count += self.batch_size
        ed = 0
        mean_ed = 0.0
        mean_norm_ed = 0.0
        for i in range(func_out.shape[0]):
            rnn_out = []
            output = []
            prev = -1
            for j in range(func_out.shape[1]):
                out = np.argmax(func_out[i][j])
                rnn_out.append(out)

                if out != prev and out != self.blank:
                    output.append(out)
                prev = out

            ed = self.levenshtein(self.inputs['the_labels'][i].tolist(),output)
            mean_ed += float(ed)
            mean_norm_ed += float(ed) / self.inputs['label_length'][i]

            if self.log_level == True:
                log_file.write('Test: \t' + str(self.inputs['the_labels'][i].tolist()) + ' \n')
                log_file.write('RNN output: \t' + str(rnn_out) + ' \n')
                log_file.write('Hypothesis:\t' + str(output) + ' \n')
                log_file.write('\tED: ' + str(ed) + ' | MED: ' + str(float(ed) / self.inputs['label_length'][i]) + '\n')
                log_file.write('\n')

        mean_ed = mean_ed / len(func_out)
        mean_norm_ed = mean_norm_ed / len(func_out)
        self.history['mean_ed'].append(mean_ed)
        self.history['mean_norm_ed'].append(mean_norm_ed)
        message = "---- MED: %0.3f, MED_norm: %0.3f ----\n" % (mean_ed, mean_norm_ed)
        print(message)
        if self.log_level == True:
            log_file.write(message)

    def on_train_end(self,logs=None):
        plt.clf()
        plt.plot(self.history['mean_ed'])
        plt.plot(self.history['mean_norm_ed'])
        plt.title('Model edit distance')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['mean_ed','mean_norm_ed'], loc='upper left')
        plt.savefig(self.name + '/acc_plot.png')

    def levenshtein(self,raw_a,raw_b):
        # Remove -1 from the lists
        a = [s for s in raw_a if s != -1]
        b = [s for s in raw_b if s != -1]

        "Calculates the Levenshtein distance between a and b."
        n, m = len(a), len(b)
        if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
            a,b = b,a
            n,m = m,n

        current = range(n+1)
        for i in range(1,m+1):
            previous, current = current, [i]+[0]*n
            for j in range(1,n+1):
                add, delete = previous[j]+1, current[j-1]+1
                change = previous[j-1]
                if a[j-1] != b[i-1]:
                    change = change + 1
                current[j] = min(add, delete, change)

        return current[n]
