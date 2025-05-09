import os
import tensorflow as tf
import numpy as np
from critical_nca import CriticalNCA
# import powerlaw
# from sklearn.linear_model import LinearRegression
import utils
# import matplotlib.image
import matplotlib.pyplot as plt
# import time
# from PIL import Image
# from skimage import measure
# from sklearn import svm
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score
import csv

# import tensorflow as tf
# import numpy as np
import random
import time
from datetime import datetime
# import evodynamic.experiment as experiment
# import evodynamic.connection.cellular_automata as ca
# import evodynamic.connection as connection
# import evodynamic.cells.activation as act
from sklearn.svm import SVC
import helpers.helper as helper
from PIL import Image

plt.rcParams.update({'font.size': 14})

class ReservoirMemorySingleExperiment:
    def __init__(self, bits, r, itr, r_total_width, d_period, ca_rule, nca):

        self.recurrence = r
        self.iterations_between_input = itr + 1
        self.reservoir_height = itr
        self.reservoir_total_width = r_total_width
        self.distractor_period = d_period
        self.distractor_period_input_output = d_period + (2 * bits)
        self.number_of_bits = bits
        self.ca_rule = ca_rule
        self.reg = SVC(kernel="linear")
        self.input_channels = 4


        self.ca_height = self.distractor_period_input_output * self.iterations_between_input
        self.ca_width = self.reservoir_total_width

        self.input_true_locations = self.create_input_locations()
        # evo = self.set_up_evodynamics()
        # self.evo_exp = evo[0]
        # self.input_connections = evo[1]
        self.nca = nca

        self.current_state = np.zeros((1, self.ca_width, self.nca.channel_n), dtype=np.float32)

        self.x_training = []
        self.x_labels = []
        self.exp_history = []
        self.exp_memory_history = []
        self.correct_predictions = 0
        self.attempted_predictions = 0

    def create_input_locations(self):
        single_r_minwidth = self.reservoir_total_width // self.recurrence
        r_width_rest = self.reservoir_total_width % self.recurrence
        r_widths = np.full(self.recurrence, single_r_minwidth, dtype=int)
        for i in range(r_width_rest):
            r_widths[i] += 1
        input_true_locations = []
        for i in range(self.recurrence):
            input_locations = np.add(random.sample(range(r_widths[i]), self.input_channels),
                                     r_widths[:i].sum())
            input_true_locations.extend(input_locations)
        # return [8, 36, 25, 23, 50, 51, 57, 56, 113, 106, 104, 108]
        print("input_true_locations", input_true_locations)
        return input_true_locations

    def create_input_streams(self, input_array):
        input_streams = []

        input_stream = np.zeros(self.distractor_period_input_output, dtype=int)
        input_stream[:len(input_array)] = input_array
        input_streams.append(input_stream.tolist())

        input_stream = np.zeros(self.distractor_period_input_output, dtype=int)
        input_stream[:len(input_array)] = np.bitwise_xor(input_array, 1)
        input_streams.append(input_stream.tolist())

        input_stream = np.ones(self.distractor_period_input_output, dtype=int)
        input_stream[:len(input_array)] = np.zeros(self.number_of_bits)
        input_stream[self.distractor_period_input_output - len(input_array) - 1] = 0
        input_streams.append(input_stream.tolist())

        input_stream = np.zeros(self.distractor_period_input_output, dtype=int)
        input_stream[self.distractor_period_input_output - len(input_array) - 1] = 1
        input_streams.append(input_stream.tolist())

        return input_streams

    def create_output_streams(self, input_array):
        output_streams = []

        output_stream = np.zeros(self.distractor_period_input_output, dtype=int)
        output_stream[-len(input_array):] = input_array
        output_streams.append(output_stream.tolist())

        output_stream = np.zeros(self.distractor_period_input_output, dtype=int)
        output_stream[-len(input_array):] = np.bitwise_xor(input_array, 1)
        output_streams.append(output_stream.tolist())

        output_stream = np.ones(self.distractor_period_input_output, dtype=int)
        output_stream[-len(input_array):] = np.zeros(self.number_of_bits)
        output_streams.append(output_stream.tolist())

        return output_streams


    def run(self, evaluate=False):
        for bits in range(0, pow(2, self.number_of_bits)):
            # self.input_true_locations = self.create_input_locations()
            # evo = self.set_up_evodynamics_v2()
            # self.evo_exp = evo[0]
            # self.input_connections = evo[1]

            input_array = helper.int_to_binary_string(bits, self.number_of_bits)
            self.run_bit_string(input_array, evaluate)
            # self.evo_exp.close()

        if not evaluate:
            self.reg.fit(self.x_training, self.x_labels)
            this_score = self.reg.score(self.x_training, self.x_labels)
        else:
            this_score = self.correct_predictions / self.attempted_predictions

        return this_score

    def run_bit_string(self, input_array, evaluate):
        short_term_history = np.zeros((self.reservoir_height, self.ca_width*self.nca.channel_n), dtype=int).tolist()
        grid_ca = np.zeros((self.ca_height, self.ca_width, self.nca.channel_n))

        input_streams = self.create_input_streams(input_array)
        output_streams_labels = self.create_output_streams(input_array)
        self.current_state = np.zeros_like(self.current_state)

        for i in range(0, self.ca_height):
            self.run_step(i, input_streams, output_streams_labels, short_term_history, grid_ca, evaluate)

    def run_step(self, i, input_streams, output_streams_labels, short_term_history, grid_ca, evaluate):
        # g_ca_bin_current = self.evo_exp.get_group_cells_state("g_ca", "g_ca_bin")
        # step = np.zeros(self.ca_width)
        # if len(short_term_history) > 0:
        #     step = short_term_history[-1]
        # step = g_ca_bin_current[:, 0]
        step = self.current_state

        if i % self.iterations_between_input == 0:
            input_bits = helper.pop_all_lists(input_streams)
            for j in range(len(self.input_true_locations)):
                # print("len(self.input_true_locations), input_bits", len(self.input_true_locations), input_bits)
                input_bit = input_bits[j % 4]
                step[0, self.input_true_locations[j], 0] = float(int(step[0, self.input_true_locations[j], 0]) ^ input_bit)
                # step[0, self.input_true_locations[j], 0] = float(input_bit)

        short_term_history.append(step[0, :, :].flatten())
        short_term_history = short_term_history[-self.reservoir_height:]

        if i % self.iterations_between_input == 0:
            correct_answer = helper.pop_all_lists(output_streams_labels)
            reservoir_flattened_state = helper.flatten_list_of_lists(short_term_history)
            if correct_answer[0] == 1:
                correct_answer_class = 0
            elif correct_answer[1] == 1:
                correct_answer_class = 1
            else:
                correct_answer_class = 2

            if not evaluate:
                self.x_training.append(reservoir_flattened_state)
                self.x_labels.append(correct_answer_class)
            else:
                predicted_class = self.reg.predict([reservoir_flattened_state])
                self.attempted_predictions += 1
                if predicted_class[0] == correct_answer_class:
                    self.correct_predictions += 1

        if i < self.ca_height:
            grid_ca[i] = step[0]#.flatten()
        else:
            grid_ca = np.vstack((grid_ca[1:], step[0])) #step[0, :, 0].flatten()))
        # visuals
        self.exp_history.append(grid_ca.copy())
        self.exp_memory_history.append(short_term_history.copy())

        self.current_state = self.nca(step).numpy()

        # self.evo_exp.run_step(feed_dict={self.input_connections: step.reshape((-1, 1))})

        # _ = self.evo_exp.get_group_cells_state("g_ca", "g_ca_bin")

    def save_img(self, filename):
        import matplotlib.pyplot as plt
        plt.rcParams['image.cmap'] = 'binary'
        print("self.number_of_bits", self.number_of_bits)
        for i in range(0, 2 ** self.number_of_bits):
            arr = np.array(self.exp_history[((i + 1) * self.ca_height) - 1])#[:,:,0]

            # # fig, ax = plt.subplots(figsize=(self.ca_width//10, self.ca_height//10))
            # fig, ax = plt.subplots()

            # # arr2 = np.array(self.exp_history[((i + 1) * self.ca_height) - 2])
            # print("arr.shape", arr.shape)
            # # print("arr2.shape", arr2.shape)
            # print("len(self.exp_history)", len(self.exp_history))
            # # print(self.exp_history)
            # #ax.matshow(arr)
            # ax.imshow(arr)
            # # ax.matshow(self.exp_history)

            # ax.axis(False)
            # # fig.show()
            # img_fname = filename.replace(".txt", "_"+str(i) +".png")
            # #fig.savefig("test"+str(i) +".png")
            # fig.savefig(img_fname, bbox_inches='tight')
            # plt.close(fig)

            img_arr = 255 - arr*255
            img_arr = img_arr.reshape((img_arr.shape[0], -1), order="F")
            print("img_arr.shape", img_arr.shape)

            # img = Image.fromarray(img_arr.resize(5*img_arr.shape[1],5*img_arr.shape[0], Image.NEAREST)
            img = Image.fromarray(img_arr).resize((5*img_arr.shape[1],5*img_arr.shape[0]), Image.Resampling.NEAREST)
            # print(arr.shape)
            # img = Image.fromarray()
            img = img.convert("L")
            img_fname = filename.replace(".txt", "_"+str(i) +".png")
            img.save(img_fname)
            # fig.savefig("test_img_"+str(i) +".png")

    def show_visuals(self):

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.animation as animation

        def updatefig(*args):
            im.set_array(self.exp_history[self.idx_anim, :, 0])
            im2.set_array(self.exp_memory_history[self.idx_anim, :, 0])
            if self.idx_anim % self.iterations_between_input == 0:
                pred = self.reg.predict([self.x_training[self.idx_anim // self.iterations_between_input]])
                # print(pred)
                # print(x_labels[200])
                if pred == 0:
                    im3.set_array([[1, 0, 0]])
                elif pred == 1:
                    im3.set_array([[0, 1, 0]])
                else:
                    im3.set_array([[0, 0, 1]])
                # im3.set_array([list(map(round, pred[0]))])
                ax3.set_title("model prediction: " + str(pred))
                cor = self.x_labels[self.idx_anim // self.iterations_between_input]
                if cor == 0:
                    im4.set_array([[1, 0, 0]])
                elif cor == 1:
                    im4.set_array([[0, 1, 0]])
                else:
                    im4.set_array([[0, 0, 1]])
                # im4.set_array([x_labels[idx_anim // (iterations_between_input + 1)]])
            fig.suptitle(
                'Step: ' + str(self.idx_anim % self.ca_height) + " Exp: " + str(self.idx_anim // self.ca_height))
            self.idx_anim += 1

        fig = plt.figure()
        gs = fig.add_gridspec(4, 8)
        ax1 = fig.add_subplot(gs[:-1, :-1])
        ax1.set_title("reservoir full history")
        ax2 = fig.add_subplot(gs[3, :-1])
        ax2.set_title("model perceived history")
        ax3 = fig.add_subplot(gs[:-2, 7])
        ax3.set_title("model prediction")
        ax3.axis("off")
        ax4 = fig.add_subplot(gs[2:, 7])
        ax4.set_title("model desired output")
        ax4.axis("off")

        im_ca = np.zeros((self.ca_height, self.ca_width))

        shortTermHistory = np.zeros((self.reservoir_height, self.ca_width), dtype=int).tolist()

        im = ax1.imshow(im_ca, animated=True, vmax=1)
        im2 = ax2.imshow(shortTermHistory, animated=True, vmax=1)
        im3 = ax3.imshow(np.zeros((1, 3), dtype=int).tolist(), animated=True, vmax=1)
        im4 = ax4.imshow(np.zeros((1, 3), dtype=int).tolist(), animated=True, vmax=1)

        fig.suptitle('Step: 0 Exp: 0')

        print(self.input_true_locations)

        # implement as list of arrays instead?

        self.idx_anim = 0
        ani = animation.FuncAnimation(
            fig,
            updatefig,
            frames=(self.ca_height - 1) * 32,
            interval=200,
            blit=False,
            repeat=False
        )

        plt.show()

        # plt.connect('close_event', self.exp.close())


def recordingExp(bits, r_total_width, d_period, ca_rule, runs, r, itr, nca):
    filename = f'nca {datetime.now().isoformat().replace(":", " ")}.txt'
    file = open(filename, "a")
    file.writelines(
        f'bits={bits}, r={r}, itr={itr}, r total width={r_total_width}, distractor period={d_period}, CA rule={ca_rule}, number of runs={runs}, started at: {datetime.now().isoformat()}')
    file.writelines("\nscore")
    file.close()

    start_time = time.time()
    scores = []
    for expRun in range(0, runs):
        print("starting exp nr" + str(expRun))
        start_time_sub = time.time()
        exp = ReservoirMemorySingleExperiment(bits=bits, r=r, itr=itr, r_total_width=r_total_width, d_period=d_period,
                                              ca_rule=ca_rule, nca=nca)
        score = exp.run()
        # scoreEval = exp.run(True)
        # exp.show_visuals()
        exp.save_img(filename)
        file = open(filename, "a")
        file.write("\n" + str(score) + "\t" + str(exp.input_true_locations))
        # file.write("\n" + str(score) + "\t" + str(exp.input_true_locations) + "\t" + str(scoreEval))
        file.close()
        print(score)
        scores.append(score)
        this_runtime = time.time() - start_time_sub
        print(this_runtime)

    print(time.time() - start_time)
    number_of_successes = (sum(map(lambda i: i == 1.0, scores)))
    # present as %
    print(number_of_successes)
    file = open(filename, "a")
    file.writelines("\nSuccesses: ")
    file.writelines(str(number_of_successes))
    file.close()


def get_nca(args, ckpt=""):
    print("Testing checkpoint saved in: " + args.logdir)

    del args.nca_model["built"]
    del args.nca_model["inputs"]
    del args.nca_model["outputs"]
    del args.nca_model["input_names"]
    del args.nca_model["output_names"]
    del args.nca_model["stop_training"]
    del args.nca_model["history"]
    del args.nca_model["compiled_loss"]
    del args.nca_model["compiled_metrics"]
    del args.nca_model["optimizer"]
    del args.nca_model["train_function"]
    del args.nca_model["test_function"]
    del args.nca_model["predict_function"]
    del args.nca_model["channel_n"]
    # del args.nca_model["padding_size"]

    nca = CriticalNCA(**args.nca_model)
    nca.dmodel.summary()

    ckpt_filename = ""
    if ckpt == "":
        checkpoint_filename = "checkpoint"
        with open(os.path.join(args.logdir, checkpoint_filename), "r") as f:
            first_line = f.readline()
            start_idx = first_line.find(": ")
            ckpt_filename = first_line[start_idx+3:-2]
    else:
        ckpt_filename = os.path.basename(ckpt)

    print("Testing model with lowest training loss...")
    nca.load_weights(os.path.join(args.logdir, ckpt_filename))

    return nca

if __name__ == "__main__":
    # python reservoir_X-bit_make_dataset.py --logdir ..\..\CriticalNCA\logs\train_nca\20230501-133904
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", help="path to log directory")
    p_args = parser.parse_args()

    if p_args.logdir:
        args_filename = os.path.join(p_args.logdir, "args.json")
        argsio = utils.ArgsIO(args_filename)
        args = argsio.load_json()
        args.logdir = p_args.logdir

        nca = get_nca(args)
        # train_readout(args)
        bits = 5
        r_total_width = 80 #64 #160
        d_period = 200 #200
        runs = 1
        ca_rule = 0

        r = 4
        itr = 2
        recordingExp(bits, r_total_width, d_period, ca_rule, runs, r, itr, nca)

    else:
        print("Add --logdir [path/to/log]")
