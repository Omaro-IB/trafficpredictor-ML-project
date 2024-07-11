from TrafficPredictor.Probability import distributions
from TrafficPredictor.Probability.bayesnet import BayesNet
from TrafficPredictor.RNN.rnn import RNN
from TrafficPredictor.Predictor import dataprocessing
import Packets as Packets
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")
import torch
import torch.nn as nn


class Predictor:
    def __init__(self, int_fields, str_fields, custom_str_groups=None):
        """
        A predictor for traffic types based on pyshark and Bayes Networks
        :param int_fields: int fields to discretize
        :param str_fields: str fields to group together
        :param custom_str_groups: any groupings through manual inspection
        """
        # Pre-processing attributes
        if custom_str_groups is None:
            self.custom_str_groups = {}
        else:
            self.custom_str_groups = custom_str_groups
        self.int_fields = int_fields
        self.str_fields = str_fields
        self.minmax_dict = None  # Minimum and maximum values for each int_field

        # Bayes attributes
        self.ALL_STAT_DICTS = []  # List of statistics dictionaries
        self.ALL_STAT_DICTS_DISCRETIZED_GROUPED = None  # statistics dictionaries with consistent bins
        self.CONDITIONAL_PROBS = {}  # Conditional probabilities of each statistic field for each label
        self.B_LABELS = {}  # Labels of different types of traffic types for bayes net
        self.traffic_table = None  # the traffic table (% of packets for each label)
        self.nbins = None
        self.conditional_tables = {}  # the conditional tables
        self.bnet = BayesNet()

        # RNN attributes
        self.ALL_PACKET_MATRICES = None  # All matrices representing packets from .traffic file
        self.packets_added = 0
        self.R_LABELS = {}  # Labels of different types of traffic types for RNN
        self.STR_LEGEND = {}  # all possible values for each str_field
        self.rnn_model = None
        self.str_max_dict = None  # Maximum values for each str_field

    # PREPROCESSING METHODS
    def add_stat_dict(self, directories, label, mode="backup"):
        """
        Add a statistic dictionary to the predictor model - necessary for both learn_bayes and learn_rnn
        :param directories: list: list of directories of .pydict files if using backup, or directories of .pcapng if
                                  reading new Wireshark. If reading new, a .pydict backup file will be created to save
                                  time in future
        :param label: str: label of the imported directories
        :param mode: str (default="backup"): either 'backup' or 'new'
        :return: None
        """
        if label in self.B_LABELS:
            raise ValueError(f"{label} statistics already added")

        # Preprocessing - convert .pcapng to .traffic file and also export statistics dictionary as .pydict
        if mode == "new":
            pydict_dirs = []
            for directory in directories:
                x = directory.split("\\")
                tname = x[-1].split(".")[0]
                x[-1] = tname
                tdir = "\\".join(x)
                Packets.create_traffic_file(tname, tdir + ".pcapng", tdir + ".traffic")
                pydict_dirs.append(tdir + ".pydict")
            self.add_stat_dict(pydict_dirs, label)

        # Load from backup and update LABELS
        if mode == "backup":
            first = len(self.ALL_STAT_DICTS)
            last = first + len(directories) - 1
            self.B_LABELS[label] = (first, last)
            for directory in directories:
                self.ALL_STAT_DICTS.append(Packets.open_backup(directory))

    def preprocess(self, nbins=None, medoids_k=None, laplace_k=None, group_list_cachedir=None):
        """
        Preprocess given statistic dictionaries added with add_stat_dict - necessary for both learn_bayes and learn_rnn
        :param nbins: int optional (default = 5): number of bins to discretize integer fields into
        :param medoids_k: int: (optional, default = 87% * number of strings in str_list): number of clusters to use when
                               grouping string fields using k medoids
        :param laplace_k: int (optional, default = 0): laplace smoothing hyperparameter k
        :param group_list_cachedir: str (optional, default = None): directory to save/load cached grouping, if None then don't cache
        :return:
        """
        if self.minmax_dict is not None:
            raise RuntimeError("the preprocess method has already been called")
        if not self.ALL_STAT_DICTS:
            raise ValueError("at least one stat_dict must be added using add_stat_dict")
        if nbins is None:
            nbins = 5
        self.nbins = nbins
        if medoids_k is None:
            medoids_k = -1
        if laplace_k is None:
            laplace_k = 0

        # Discretizing & Grouping statistics dictionaries
        self.ALL_STAT_DICTS_DISCRETIZED_GROUPED, self.minmax_dict = dataprocessing.discretize_group_together(self.ALL_STAT_DICTS, self.int_fields, self.str_fields, self.custom_str_groups, nbins, medoids_k, group_list_cachedir)

        # Conditional probabilities
        for label in self.B_LABELS:
            self.CONDITIONAL_PROBS[label] = dataprocessing.dict_list_to_probabilities_horizontal(self.ALL_STAT_DICTS_DISCRETIZED_GROUPED[self.B_LABELS[label][0]:self.B_LABELS[label][1] + 1], laplace_k)

    # BAYES METHODS
    def learn_bayes(self):
        """
        Learn Bayesian Network conditional probability tables
        :return: None
        """
        if self.ALL_STAT_DICTS_DISCRETIZED_GROUPED is None:
            raise RuntimeError("the preprocess method must first be called")
        if self.traffic_table is not None:
            raise RuntimeError("the learn_bayes method has already been called")

        # Creating the Bayes Net
        # P(Traffic)
        self.traffic_table = distributions.Distribution("Traffic", tuple(self.B_LABELS.keys()))
        for traffic_type in self.B_LABELS:
            self.traffic_table.set_p(traffic_type, 1 / len(self.B_LABELS))  # assuming equal probability for each traffic

        # P(Feature | Traffic)
        for field in self.ALL_STAT_DICTS_DISCRETIZED_GROUPED[0]:
            vals = set()
            for label in self.B_LABELS:
                vals = vals | set(self.CONDITIONAL_PROBS[label][field].keys())
            vals = tuple(vals)
            self.conditional_tables[field] = distributions.ConditionalDistribution(tuple(["Traffic"]), tuple([tuple(self.B_LABELS.keys())]), field, vals)
            for label in self.B_LABELS:
                for key in vals:
                    try:
                        self.conditional_tables[field].set_p(tuple([label]), key, self.CONDITIONAL_PROBS[label][field][key])
                    except KeyError:
                        self.conditional_tables[field].set_p(tuple([label]), key, 0)

        # Network
        self.bnet.add_node(self.traffic_table)
        for conditional in self.conditional_tables:
            self.bnet.add_node(self.conditional_tables[conditional])

    def predict_packet_bayes(self, cap):
        """
        Predict the traffic type using the learned Bayesian network
        :param cap: pyshark.packet.packet.Packet: the pyshark packet to predict
        :return: str: one of the traffic types or "uncertain"
        """
        if self.traffic_table is None:
            raise RuntimeError("learn_bayes method must first be called")

        from math import e
        from Packets import packet2characteristics
        try:
            q = packet2characteristics(cap)
            if q is None:
                return
        except KeyError:
            return
        for x in self.minmax_dict:  # Convert int fields to proper bin
            q[x] = dataprocessing.find_bin(self.minmax_dict[x], q[x], self.nbins)
        ps = []
        traffic_types = tuple(self.B_LABELS.keys())
        for T in traffic_types:
            q["Traffic"] = T
            try:
                ps.append(e**self.bnet.probability(q))
            except KeyError:
                return "uncertain"
        if all(j == 0 for j in ps):
            return "uncertain"
        return traffic_types[max(range(len(ps)), key=ps.__getitem__)]

    # RNN METHODS
    def get_classes(self):
        return tuple(self.R_LABELS.keys())

    def initialize_rnn(self, num_packets, num_features, str_max_dict):
        """
        Initializes the ALL_PACKET_MATRICES tensor to correct size
        :param num_packets:  number of packets to be added into the tensor (X dimension size)
        :param num_features: number of features in a given packet matrix (YxZ dimension size)
        :param str_max_dict: dict: str keys, int values of estimated maximum value for each packet field
        :return:
        """
        self.ALL_PACKET_MATRICES = np.ndarray((num_packets, num_features, num_features))
        self.str_max_dict = str_max_dict

    def _packet_to_vector(self, packet):
        # Convert packet to vector representation
        for x in packet:
            if x not in self.minmax_dict and x not in self.str_max_dict:
                raise ValueError(f"{x} not found in estimated_str_max or minmax_dict dictionaries")

            if x in self.minmax_dict:  # Convert int fields to proper bin
                # bindex = dataprocessing.find_bin(self.minmax_dict[x], packet[x], self.nbins, index=True)
                # if bindex is None:
                #     packet[x] = 0
                # else:
                #     packet[x] = (bindex + 1) / self.nbins  # set to normalized bindex
                packet[x] = packet[x] / self.minmax_dict[x][1]
            else:  # convert str fields to their index
                try:
                    packet[x] = self.STR_LEGEND[x].index(packet[x]) + 1  # field and key exists in legend
                except ValueError:  # field exists but not key
                    self.STR_LEGEND[x].append(packet[x])
                    packet[x] = len(self.STR_LEGEND)
                except KeyError:  # field does not exist
                    self.STR_LEGEND[x] = [packet[x]]
                    packet[x] = 1
                packet[x] = packet[x] / self.str_max_dict[x]  # Normalize
        return tuple(packet.values())

    def _add_packets(self, feature_vectors, label):
        # Add packet matrices from 2D list of features (and update R_LABELS)
        if label in self.R_LABELS:
            raise ValueError(f"{label} traffic already added")
        if self.ALL_PACKET_MATRICES is None or self.str_max_dict is None:
            raise RuntimeError("initialize_rnn method must be called first")

        # Update global current packet index
        first = self.packets_added
        self.packets_added += len(feature_vectors)

        # Add diagonal matrix of feature vector to global list
        for i in range(len(feature_vectors)):
            self.ALL_PACKET_MATRICES[first+i] = np.diag(feature_vectors[i])
        self.R_LABELS[label] = (first, self.packets_added)

    def add_packet_matrices(self, path, label, backup=None):
        """
        Add traffic packets to the predictor model - necessary for learn_rnn
        :param path: str or list: path to .pylist file or .traffic files
        :param label: str: label of the imported directories
        :param backup: str (optional): folder directory for backup if importing from .traffic
        :return: None
        """
        if type(path) is str and backup is not None:
            raise ValueError("if opening a single pylist, backup should not be provided")
        if type(path) is list and backup is None:
            raise ValueError("if opening multiple traffic files, backup should be provided")
        if self.ALL_PACKET_MATRICES is None:
            raise RuntimeError("initialize_rnn method must be called first")

        # Open multiple .traffic and back up
        if type(path) is list:
            to_add = []  # 2D list of packet features
            for directory in path:
                print(directory)
                t = Packets.open_backup(directory)
                counter = 0
                for packet in range(len(t.packets)):
                    if round(len(t.packets) / 100) * counter == packet:
                        print(f"{counter}%")  # display progress for each .traffic file
                        counter += 1
                    to_add.append(self._packet_to_vector(t.packets[packet]))
            self._add_packets(to_add, label)
            with open(backup, "wb") as file:
                pickle.dump(to_add, file)
            with open(backup+".legend", "wb") as file:
                pickle.dump(self.STR_LEGEND, file)

        # Load from .pylist backup and update LABELS
        elif type(path) is str:
            to_add = Packets.open_backup(path)
            self._add_packets(to_add, label)

        else:
            raise TypeError("path must be a list or string")

    def import_legend(self, legend_file):
        self.STR_LEGEND = Packets.open_backup(legend_file)

    def learn_rnn(self, num_layers, hidden_size, num_epochs, batch_size, learning_rate, verbose=False):
        """
        Learn the RNN model
        :param num_layers: int: number of layers
        :param hidden_size: int: size of the hidden layers
        :param num_epochs: int: number of epochs
        :param batch_size: int: batch size
        :param learning_rate: float: learning rate
        :param verbose: bool (optional, default = False): print learning progress and corresponding loss values
        :return: None
        """
        if self.ALL_PACKET_MATRICES is None:
            raise ValueError("at least one .traffic file must be added using add_traffic_file method")

        r_labels = tuple(self.R_LABELS.keys())

        def labels_from_range(s, e):
            # From a range from start to end, return a list of numeric labels corresponding to keys in R_LABELS
            curr = 0
            lis = []
            while s < e:
                if s in range(self.R_LABELS[r_labels[curr]][0], self.R_LABELS[r_labels[curr]][1]):
                    to_add = min((self.R_LABELS[r_labels[curr]][1] - s, e - s))
                    lis.extend([curr] * to_add)
                    s += to_add
                else:
                    curr += 1
            return lis

        input_size = len(self.ALL_PACKET_MATRICES[0])  # num features (17)
        num_classes = len(self.R_LABELS)  # num classes (4)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rnn_model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
        # self.rnn_model = RNN(input_size, hidden_size, num_layers, num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.rnn_model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Train model
        n_total = len(self.ALL_PACKET_MATRICES)
        order = 10**(len(str(n_total))-2)
        for epoch in range(num_epochs):
            for start in range(0, n_total, batch_size):
                end = start + batch_size
                if end > n_total:
                    end = n_total
                packets = torch.tensor(self.ALL_PACKET_MATRICES[start:end], dtype=torch.float32).to(device)
                labels = torch.tensor(labels_from_range(start, end)).to(device)
                # packets = torch.tensor(self.ALL_PACKET_MATRICES[start:end])
                # labels = torch.tensor(labels_from_range(start, end))

                # Forward pass
                outputs = self.rnn_model(packets)
                del packets
                loss = criterion(outputs, labels)
                del labels

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and start % order == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{start}/{n_total}], Loss: {loss.item():.4f}')

    def predict_packet_rnn(self, caps):
        """
        Predict the next packet and traffic type given a sequence of packets
        :param caps: list(pyshark.packet.packet.Packet): list of the pyshark packets to predict
        :return: list: feature vector of next packet, str: one of the traffic types or "uncertain"
        """
        if self.rnn_model is None:
            raise RuntimeError("learn_rnn method must first be called")
        if not self.STR_LEGEND:
            raise RuntimeError("STR_LEGEND must be imported either by importing .traffic files with add_packet_matrices"
                               " or import_legend method if importing .pylist from backup")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vector_sequence = []
        for cap in caps:
            vector_sequence.append(self._packet_to_vector(cap))
        matrix_sequence = torch.tensor(np.array([np.diag(v) for v in vector_sequence]), dtype=torch.float32).to(device)
        # matrix_sequence = torch.tensor([np.diag(v) for v in vector_sequence])

        with torch.no_grad():
            outputs = self.rnn_model(matrix_sequence)
            _, predicted = torch.max(outputs.data, 1)

        key = tuple(self.R_LABELS.keys())
        return [key[i.item()] for i in predicted], outputs
