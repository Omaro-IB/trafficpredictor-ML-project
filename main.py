from TrafficPredictor.Predictor import predictor as pred
import pyshark
from os import getcwd
from Packets import open_backup
DATA_DIR = getcwd() + "\\data\\"

# Directories to .traffic & .pydict files (consistent naming)
# downloading_dirs = ["downloading\\archlinux-p2p", "downloading\\debian-p2p", "downloading\\manjaro-p2p",
#                     "downloading\\kali-p2p", "downloading\\mint-p2p"]
downloading_dirs = ["downloading\\manjaro-p2p"]
social_dirs = ([f"socialmedia\\insta\\insta{i}" for i in range(1, 4+1)] +
               [f"socialmedia\\reddit\\reddit{i}" for i in range(1, 4+1)] +
               [f"socialmedia\\twitter\\twitter{i}" for i in range(1, 4+1)])
streaming_dirs = [f"streaming\\netflix{i}" for i in range(1, 12+1)]
zoom_dirs = [f"voip\\zoom{i}" for i in range(1, 6+1)]


# ==================================== PRE-PROCESSING ====================================
int_fields = ('ip_length', 'asn', 'transport_length', 'window', 'app_length', 'ttl', 'port')  # int fields to discretize
str_fields = tuple(["isp"])  # str fields to group together
custom_str_groups = {'country': [["Netherlands", "The Netherlands"]]}  # important grouping, manual inspection

# Create predictor using statistics information gathered using Wireshark & converted to .pydict using Packets.py
predictor = pred.Predictor(int_fields, str_fields, custom_str_groups)
predictor.add_stat_dict([DATA_DIR + d + ".pydict" for d in downloading_dirs], "downloading")
predictor.add_stat_dict([DATA_DIR + d + ".pydict" for d in social_dirs], "socialmedia")
predictor.add_stat_dict([DATA_DIR + d + ".pydict" for d in streaming_dirs], "streaming")
predictor.add_stat_dict([DATA_DIR + d + ".pydict" for d in zoom_dirs], "voip")

# Discretize and group statistics together for consistent binning + calculate all conditional probabilities
predictor.preprocess(laplace_k=250000, group_list_cachedir=DATA_DIR + "groupingscache")


# ==================================== BAYES ====================================
predictor.learn_bayes()  # create the bayes net


# ==================================== RNN ====================================
# Get total number of packets to be analyzed (= number of incoming + outgoing packets for all stat dicts)
num_packets = sum(i['direction']['incoming'] + i['direction']['outgoing'] for i in predictor.ALL_STAT_DICTS)
# Get number of features per packet (= 17)
num_features = len(predictor.ALL_STAT_DICTS[0])
# Maximum values for each string field for normalization (gathered empirically)
str_max_dict = {'direction': 10.0, 'flags': 10.0, 'isp': 1037.0, 'country': 120.0, 'city': 1226.0, 'threat_level': 10.0, 'transport_protocol': 10.0, 'urgent': 10.0, 'l7_protocol': 10.0, 'app_protocol': 10.0}
predictor.initialize_rnn(num_packets, num_features, str_max_dict)  # must initialize size of tensor first for efficiency

run_rnn_matrix_creation = False
if run_rnn_matrix_creation:
    # Create feature vectors from .traffic and back up to .pylist
    predictor.add_packet_matrices([DATA_DIR + d + ".traffic" for d in downloading_dirs], "downloading", DATA_DIR+"matrixbackup\\downloading.pylist")
    predictor.add_packet_matrices([DATA_DIR + d + ".traffic" for d in social_dirs], "socialmedia", DATA_DIR+"matrixbackup\\socialmedia.pylist")
    predictor.add_packet_matrices([DATA_DIR + d + ".traffic" for d in streaming_dirs], "streaming", DATA_DIR+"matrixbackup\\streaming.pylist")
    predictor.add_packet_matrices([DATA_DIR + d + ".traffic" for d in zoom_dirs], "voip", DATA_DIR+"matrixbackup\\voip.pylist")
else:
    # Load feature vectors from backup (.pylist files created using same function)
    predictor.add_packet_matrices(DATA_DIR+"matrixbackup\\downloading.pylist", "downloading")
    predictor.add_packet_matrices(DATA_DIR+"matrixbackup\\socialmedia.pylist", "socialmedia")
    predictor.add_packet_matrices(DATA_DIR+"matrixbackup\\streaming.pylist", "streaming")
    predictor.add_packet_matrices(DATA_DIR+"matrixbackup\\voip.pylist", "voip")
    predictor.import_legend(DATA_DIR+"matrixbackup\\voip.pylist.legend")  # import from last created packet matrix

predictor.learn_rnn(4, 256, 3, 10000, 0.0001, True)  # create the RNN


# ==================================== TESTS ====================================
run_b_predictions = False
if run_b_predictions:  # this takes some time as pyshark is not very optimized
    cap_downloading = pyshark.FileCapture(DATA_DIR + "test\\downloading.pcapng")
    cap_socialmedia = pyshark.FileCapture(DATA_DIR + "test\\socialmedia.pcapng")
    cap_streaming = pyshark.FileCapture(DATA_DIR + "test\\streaming.pcapng")
    cap_voip = pyshark.FileCapture(DATA_DIR + "test\\voip.pcapng")

    b_downloading_predictions = [predictor.predict_packet_bayes(i) for i in cap_downloading]
    noneCount = b_downloading_predictions.count(None)
    print(b_downloading_predictions.count("downloading") / (len(b_downloading_predictions) - noneCount))  # ~58%

    b_social_predictions = [predictor.predict_packet_bayes(i) for i in cap_socialmedia]
    noneCount = b_social_predictions.count(None)
    print(b_social_predictions.count("socialmedia") / (len(b_social_predictions) - noneCount))  # ~68%

    b_streaming_predictions = [predictor.predict_packet_bayes(i) for i in cap_streaming]
    noneCount = b_streaming_predictions.count(None)
    print(b_streaming_predictions.count("streaming") / (len(b_streaming_predictions) - noneCount))  # ~99%

    b_voip_predictions = [predictor.predict_packet_bayes(i) for i in cap_voip]
    noneCount = b_voip_predictions.count(None)
    print(b_voip_predictions.count("voip") / (len(b_voip_predictions) - noneCount))  # ~19%

run_r_predictions = True
if run_r_predictions:  # this takes some time as pyshark is not very optimized
    t_downloading = open_backup(DATA_DIR + "test\\downloading.traffic")
    t_socialmedia = open_backup(DATA_DIR + "test\\socialmedia.traffic")
    t_streaming = open_backup(DATA_DIR + "test\\streaming.traffic")
    t_voip = open_backup(DATA_DIR + "test\\voip.traffic")

    r_downloading_prediction, r_downloading_outs = predictor.predict_packet_rnn(t_downloading.packets)
    r_social_predictions, r_social_outs = predictor.predict_packet_rnn(t_socialmedia.packets)
    r_streaming_predictions, r_streaming_outs = predictor.predict_packet_rnn(t_streaming.packets)
    r_voip_predictions, r_voip_outs = predictor.predict_packet_rnn(t_voip.packets)

    print(r_downloading_prediction.count('downloading') / len(r_downloading_prediction))  # ~47%
    print(r_social_predictions.count('socialmedia') / len(r_social_predictions))  # ~63%
    print(r_streaming_predictions.count('streaming') / len(r_streaming_predictions))  # ~81%
    print(r_voip_predictions.count('voip') / len(r_voip_predictions))  # ~65%
