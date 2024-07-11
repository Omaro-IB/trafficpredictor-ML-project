import pickle
import pyshark
from ipvars import ip_vars


def packet2characteristics(packet):
    # IP Characteristics
    ip_length = packet['ip'].len
    flags = packet['ip'].flags
    ttl = packet['ip'].ttl
    try:
        ipinfo = ip_vars(packet['ip'].src)  # 'ASNum', 'ISP', 'Country', 'City', 'Crawler', 'Proxy', 'Attacker'
        direction = 'incoming'
    except ValueError:
        try:
            ipinfo = ip_vars(packet['ip'].dst)
            direction = 'outgoing'
        except ValueError:
            print(f"Packet number {packet.number} has both source and destination local IPs ({packet['ip'].src} -> {packet['ip'].dst})")
            return
    if ipinfo is None:
        raise Exception("Error with getting IP info, possible rate limits exceeded")

    # Transport layer characteristics
    try:
        l4 = packet['tcp']
        window = l4.window_size
        transport_protocol = 'tcp'
        transport_length = l4.len
        urgent = l4.urgent_pointer
    except KeyError:
        l4 = packet['udp']
        window = -1
        transport_protocol = 'udp'
        transport_length = l4.length
        urgent = -1

    if direction == 'outgoing':
        port = l4.dstport
    else:
        port = l4.srcport

    # Threat level
    threat_level = 0
    if ipinfo['Crawler'] == "Y":
        threat_level += 1
    if ipinfo['Proxy'] == "Y":
        threat_level += 2
    if ipinfo['Attacker'] == "Y":
        threat_level += 4

    # Application layer characteristics
    l7_protocol = None
    dirpacket = list(dir(packet))
    tcp_apps = tuple(["tls"])
    udp_apps = ("dns", "quic", "wg", "bt-dht", "uaudp")
    if dirpacket[-1] == "udp" and dirpacket[-2] == "transport_layer":  # UDP No L7
        assert transport_protocol == "udp"
        l7_protocol = "nol7"
    if "tcp" in dirpacket[-2:]:  # TCP No L7
        assert transport_protocol == "tcp"
        l7_protocol = "nol7"
    elif "udp" in dirpacket[-2:]:  # UDP with L7
        assert transport_protocol == "udp"
        for app in udp_apps:
            if app in dirpacket:
                l7_protocol = app
                break
    elif dirpacket[-1] == "transport_layer":  # TCP with L7
        assert transport_protocol == "tcp"
        for app in tcp_apps:
            if app in dirpacket:
                l7_protocol = app
                break
    if not l7_protocol:
        print(dirpacket)
        raise ValueError(f"{transport_protocol} Packet # {packet.number} has no known L7 protocol (known protocols: {tcp_apps+udp_apps})")

    if l7_protocol != "nol7":
        try:
            app_length = packet[l7_protocol].record_length
            app_protocol = packet[l7_protocol].app_data_proto
        except AttributeError:
            app_length = 0
            app_protocol = l7_protocol
    else:
        app_protocol = "noapp"
        app_length = 0

    # Return packet
    return ({
        "direction": direction,
        "ip_length": int(ip_length),
        "flags": str(flags),
        "ttl": int(ttl),
        "asn": int(ipinfo["ASNum"]),
        "isp": ipinfo["ISP"],
        "country": ipinfo["Country"],
        "city": ipinfo["City"],
        "threat_level": int(threat_level),
        "transport_protocol": transport_protocol,
        "transport_length": int(transport_length),
        "port": int(port),
        "window": int(window),
        "urgent": int(urgent),
        "l7_protocol": l7_protocol,
        "app_protocol": str(app_protocol),
        "app_length": int(app_length)
    })


class Traffic:
    def __init__(self, name):
        self.name = name
        self.packets = []
        self._required_fields = ("direction", "ip_length", "flags", "ttl", "asn", "isp", "country", "city", "threat_level",
                                 "transport_protocol", "transport_length", "port", "window", "urgent",
                                 "l7_protocol", "app_protocol", "app_length")
        self.statistics = dict(zip(self._required_fields, [None]*len(self._required_fields)))
        for s in self.statistics:
            self.statistics[s] = {}

    def create_backup(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    def _add_packet(self, packet):
        if set(packet.keys()) != set(self._required_fields):
            raise ValueError(f"Packet dictionary keys must be {self._required_fields}")
        self.packets.append(packet)
        # Statistics
        for field in packet:
            try:
                self.statistics[field][packet[field]] += 1
            except KeyError:
                self.statistics[field][packet[field]] = 1

    def import_pyshark_packet(self, packet):
        try:
            self._add_packet(packet2characteristics(packet))
        except Exception as e:
            print(f"Error with importing packet # {packet.number}: {e}")


def create_traffic_file(name, pcapng_file, traffic_file_dir):
    """
    Create a .traffic file from a pcpang (wireshark) file. This .traffic file contains the bytecode of a Packets.traffic
    object, and can be opened using the open_backup function. The statistics dictionary is also saved to a file.
    :param name: str: name property of Traffic object (can be anything)
    :param pcapng_file: str: path to pcapng file
    :param traffic_file_dir: str: path to traffic file to create
    :return:  the traffic object
    """
    cap = pyshark.FileCapture(pcapng_file)
    traffic = Traffic(name)
    for p in cap:
        traffic.import_pyshark_packet(p)
    traffic.create_backup(traffic_file_dir)  # .traffic backup
    with open(traffic_file_dir.split(".")[0] + ".pydict", "wb") as file:  # .pydict backup
        pickle.dump(traffic.statistics, file)
    return traffic


def open_backup(file):
    """
    Open file (.traffic or .pydict) and return object
    :param file: str: path to file
    :return: the traffic/dictionary object
    """
    with open(file, "rb") as file:
        return pickle.load(file)

