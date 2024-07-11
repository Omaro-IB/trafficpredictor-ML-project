from bs4 import BeautifulSoup
import requests
import json
import ipaddress


def private(ip):
    return ipaddress.ip_address(ip).is_private or ipaddress.ip_address(ip).is_multicast


def ip_vars(ip, cachedir=None):
    """
    Function to extract IP address variables
    Call this function instead of 'ip_info' to maintain a cache 'ipcache.json'
    :param ip: str: IP address formatted as "X.X.X.X"
    :param cachedir: str (optional, default = CWD): directory of cache file
    :return: dict: IP address variables - keys are 'ASNum', 'ISP', 'Country', 'City', 'Crawler', 'Proxy', 'Attacker'
    """
    if private(ip):  # private address
        raise ValueError("address must be a public IP")

    if cachedir is None:
        from os import getcwd
        filedir = getcwd() + "\\ipcache.json"
    else:
        filedir = cachedir + "\\ipcache.json"

    try:
        with open(filedir, "r") as file:
            cache = json.load(file)
    except FileNotFoundError:
        cache = {}

    if ip in cache:
        return cache[ip]
    else:
        try:
            info = _ip_info_dbip(ip)
        except Exception:
            try:
                info = _ip_info_ipinfo(ip)
            except Exception:
                print(f"Error with getting info for {ip}")
                return
        cache[ip] = info
        with open(filedir, "w+") as file:
            json.dump(cache, file)
        return info


def _ip_info_dbip(ip):
    """
    Function to extract IP address variables from db-ip.com
    :param ip: str: IP address formatted as "X.X.X.X"
    :return: dict: IP address variables
    """
    # Parse IP request
    r = requests.get(f"https://db-ip.com/{ip}", headers={"User-Agent": "Mozilla/5.0 (X11; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0"})
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all("div", {"class": "menu results shadow"})
    info = {}

    # Get IP ASN, ISP, and Location
    address_tags = results[0].find_all("tr")+results[-1].find_all("tr")
    for tag in address_tags:
        var, value = [i.text for i in tag.children]
        if 'ASN' in var:  # ASN
            info["ASNum"], info["ASName"] = value.split(" - ")
        if 'ISP' in var:  # ISP
            info["ISP"] = value.strip()
        if 'Country' in var:
            info["Country"] = value.strip()
        if 'City' in var:
            info["City"] = value.strip()

    # Get IP threat level: is it a crawler, proxy, or attack source?
    threat_tags = results[1].find_all("span")[:3]
    info["Crawler"] = threat_tags[0].text[0]
    info["Proxy"] = threat_tags[1].text[0]
    info["Attacker"] = threat_tags[2].text[0]

    return info


def _ip_info_ipinfo(ip):
    """
    Function to extract IP address variables from db-ip.com
    :param ip: str: IP address formatted as "X.X.X.X"
    :return: dict: IP address variables
    """
    # Parse IP request
    r = requests.get(f"https://ipinfo.io/{ip}", headers={"User-Agent": "Mozilla/5.0 (X11; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0"})
    soup = BeautifulSoup(r.text, 'html.parser')

    results = soup.find_all("div", {"class": "card card-details mt-0"})
    x = results[0].find_all("td")
    asnum = str(x[1]).split("AS")[1][:-2]
    isp = str(x[1]).split("-")[1].split("\n")[0][1:]

    results2 = soup.find_all("div", {"class": "pt-3 pb-1"})
    y = results2[0].find_all("td")
    city = str(y[1]).split("<")[1][3:]
    country = str(y[5]).split("</a>")[0].split(">")[-1]

    return {"ASNum": asnum,
            "ASName": '',
            "ISP": isp,
            "Country": country,
            "City": city, "Crawler": "N", "Proxy": "N", "Attacker": "N"}
