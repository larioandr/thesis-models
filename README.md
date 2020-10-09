# thesis-models
Models and calculations for my PhD thesis

## Experiments



### Example of building a tcpdump:

```
> sudo tcpdump -i en0 -s 0 -nn -l -q -v -ttt > data/dump1.txt
```

### Example of sed-ing only timestamps and packet sizes from tcpdump:
```
> cat data/dump1.txt | grep '0\+:' | sed 's/00:00:00.//' | sed 's/\ IP\ .*\(length\)//' | sed 's/)$//'
```

### Python script for parsing pcap:

```python
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether

def process_pcap(file_name):
    count = 0
    prev_time = 0
    data = []
    for pkt_data, pkt_metadata in RawPcapReader(file_name):
        count += 1
        ether_pkt = Ether(pkt_data)
        dt = ether_pkt.time - prev_time
        prev_time = ether_pkt.time
        data.append((ether_pkt.time, dt, pkt_metadata.wirelen))

    for record in data:
        print(record)
    print(f'Printed {count} packets')

process_pcap('data/dump1.pcapng')
```


### Live streaming with VLC:

See details [here](https://wiki.videolan.org/Documentation:Streaming_HowTo/Command_Line_Examples/).

For RTSP (UDP streaming):

- Run the server (`input_stream` is just an .mp4 file):

```bash
% vlc -vvv input_stream --sout '#rtp{dst=192.168.0.12,port=1234,sdp=rtsp://server.example.org:8080/test.sdp}' 
```

- Run on the client:

```bash
% vlc rtsp://server.example.org:8080/test.sdp
```
