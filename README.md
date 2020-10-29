# thesis-models

Модели и численные эксперименты из моей диссертации.


## Утилиты

### RFID-симулятор `rfidsim`

Утилита `rfidsim` выполняет имитационную модель RFID, в которой моделируются проезды
автомобилей с RFID-метками и их чтение с помощью считывателя. В программе есть две
основные функции:

1. `rfidsim simulate` - выполнение имитационной модели, в том числе - для групп входных данных.
2. `rfidsim analyze` - анализ результатов имитационной модели.

При выполнении первой подпрограммы результаты выполнения записываются в набор выходных файлов,
которые впоследствие можно анализировать с помощью второй подпрограммы. Промежуточные файлы
включают логи считывателя и меток, общие данные о каждой промоделированной конфигурации
и выборки интервалов между пакетами, которые считыватель передает в ЦОД.

#### Имитационная модель RFID

Формат вызова для выполнения симуляции:

```sh
> rfidsim simulate <имя конфигурации> \
    [-c|--config <JSON-файл конфигурации>] \
    [-o|--output <директория>] \
```

Конфигурация задается в JSON-файле, в котором должны присутствовать объект
`config.rfidsim` с основной конфигурацией модели, а также объект в массиве 
`custom`. Структура файла следующая:

```js
{
    "config": {
        "rfidsim": {
            //...
        }
    },
    "custom": [{
        "name": "имя конфигурации"^
        "config": {
            "rfidsim": {
                ...
            }
        }
    }]
}
```

Для каждого параметра конфигурация может содержать либо фиксированное значение
(его тип определяется конкретным параметром), либо группу значений. Группы можно
задавать в виде массива или диапазона (для чисел):

```js
// Задание массива параметров:
{
    "type": "array", 
    "data": [value1, value2, ..., valueN]
}
// Диапазон:
{
    "type": "range",
    "args": {"min": valueMin, "max": valueMax, "step": 10}
}
```

Если в конфигурации присутствуют групповые значения, будет построено множество
всевозможных конфигураций со значениями из этих групп, и на каждой из них будет
выполнена симуляция. Результаты каждой симуляции записываются в отдельные журналы,
а соответствие конфигураций номерам записывается в специальный json-файл.

Пример конфигурации, в котором скорость задана через диапазон (30, 40, ..., 120 км/ч),
а всевозможные значения `Tari` и `m` - через массивы, что в общей сложности дает 
`10 * 4 * 4 = 160` конфигураций симуляций:

```json
{
  "config": {
    "rfidsim": {
      "model": {
        "numLanes": 2,
        "laneWidth": 3.5,
        "speed": 40,
        "vehicleDirection": [1, 0, 0],
        "vehicleLength": 4,
        "vehiclePlates": ["front", "back"],
        "plateHeight": 0.5,
        "posUpdateInterval": 1e-2,
        "vehicleStartOffset": [-10.0, 0, 0],
        "vehicleLifetime": 2.0,
        "vehicleInterval": {
          "dist": "uniform",
          "args": { "min": 0.9, "max": 1.1 }
        },
        "tagModulationLoss": -10.0,
        "tagSensitivity": -18.0,
        "tagAntennaAngle": 0,
        "tagAntennaRadiationPattern": "dipole",
        "tagAntennaGain": 2.0,
        "tagAntennaPolarization": 1.0,
        "readerAntennaSides": ["front", "back"],
        "readerAntennaAngle": 45.0,
        "readerAntennaOffset": 1.0,
        "readerAntennaRadiationPattern": "dipole",
        "readerAntennaGain": 8.0,
        "readerAntennaPolarization": 0.5,
        "readerCableLoss": -1.0,
        "readerTxPower": 31.5,
        "readerCirculatorNoise": -80.0,
        "roundsPerAntenna": 1,
        "roundsPerInventoryFlag": 1,
        "sessionStrategy": "A",
        "tari": "12.5us",
        "m": 4,
        "data0Mul": 2.0,
        "rtcalMul": 2.0,
        "sl": "ALL",
        "session": "S0",
        "dr": "8",
        "trext": false,
        "q": 4,
        "frequency": 860e6,
        "powerOnInterval": 2000e-3,
        "powerOffInterval": 100e-3,
        "doppler": true,
        "thermalNoise": -114.0,
        "permittivity": 15.0,
        "conductivity": 3e-2,
        "berModel": "rayleigh"
      },
      "simulation": {
        "maxTime": 1000,
        "maxVehicles": 1000
      },
      "variate": {
        "join": [
          {
            "val": {
              "speed": {"range": {"min": 30, "max": 80, "step": 10}},
              "tari": {"array": ["6.25us", "12.5us", "18.75us"]},
              "m": {"array": [1, 2, 4, 8]},
            }
          }, 
          {
            "product": [
              {          
                "val": {
                  "speed": {"array": {"min": 90, "max": 120, "step": 5}}
                }
              }, 
              {
                "zip": [
                  {
                    "val": {
                      "tari": {"array": ["12.5us", "18.75us", "25.0us"]} 
                    },
                  }, {
                    "val": {
                      "m": { "array": [8, 4, 2]}
                    }
                  }
                ]
              }
            ]
          }
        ]
      }
    }
  },
  "custom": [
    {
      "name": "coarse",
      "config": {
        "rfidsim": {
          "simulation": {
            "maxTime": 50,
            "maxVehicles": 40
          }
        }
      }
    }
  ]
}
```

Выходные файлы из `rfidsim simulate` записываются в папку `data/results/rfidsim/`:

- `rfidsim_<имя конфигурации>_info.json` - информация о соответствиях между номером симуляции и параметрами и общая информация о симуляции;
- `rfidsim_<имя конфигурации>_<n>_vehicles.csv` - данные о сгенерированных автомобилях (время появления, номера, метки);
- `rfidsim_<имя конфигурации>_<n>_rounds.txt` - журнал считывателя из n-й симуляции;
- `rfidsim_<имя конфигурации>_<n>_tags.txt` - журнал меток из n-й симуляции;
- `rfidsim_<имя конфигурации>_<n>_packets.txt` - журнал отправленных считывателем пакетов (если задан флаг `--log-packets`).

В файле `rfidsim_<имя конфигурации>_sim.json` сохраняются конкретные значения параметров,
которые использовались при выполнении n-й симуляции, а также записывается общая информация 
(например, сколько меток и машин было сгенерировано):

```js
[{
    "simulation": 1,  // номер симуляции
    "summary": {
        "numVehiclesGenerated": 100,  // сколько машин было сгенерировано
        "numVehiclesDeparted": 96,  // сколько машин успело покинуть область
        "numTagsGenerated": 200,  // сколько меток было сгенерировано
        "numTagsDeparted": 192,  // сколько меток успело покинуть область
        "numRounds": 23156,  // сколько раундов было проиграно
        "simTime": 10000,  // время на модельных часах
        "realTime": 56.8  // сколько реального времени заняла симуляция
    },
    "config": {
        // Конфигурация, в которой все параметры - скаляры.
        // Например, указано точное значение скорости:
        "vehicle": {
            "speed": 60,
            // ...
        }
    }
}]
```

CSV-файл с данными автомобилей имеет следующие колонки:

- `ID`: идентификатор автомобиля
- `Timestamp` - время генерации
- `FrontPlate` - номер переднего номерного знака
- `BackPlate` - номер заднего номерного знака
- `FrontEPC` - EPCID метки в переднем номере
- `FrontTID` - TID метки в переднем номере 
- `BackEPC` - EPCID метки в заднем номере
- `BackTID` - TID метки в заднем номере 

Файл с журналом считывателя `rfidsim_<имя конфигурации>_<n>_rounds.txt` содержит 
данные о каждом раунде и слоте. Про раунды записывается: когда раунд начался, 
по какой антенне происходил опрос, какая сессия использовалась.
Про слоты: номер, время начала слота и ответы, полученные от меток. Про каждый ответ:
тип сообщения, принятая мощность в начале ответа, результат приема ответа и внутренний идентификатор
метки.

Формат строки раунда:
```
R <номер> <время начала> A:<номер антенны> S:<номер сессии> T:<флаг опроса>
```

Формат строки слота (для удобства, задается с отступом в два пробела):
```
S <номер слота> <время начала>
```

Формат строки сообщения (с отступом в 4 пробела):
```
- <тип ответа> <мощность в начале ответа> <результат> tag:<внутренний идентификатор метки>
```

Также в начале файла указывается ориентация антенн. Пример:
```
# Antenna 1: (0.7071, -0.7071, 0)
# Antenna 2: (-0.7071, -0.7071, 0)

...
R 123 10.580 A:1 S:0 T:A
  S 1 10.580
  S 2 10.630
    - RN16 -71.0 ER tag:10
    - RN16 -69.0 ER tag:11
  S 3 10.650
    - RN16 -65.0 OK tag:12
    - EPC -62.0 OK tag:12
    - Handle -66.0 OK tag:12
    - Data -67.0 OK tag:12
  S 4 10.750
    - RN16 -75.0 OK tag:14
    - EPC -79.0 ER tag:14
OFF 10.790
R 124 10.890 A:2 S:0 T:B
  S 1 10.890
...
```

Файл журнала меток `rfidsim_<имя конфигурации>_<n>_tags.txt` содержит информацию, 
когда каждая метка появилась, включалась и выключалась, как менялась мощность поля 
и как она реагировала на принятые команды.

Каждая строчка имеет следующий формат (в качестве разделителя используются табуляции):

```
<время> <ID метки>  <тип события>   <мощность>  <координаты>    <сообщение|состояние> [<параметр>=<значение>*]
```

Типы событий кодируются одной буквой:

- `A`: вход метки в область чтения
- `D`: выход метки из области чтения
- `U`: обновление положения и/или мощности поля на метке
- `R`: завершение приёма команды и выполнение действий в ответ на команду
- `T`: начало отправки ответа

При описании команд или ответов в фигурных скобках могут указываться значения полей. Если
при изменении состояния или получении команды в метке меняются параметры (флаги, счетчики), то они
указываются в квадратных скобках в конце. Возможные параметры:

- `epc`: идентификатор EPCID, указывается только при входе метки в область (A-строка).
- `epc`: содержимое банка TID, указывается только в A-строке.
- `ao`: вектор ориентации антенны метки.
- `s`: флаги сессий, строка длины 4, состоящая из символов `'A'` и `'B'`.
- `cnt`: счетчик числа слотов.

Все, что указывается посе символа `#`, считается комментарием.


Пример файла журнала меток:
```
...
08.000  12  A   -24.0   (10.0,0,1.75)   OFF [ao=(-1,0,0),epc=E0F1251234,tid=12345678,s=AAAA]
...
10.000  12  U   -17.9   (4.1,0,1.75)    IDLE
...
10.580  12  R   -13.5   (4.0,0,1.75)    QUERY{S=0,T=A,Q=2}  [cnt=1]
10.667  12  R   -11.9   (3.89,0,1.75)   QREP                [cnt=0]
10.668  12  T   -11.9   (3.89,0,1.75)   RN16
10.690  12  R   -11.7   (3.6,0,1.75)    ACK
...
10.750  12  R   -11.7   (3.6,0,1.75)    QREP                [cnt=65535,s=BAAA]
...
10.790  12  U   -100.0  (2.5,0,1.75)    OFF
10.890  12  U   -11.0   (2.2,0,1.75)    IDLE                [s=AAAA]
10.900  12  R   -11.1   (2.15,0,1.75)   QUERY{S=0,T=B,Q=2}  # ignored
...
14.000  12  D   # depart
```

В файле `rfidsim_<имя конфигурации>_<n>_packets.txt` сохраняется журнал передачи пакетов
о считанных метках. Так как сам протокол передачи данных в ЦОД не моделируется,
здесь просто сохраняем данные о прочитанных метках и времени, когда они были прочитаны.
Каждая строка имеет следующий формат:

```
<время> epc:<EPC> tid:<TID| - > ant:<номер антенны> rssi:<средняя мощность сигнала>
```

Пример:

```
...
10.110 epc:E0F1251234 tid:12345678 ant:1 rssi:-78.1
...
```

#### Анализ модели RFID

Формат вызова для анализа результатов:

```sh
> rfidsim analyze <команда> <имя конфигурации> [-n|--number=<n>]
```

Анализатор поддерживает следующие команды:

- `packets intervals`: построить по журналу чтения меток (`rfidsim_<имя конфигурации>_<n>_packets.txt`) CSV-файл с интервалами между чтениями (он содержит колонки `N,Interval`). Результат будет записан в файл `rfidsim_<имя конфигурации>_<n>_intervals.txt`.
- `export config vars`: экспортирует в CSV-файл те параметры из конфигурации, которые изменяются между симцляциями, то есть те параметры, которые были указаны с помощью массивов или диапазонов. Названия колонок соответствуют самым последним ключам конфигурации, например `m`, `tari`, `speed`.
- `stats id`: рассчитать вероятность идентификации для каждого сценария. Рассчитываются вероятности идентификации отдельных меток и автомобилей, результат записывается в CSV-файл `rfidsim_<имя конфигурации>_probability.csv` с колонками `N`, `FrontEpcProb`, `BackEpcProb`, `FrontTidProb`, `BackTidProb`, `VehicleProb`. 
- `stats tags inventory`: рассчитать, в каком числе раундов метка в среднем принимала участие;
- `stats reader inventory`: рассчитать средние длительности раундов, слотов.


## Структура данных

```sh
data/
|- dumps/
   |- video_01.pcapng         # короткий видео файл <КАКОЙ?>, RTP-трафик
   |- video_05.pcapng         # другой видео файл <КАКОЙ> большего размера
|- traffic/
   |- intervals_video_01.csv  # CSV-файл из video_00.pcapng
   |- intervals_video_05.csv  # CSV-файл из video_01.pcapng
|- results/
   |- rfidsim/
      # Здесь оказываются все файлы, генерируемые утилитой rfidsim
   |- rfidana/
      # Здесь файлы, генерируемые rfidana


|- datasets/
   |- coarse/
      |- traffic/
         |- intervals_rfid_30.csv
         |- intervals_rfid_120.csv
         #
         # Результаты восстановления MAP-потоков для видео и RFID-трафика:
         # > qumos fit arrivals --config=coarse
         #
         |- arrivals_video_00.json
         |- arrivals_video_05.json
         |- arrivals_rfid_30.json
         |- arrivals_rfid_120.json
      |- channels/
         |- delays_dcf_video_00.csv     # > wisim channel dcf --config=coarse
         ...
         |- delays_dcf_rfid_120.csv
         |- delays_relay_video_00.csv   # > wisim channel relay --config=coarse
         ...
         |- delays_relay_rfid_120.csv
         |- service_dcf_video_00.json   # > qumos fit delays --config=coarse
         ...
         |- service_relay_rfid_120.json
      |- networks/
         |- simul_dcf_1_video_00.csv   # > wisim net dcf --config=coarse
                                       # формат: simul_<тип канала>_<число станций>_<имя трафика>.csv
         ...
         |- simul_relay_10_rfid_120.csv
         |- mc_dcf_1_video_00.json     # > qumos simulate dcf --config=coarse
         ...
         |- mc_relay_10_rfid_120.json
         |- dpa_dcf_1_video_00.json    # > qumos approx --config=coarse
         ...
         |- dpa_relay_10_rfid_120.json
         |- ana_dcf_1_video_00.json    # > qumos analytic --config=coarse
         ...
         |- ana_relay_3_rfid_120.json
      |- rfidsim
         |- rounds_30.csv              # > rfidsim --config=coarse
         ...
         |- rounds_120.csv
         |- results_30.json
         ...
         |- results_120.json
      |- rfidana
         |- results_??.json             # > rfidana --config=coarse
```

## Конфигурация экспериментов

Все конфигурации хранятся в config.json. Пример конфигурации (в реальном json-файле комментариев нет):

```json
[
    {
        "name": "coarse",  // имя конфигуркции (датасета)
        "traffic": [
            {
                "name": "low",
                "dist": "normal",
                "mean": 
            }
        ],
        "channels": {
            "dcf": {
                "bitrate": 1000,  // кбит/с
                "cwmin": 2,
                "cwmax": 16,
                "slot": 1e-9,
                "sifs": 1e-8,
            }   
        }
        "config": {
            "rfidsim": {
            },
            "rfidana": {

            },
            "qumos": {

            },
            "wisim": {

            },
            "traf": {

            }
        }
    }
]
```

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
