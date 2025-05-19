import requests
import time

with open('test_FD001.txt', 'r') as file:
    for line_number, line in enumerate(file, 1):
        cleaned_line = line.strip()
        unit_ID, cycles, setting_1, setting_2, setting_3, T2, T24, T30, T50, P2, P15, P30, Nf, Nc, epr, Ps30, phi, NRf, NRc, BPR, farB, htBleed, Nf_dmd, PCNfR_dmd, W31, W32 = map(float, cleaned_line.split(' '))
        unit_ID = int(unit_ID)
        cycles = int(cycles)
        response = requests.post(
                    'http://127.0.0.1:5000/data',
                    json={'unit_ID': unit_ID,
                        'cycles': cycles,
                        'setting_1': setting_1,
                        'setting_2': setting_2,
                        'setting_3': setting_3,
                        'T2': T2,
                        'T24': T24,
                        'T30': T30,
                        'T50': T50,
                        'P2': P2,
                        'P15': P15,
                        'P30': P30,
                        'Nf': Nf,
                        'Nc': Nc,
                        'epr': epr,
                        'Ps30': Ps30,
                        'phi': phi,
                        'NRf': NRf,
                        'NRc': NRc,
                        'BPR': BPR,
                        'farB': farB,
                        'htBleed': htBleed,
                        'Nf_dmd': Nf_dmd,
                        'PCNfR_dmd': PCNfR_dmd,
                        'W31': W31,
                        'W32': W32}
                )
        print("Ответ сервера:", response.json())
        time.sleep(0.5)
