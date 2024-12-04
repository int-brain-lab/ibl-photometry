"""
Neurophotometrics FP3002 specific information.
The light source map refers to the available LEDs on the system.
The flags refers to the byte encoding of led states in the system.
"""

LIGHT_SOURCE_MAP = {
    'color': ['None', 'Violet', 'Blue', 'Green'],
    'wavelength': [0, 415, 470, 560],
    'name': ['None', 'Isosbestic', 'GCaMP', 'RCaMP'],
}

LED_STATES = {
    'Condition': {
        0: 'No additional signal',
        1: 'Output 1 signal HIGH',
        2: 'Output 0 signal HIGH',
        3: 'Stimulation ON',
        4: 'GPIO Line 2 HIGH',
        5: 'GPIO Line 3 HIGH',
        6: 'Input 1 HIGH',
        7: 'Input 0 HIGH',
        8: 'Output 0 signal HIGH + Stimulation',
        9: 'Output 0 signal HIGH + Input 0 signal HIGH',
        10: 'Input 0 signal HIGH + Stimulation',
        11: 'Output 0 HIGH + Input 0 HIGH + Stimulation',
    },
    'No LED ON': {
        0: 0,
        1: 8,
        2: 16,
        3: 32,
        4: 64,
        5: 128,
        6: 256,
        7: 512,
        8: 48,
        9: 528,
        10: 544,
        11: 560,
    },
    'L415': {
        0: 1,
        1: 9,
        2: 17,
        3: 33,
        4: 65,
        5: 129,
        6: 257,
        7: 513,
        8: 49,
        9: 529,
        10: 545,
        11: 561,
    },
    'L470': {
        0: 2,
        1: 10,
        2: 18,
        3: 34,
        4: 66,
        5: 130,
        6: 258,
        7: 514,
        8: 50,
        9: 530,
        10: 546,
        11: 562,
    },
    'L560': {
        0: 4,
        1: 12,
        2: 20,
        3: 36,
        4: 68,
        5: 132,
        6: 260,
        7: 516,
        8: 52,
        9: 532,
        10: 548,
        11: 564,
    },
}
