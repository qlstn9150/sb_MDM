import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json

def plot(channel, model_list):
    color_list = list(mcolors.TABLEAU_COLORS)
    ber_list = []
    label_list = []

    if channel == 'awgn':
        path = './awgn/'
    elif channel == 'bursty':
        path = './bursty/'
    else:
        path = './rayleigh/'

    for name in model_list:
        with open(path + name + '.txt', 'r') as f:
            text = f.read()
            Vec_Eb_N0 = text.split('\n')[0]
            Vec_Eb_N0 = json.loads(Vec_Eb_N0)
            ber = text.split('\n')[1]
            ber = json.loads(ber)
        ber_list.append(ber)
        label_list.append(name)


    for i in range(len(model_list)):
        plt.semilogy(Vec_Eb_N0, ber_list[i], label=label_list[i], color=color_list[i], marker='o')

    plt.legend(loc=0)
    plt.xlabel('Eb/N0(dB)')
    plt.ylabel('BER')
    plt.title(channel + ' Channel')
    plt.grid('true')
    plt.show()

# RUN
model_list = ['basic', 'model5']
channel = 'bursty'
plot(channel, model_list)