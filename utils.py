import numpy as np
import matplotlib.pyplot as plt
import json
import re
import os
import config


def split_sequence(sequences, n_steps, n_features_in, n_features_out): 
    X1, y1 = sequences[:, :n_features_in], sequences[:, -n_features_out:]
    X, y = [], []
    for i in range(len(sequences)): 
        # find the end of this pattern 
        end_ix = i + n_steps 
        # check if we are beyond the dataset 
        if end_ix > len(sequences) - 1: 
            break
        # gather input and output parts of the pattern 
        seq_x, seq_y = X1[i:end_ix, :], y1[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def error(pred, g_truth):
    return (abs(pred - g_truth) / (g_truth.max(axis=0) - g_truth.min(axis=0))) * 100

def list_ext(directory, ext):
    return [re.sub(f'.{ext}', '', f) for f in os.listdir(directory) if f.endswith('.' + ext)]

# --------------------------------------------------
# CHOKE
# --------------------------------------------------

def generate_choke(t_lim, mean_ts_global=19, std_ts=4, max_div=3, mean_pick=55, std_pick=2):
    mean_slow = mean_pick // 2
    std_slow = mean_slow * 3 // 4

    opened = np.random.randint(2)
    steps = []

    while len(steps) < t_lim:
        mean_ts = mean_ts_global
        if len(steps) == 0:
            mean_ts = mean_ts // (np.random.randint(max_div) + 1)     # first sequence
        ts = mean_ts + np.random.randint(std_ts * 2 + 1) - std_ts     # 19 +- 4

        if opened == False:
            steps.extend([0] * ts)
            opened = True
        else:
            picks = [mean_pick + np.random.randint(std_pick) + 1 for _ in range(ts)]
            steps.extend(picks)
            opened = False

#         slow = np.random.randint(2)
#         if slow == 1:
        steps.append(mean_slow + abs(np.random.randint(std_slow * 2 + 1) - std_slow))
    
    steps = [s / 100 for s in steps]
    return steps[:t_lim]

# from http://chris35wills.github.io/parabola_python/
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    denom = (x1-x2) * (x1-x3) * (x2-x3);
    A = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom;
    B = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom;
    C = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;

    return A, B, C

def generate_new_choke(t_lim, mean_ts_global=19, std_ts=4, max_div=3, mean_pick=55, std_pick=3):
    mean_slow = mean_pick * 2 // 5
    std_slow = mean_slow
    
    left_inc = np.mean([49, 51, 53])
    std_left_inc = np.std([49, 51, 53])
    right_inc = np.mean([61, 65, 69])
    std_right_inc = np.std([61, 65, 69])
    
    left_dec = 73
    std_left_dec = 3
    right_dec = 57
    std_right_dec = 3
    
    std_i = 3
    std_d = 2
    
    left_vinc = np.mean([59, 61])
    std_left_vinc = np.std([59, 61])
    min_vinc = np.mean([52, 53])
    std_min_vinc = np.std([52, 53])
    right_vinc = np.mean([73, 93])
    std_right_vinc = np.std([73, 93])
    
    left_vdec = np.mean([59, 66])
    std_left_vdec = np.std([59, 66])
    min_vdec = np.mean([51, 55])
    std_min_vdec = np.std([51, 55])
    right_vdec = np.mean([54, 59])
    std_right_vdec = np.std([54, 59])
    
    left_v = np.mean([54, 59, 75])
    std_left_v = np.std([54, 59, 75])
    min_v = np.mean([50, 53, 54])
    std_min_v = np.std([50, 53, 54])
    right_v = np.mean([55, 67, 74])
    std_right_v = np.std([55, 67, 74])
    
    left_z = 0.4
    std_left_z = 0.1
    right_z = 0.1
    std_right_z = 0.1
    std_i_z = 0.001
    
    opened = np.random.randint(2)
    steps = []

    while len(steps) < t_lim:
        # sample time step
        mean_ts = mean_ts_global
        if len(steps) == 0:
            mean_ts = mean_ts // (np.random.randint(max_div) + 1)     # first sequence
        ts = mean_ts + np.random.randint(std_ts * 2 + 1) - std_ts     # 19 +- 4

        # sample val
        opcode = np.random.choice(['INC', 'DEC', 'VINC', 'VDEC', 'V'], p=[4/12, 1/12, 2/12, 2/12, 3/12])
        zeros_p = np.random.choice(['zero', 'posit'], p=[10/15, 5/15])
        if opened == False:
            if zeros_p == 'zero':
                vals = [0] * ts
            else:
                l_z = left_z + np.random.uniform(std_left_z * 2 + 1) - std_left_z
                r_z = right_z + np.random.uniform(std_right_z * 2 + 1) - std_right_z
                vals = [l_z - (l_z - r_z)/ts * t for t in range(ts)]
            steps.extend(vals)
            opened = True
        else:
            if opcode == 'INC':
                l_inc = left_inc + np.random.randint(std_left_inc * 2 + 1) - std_left_inc
                r_inc = right_inc + np.random.randint(std_right_inc * 2 + 1) - std_right_inc
                picks = [l_inc + np.random.randint(std_i) + (r_inc - l_inc)/ts * t for t in range(ts)]
            elif opcode == 'DEC':
                l_dec = left_dec + np.random.randint(std_left_dec * 2 + 1) - std_left_dec
                r_dec = right_dec + np.random.randint(std_right_dec * 2 + 1) - std_right_dec
                picks = [l_dec + np.random.randint(std_d) - (l_dec - r_dec)/ts * t for t in range(ts)]
            elif opcode == 'VINC':
                l_vinc = left_vinc + np.random.randint(std_left_vinc * 2 + 1) - std_left_vinc
                r_vinc = right_vinc + np.random.randint(std_right_vinc * 2 + 1) - std_right_vinc
                m_vinc = min_vinc + np.random.randint(std_min_vinc * 2 + 1) - std_min_vinc
                x1, y1 = [0, l_vinc]
                x2, y2 = [ts // 3, m_vinc]
                x3, y3 = [ts, r_vinc]
                a, b, c = calc_parabola_vertex(x1, y1, x2, y2, x3, y3)
                picks = [a*t**2 + b*t + c + np.random.randint(1) for t in range(ts)]
            elif opcode == 'VDEC':
                l_vdec = left_vdec + np.random.randint(std_left_vdec * 2 + 1) - std_left_vdec
                r_vdec = right_vdec + np.random.randint(std_right_vdec * 2 + 1) - std_right_vdec
                m_vdec = min_vdec + np.random.randint(std_min_vdec * 2 + 1) - std_min_vdec
                x1, y1 = [0, l_vdec]
                x2, y2 = [ts * 2 // 3, m_vdec]
                x3, y3 = [ts, r_vdec]
                a, b, c = calc_parabola_vertex(x1, y1, x2, y2, x3, y3)
                picks = [a*t**2 + b*t + c + np.random.randint(2) for t in range(ts)]
            else: # if opcode == 'V':
                l_v = left_v + np.random.randint(std_left_v * 2 + 1) - std_left_v
                r_v = right_v + np.random.randint(std_right_v * 2 + 1) - std_right_v
                m_v = min_v + np.random.randint(std_min_v * 2 + 1) - std_min_v
                x1, y1 = [0, l_v]
                x2, y2 = [ts // 2, m_v]
                x3, y3 = [ts, r_v]
                a, b, c = calc_parabola_vertex(x1, y1, x2, y2, x3, y3)
                picks = [a*t**2 + b*t + c + np.random.randint(2) for t in range(ts)]
                
            steps.extend(picks)
            opened = False

        steps.append(mean_slow + abs(np.random.randint(std_slow * 2 + 1) - std_slow))
    
    steps = [s / 100 for s in steps]
    return steps[:t_lim]

# --------------------------------------------------
# PLOTS
# --------------------------------------------------

def plot_loss(loss, full_screen=False, model_name=''):
    width = 18 if full_screen else 6
    height = 6 if full_screen else 5
    fig = plt.figure(figsize=(width, height))
    plt.rcParams.update({'font.size': 13})

    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training loss")
    plt.grid(True)
    plt.show()
    
#     fig.savefig(f'{config.STUDY_PATH}/{model_name}_train_loss.png', dpi=fig.dpi);
    
def plot_train_val_loss(loss, val_loss, model_name=''):
    fig = plt.figure(figsize=(18, 6))
    plt.rcParams.update({'font.size': 13})

    plt.subplot(121)
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training loss")
    plt.grid(True)
    
    plt.subplot(122)
    plt.plot(val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Validation loss")
    plt.grid(True)
    
    plt.show()
    
#     fig.savefig(f'{config.STUDY_PATH}/{model_name}_losses.png', dpi=fig.dpi);

def plot_pred(data, model_name=None):
    with open(f'{config.ROOT}/code/units.json') as infile:
        units = json.load(infile)
        
    fig = plt.figure(figsize=(20, 17))
    plt.rcParams.update({'font.size': 12})

    plt.subplot(311)
    plt.plot(np.array(data["preds"])[:, -3], ls="--", color="tab:blue", label="Prediction")
    plt.plot(np.array(data["g_truth"])[:, -3], color="tab:red", label="True values")
    plt.axvline(data["train_size"], color="black", ls="--", lw=1.5)
    if "val_size" in data:
        plt.axvline(data["val_size"], color="black", ls="--", lw=1.5)
    plt.legend()
    plt.ylabel(units["BORE_OIL_VOL"])
    plt.title("Prediction vs true values: Oil")
    plt.grid(True)

    plt.subplot(312)
    plt.plot(np.array(data["preds"])[:, -2], ls="--", color="tab:orange", label="Prediction")
    plt.plot(np.array(data["g_truth"])[:, -2], color="tab:purple", label="True values")
    plt.axvline(data["train_size"], color="black", ls="--", lw=1.5)
    if "val_size" in data:
        plt.axvline(data["val_size"], color="black", ls="--", lw=1.5)
    plt.legend()
    plt.ylabel(units["BORE_OIL_VOL"])
    plt.title("Prediction vs true values: Gas")
    plt.grid(True)

    plt.subplot(313)
    plt.plot(np.array(data["preds"])[:, -1], ls="--", color="tab:green", label="Prediction")
    plt.plot(np.array(data["g_truth"])[:, -1], color="tab:brown", label="True values")
    plt.axvline(data["train_size"], color="black", ls="--", lw=1.5)
    if "val_size" in data:
        plt.axvline(data["val_size"], color="black", ls="--", lw=1.5)
    plt.legend()
    plt.ylabel(units["BORE_OIL_VOL"])
    plt.title("Prediction vs true values: Water")
    plt.grid(True)
    
#     if model_name:
#         fig.savefig(f'{config.STUDY_PATH}/{config.MODEL_NAME}_pred.png', dpi=fig.dpi);
    
def plot_err(data):
    # plot predictions vs true values
    fig = plt.figure(figsize=(20, 15))
    plt.rcParams.update({'font.size': 12})

    plt.subplot(311)
    plt.plot(np.array(data["errors"])[:, -3], ls="--", color="tab:blue")
    # plt.axvline(inputs.index[data["train_size"]], color="black", ls="--", lw=1.5)
    plt.ylabel("%")
    plt.title('Error: Oil')
    plt.grid(True)

    plt.subplot(312)
    plt.plot(np.array(data["errors"])[:, -2], ls="--", color="tab:orange")
    # plt.axvline(inputs.index[data["train_size"]], color="black", ls="--", lw=1.5)
    plt.ylabel("%")
    plt.title('Error: Gas')
    plt.grid(True)

    plt.subplot(313)
    plt.plot(np.array(data["errors"])[:, -1], ls="--", color="tab:green")
    # plt.axvline(inputs.index[data["train_size"]], color="black", ls="--", lw=1.5)
    plt.ylabel("%")
    plt.title('Error: Water')
    plt.grid(True)
    
#     fig.savefig(f'{config.STUDY_PATH}/{config.MODEL_NAME}_err.png', dpi=fig.dpi);
    
def print_results(data):
#     print (f"Experiment time:  %s\n" % (data["cur_time"][:-7]))
    print (f"Model name:       %s" % (data["model_name"]))
    print (f"Training time:    %.3f sec" % (data["training_time"]))
    print (f"Train error:      %.3f" % (np.mean(data["errors"][:data["train_size"]])))
    
    if "val_size" not in data:
        print (f"Test error (avg): %.3f" % (np.mean(data["errors"][data["train_size"]:])))
        print (f"           (oil): %.3f" % (np.mean(np.array(data["errors"][data["train_size"]:])[:, 0])))
        print (f"           (gas): %.3f" % (np.mean(np.array(data["errors"][data["train_size"]:])[:, 1])))
        print (f"           (wat): %.3f\n" % (np.mean(np.array(data["errors"][data["train_size"]:])[:, 2])))
    else:
        print (f"Val error:        %.3f" % (np.mean(data["errors"][data["train_size"]:data["val_size"]])))
        print (f"Test error (avg): %.3f" % (np.mean(data["errors"][data["val_size"]:])))
        print (f"           (oil): %.3f" % (np.mean(np.array(data["errors"][data["val_size"]:])[:, 0])))
        print (f"           (gas): %.3f" % (np.mean(np.array(data["errors"][data["val_size"]:])[:, 1])))
        print (f"           (wat): %.3f\n" % (np.mean(np.array(data["errors"][data["val_size"]:])[:, 2])))
        
def plot_well(well):
    with open(f'{config.ROOT}/code/units.json') as infile:
        units = json.load(infile)
        
    plt.figure(figsize=(18, 50))

    for i, col in enumerate(well.columns[1:]):
        plt.subplot(len(well.columns), 1, i+1)
        plt.plot(well[col])
        plt.title(f'{col}')
        plt.ylabel(units[col])
        plt.grid(True)
        
def plot_color(df):
    with open(f'{config.ROOT}/code/units.json') as infile:
        units = json.load(infile)
    
    fontsize = 16
    fontsize1 = 14
    prop = {'size': 12}
    plt.figure(figsize=(20, 40))

    plt.subplot(711)
    plt.plot(df["AVG_WHP_P"], label="Wellhead")
    plt.title("Wellhead pressure", fontsize=fontsize)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units['AVG_WHP_P'], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)
    plt.legend(prop=prop)

    plt.subplot(712)
    plt.plot(df["AVG_DOWNHOLE_PRESSURE"], label="Bottomhole")
    plt.title("Bottomhole pressure", fontsize=fontsize)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units['AVG_WHP_P'], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)
    plt.legend(prop=prop)

    plt.subplot(713)
    plt.plot(df["AVG_WHT_P"], label="Wellhead")
    plt.plot(df["AVG_DOWNHOLE_TEMPERATURE"], label="Bottomhole")
    plt.title("Temperatures", fontsize=fontsize)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units["AVG_WHT_P"], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)
    plt.legend(prop=prop)

    plt.subplot(714)
    plt.plot(df["AVG_CHOKE_SIZE_P"], color="black")
    plt.title("Choke opening", fontsize=fontsize)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units['AVG_CHOKE_SIZE_P'], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)

    plt.subplot(715)
    plt.plot(df["DP_CHOKE_SIZE"], color="brown")
    plt.title("Pressure differential in choke", fontsize=fontsize)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units['DP_CHOKE_SIZE'], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)

    plt.subplot(716)
    plt.plot(df["BORE_OIL_VOL"], color="tab:purple", label="Oil")
    plt.plot(df["BORE_WAT_VOL"], color="tab:green", label="Water")
    plt.title("Flow rates", fontsize=fontsize)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units["BORE_OIL_VOL"], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)
    plt.legend(prop=prop)

    plt.subplot(717)
    plt.plot(df["BORE_GAS_VOL"], color="tab:red", label="Gas")
    plt.title("Flow rates", fontsize=16)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units["BORE_OIL_VOL"], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)
    plt.legend(prop=prop)

    plt.show();
    # fig.savefig(f'{path}/{model_name}_pred.png', dpi=fig.dpi);
    
def plot_cases(gens, names):
    with open(f'{config.ROOT}/code/units.json') as infile:
        units = json.load(infile)

    prop = {'size': 12}
    fontsize = 16
    fontsize1 = 14
    
    plt.figure(figsize=(20, 46))
    
    plt.subplot(811)
    for (gen, name) in zip(gens, names):
        plt.plot(gen["AVG_WHP_P"], label=name)
    plt.title("Wellhead pressure", fontsize=fontsize)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units['AVG_WHP_P'], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)
#     plt.legend(prop=prop)

    plt.subplot(812)
    for (gen, name) in zip(gens, names):
        plt.plot(gen["AVG_DOWNHOLE_PRESSURE"], label=name)
    plt.title("Bottomhole pressure", fontsize=fontsize)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units['AVG_WHP_P'], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)
#     plt.legend(prop=prop)

    plt.subplot(813)
    for (gen, name) in zip(gens, names):
        plt.plot(gen["AVG_WHT_P"], label=f"WHP_{name}")
        plt.plot(gen["AVG_DOWNHOLE_TEMPERATURE"], label=f"BHP_{name}")
    plt.title("Temperatures", fontsize=fontsize)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units["AVG_WHT_P"], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)
#     plt.legend(prop=prop)

    plt.subplot(814)
    for (gen, name) in zip(gens, names):
        plt.plot(gen["AVG_CHOKE_SIZE_P"], label=name)
    plt.title("Choke opening", fontsize=fontsize)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units['AVG_CHOKE_SIZE_P'], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)
#     plt.legend(prop=prop)

    plt.subplot(815)
    for (gen, name) in zip(gens, names):
        plt.plot(gen["DP_CHOKE_SIZE"], label=name)
    plt.title("Choke size", fontsize=fontsize)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units['DP_CHOKE_SIZE'], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)
#     plt.legend(prop=prop)

    plt.subplot(816)
    for (gen, name) in zip(gens, names):
        plt.plot(gen["BORE_OIL_VOL"], label=name)
    plt.title("Oil flow rate", fontsize=fontsize)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units["BORE_OIL_VOL"], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)
#     plt.legend(prop=prop)
    
    plt.subplot(817)
    for (gen, name) in zip(gens, names):
        plt.plot(gen["BORE_WAT_VOL"], label=name)
    plt.title("Water flow rate", fontsize=fontsize)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units["BORE_OIL_VOL"], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)
#     plt.legend(prop=prop)

    plt.subplot(818)
    for (gen, name) in zip(gens, names):
        plt.plot(gen["BORE_GAS_VOL"], label=name)
    plt.title("Gas flow rate", fontsize=fontsize)
    plt.xlabel("Days", fontsize=fontsize1)
    plt.ylabel(units["BORE_OIL_VOL"], fontsize=fontsize1)
    # plt.axvline(inputs.index[train_len], color="black", ls="--", lw=1.5)
    plt.grid(True)
#     plt.legend(prop=prop)

    plt.show();
    # fig.savefig(f'{path}/{model_name}_pred.png', dpi=fig.dpi);
    
def plot_estimations(watercut, GOR, PI, Q_l):
    fontsize = 16
    fontsize1 = 14
    
    plt.figure(figsize=(18, 20))
    
    plt.subplot(411)
    plt.plot(watercut*100)
    plt.grid(True)
#     plt.xlabel('Days', fontsize=fontsize1)
    plt.ylabel('%', fontsize=fontsize1)
    plt.title("Water cut", fontsize=fontsize)
    
    plt.subplot(412)
    plt.plot(GOR)
    plt.grid(True)
#     plt.xlabel('Days', fontsize=fontsize1)
    plt.ylabel('Sm3/Sm3', fontsize=fontsize1)
    plt.title("GOR", fontsize=fontsize)
    
    plt.subplot(413)
    plt.plot(PI)
    plt.grid(True)
#     plt.xlabel('Days', fontsize=fontsize1)
    plt.ylabel('Sm3/d/bar', fontsize=fontsize1)
    plt.title("PI", fontsize=fontsize)
    
    plt.subplot(414)
    plt.plot(Q_l)
    plt.grid(True)
#     plt.xlabel('Days', fontsize=fontsize1)
    plt.ylabel('Sm3/d', fontsize=fontsize1)
    plt.title("Q_l", fontsize=fontsize)