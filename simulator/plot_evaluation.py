import csv
import matplotlib.pyplot as plt


def plot_succ_acc_scatter(filename):
    if 'grasp' in filename:
        data_succ, data_fail = {}, {}
        for err in ['pos_err', 'rot_err']:
            data_succ[err], data_fail[err] = {}, {}
            for obj in ['Spoon', 'Bottle', 'Mug']:
                data_succ[err][obj] = []
                data_fail[err][obj] = []
    else:
        data_succ, data_fail = {}, {}
        for err in ['pos_err', 'rot_err']:
            data_succ[err] = []
            data_fail[err] = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            try:
                obj_name = None
                if 'grasp' in filename:
                    for obj in ['Spoon', 'Bottle', 'Mug']:
                        if obj in line[3].strip():
                            obj_name = obj
                            break
                label = line[2].strip()
                print(label[0])
                if label[0] == 'T' or label[0] == '1':
                    if obj_name is None:
                        data_succ['pos_err'].append(float(line[0]))
                        data_succ['rot_err'].append(float(line[1]))
                    else:
                        data_succ['pos_err'][obj_name].append(float(line[0]))
                        data_succ['rot_err'][obj_name].append(float(line[1]))
                elif (label[0] == 'F' or label[0] == '0') and line[0].strip() != 'None':
                    if obj_name is None:
                        # print(line)
                        data_fail['pos_err'].append(float(line[0]))
                        data_fail['rot_err'].append(float(line[1]))
                    else:
                        data_fail['pos_err'][obj_name].append(float(line[0]))
                        data_fail['rot_err'][obj_name].append(float(line[1]))
            except:
                pass
        if 'grasp' not in filename:
            plt.plot(data_succ['pos_err'], data_succ['rot_err'], 'o', label='success')
            plt.plot(data_fail['pos_err'], data_fail['rot_err'], 'x', label='failure')
            text_succ_rate = 'succ rate: {:.2f}%'.format(
                100 * len(data_succ['pos_err']) / (len(data_succ['pos_err']) + len(data_fail['pos_err'])))
            plt.text(7, 40, text_succ_rate)
            print('{}/{}'.format(len(data_succ['pos_err']), len(data_succ['pos_err']) + len(data_fail['pos_err'])))
        else:
            i = 0
            for obj, col in zip(['Spoon', 'Bottle', 'Mug'], ['r', 'g', 'b']):
                plt.plot(data_succ['pos_err'][obj], data_succ['rot_err'][obj], 'o' + col, label=obj + ' success')
                plt.plot(data_fail['pos_err'][obj], data_fail['rot_err'][obj], 'x' + col, label=obj + ' failure')
                text_succ_rate = obj + ' succ rate: {:.2f}%'.format(
                    100 * len(data_succ['pos_err'][obj]) / (
                                len(data_succ['pos_err'][obj]) + len(data_fail['pos_err'][obj])))
                plt.text(6.5, 30 - i * 3, text_succ_rate)
                i += 1
        plt.legend()
        plt.xlabel('ACF position error (cm)')
        plt.ylabel('ACF rotation error (deg)')
        plt.xlim(0, 10)
        plt.ylim(0, 50)
        print(text_succ_rate)
        plt.savefig(filename.replace('.txt', '.png'), dpi=300)
        plt.savefig(filename.replace('.txt', '.pdf'))
        plt.show()


if __name__ == '__main__':
    plot_succ_acc_scatter('./experiments_out/pour_pipeline.txt')
