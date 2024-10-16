import pandas as pd
import os
import argparse

def process_dir_targeted(res, path="./Outputs/"):
    files = [f.name for f in os.scandir(path + res) if f.name.endswith(".txt")]
    if len([f for f in files if not 'no_adversarial_example_found' in f]) == 0:
        print('No adversarial examples found for test ' + res)
        return
    os.makedirs("./Results/" + res + "/", exist_ok=True)

    # IS
    df = pd.concat([pd.read_csv(path + res + "/" + f, sep=", ", engine="python") for f in files if "IS" in f], ignore_index=True).mean()
    df.to_csv("./Results/" + res + "/" + "IS.csv", header=False)
    # best
    df = pd.concat([pd.read_csv(path + res + "/" + f, sep=", ", engine="python") for f in files if "best" in f], ignore_index=True)
    df.loc[:, df.dtypes != 'float64'].astype('float')
    df = df.mean()
    df.to_csv("./Results/" + res + "/" + "best.csv", header=False)
    # averge
    df = pd.concat([pd.read_csv(path + res + "/" + f, sep=", ", engine="python") for f in files if "average" in f], ignore_index=True)
    df.loc[:, df.dtypes != 'float64'].astype('float')
    df = df.mean()
    df.to_csv("./Results/" + res + "/" + "average.csv", header=False)
    # worst
    df = pd.concat([pd.read_csv(path + res + "/" + f, sep=", ", engine="python") for f in files if "worst" in f], ignore_index=True)
    df.loc[:, df.dtypes != 'float64'].astype('float')
    df = df.mean()
    df.to_csv("./Results/" + res + "/" + "worst.csv", header=False)
    # time
    time, n = 0, 0
    for fname in [f for f in files if "time" in f]:
        with open(path + res + "/" + fname, "r") as f:
            line = f.readline().split(" ")
            time += float(line[1][:-1])
            n += float(line[-1])
    with open("./Results/" + res + "/" + "time.txt", "w") as f:
        f.write(str(time / n))


def process_dir_untargeted(outdir):

    files = [f.name for f in os.scandir(outdir) if f.name.endswith(".txt") or f.name.endswith(".csv")]
    if len([f for f in files if not 'no_adversarial_example_found' in f]) == 0:
        print('No adversarial examples found.')
        return
    
    resdir = outdir.replace('Outputs', 'Results')
    os.makedirs(resdir, exist_ok=True)

    df = pd.concat([pd.read_csv(outdir + f) for f in files if "results" in f], ignore_index=True)
    df.loc[:, df.dtypes != 'float64'].astype('float')
    df = df.mean()
    df.to_csv(resdir + "results.csv", header=False)

    time, succ, n, b, fw_calls, fw_calls_b, bw_calls, bw_calls_b = 0, 0, 0, 0, 0, 0, 0, 0
    for fname in [f for f in files if "time" in f]:
        with open(outdir + fname, "r") as f:
            line = f.readline().split(",")
            time += float(line[0])
            succ += float(line[1])
            n += float(line[2])
            b += float(line[3])
            fw_calls += float(line[4])
            fw_calls_b += float(line[5])
            bw_calls += float(line[6])
            bw_calls_b += float(line[7])

    string = f'Total compute time: {time}s, avg. time per image: {time / n}\n'
    string += f'Success rate: {succ / n}\n'
    string += f'Calls per batch: {fw_calls / b}, calls per image: {fw_calls_b / n}\n'
    string += f'Gradients per batch: {bw_calls / b}, gradients per image: {bw_calls_b / n}'

    with open(resdir + "other.txt", "w") as f:
        f.write(string)


if __name__  == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    if 'untargeted' in args.dir:
        process_dir_untargeted(args.dir)
    else:
        process_dir_targeted(args.dir)

