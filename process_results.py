import pandas as pd
import os

def process_dir_targeted(res, path="./Outputs/"):
    files = [f.name for f in os.scandir(path + res) if f.name.endswith(".txt")]
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


def process_dir_untargeted(res, path="./Outputs/"):
    files = [f.name for f in os.scandir(path + res) if f.name.endswith(".txt")]
    os.makedirs("./Results/" + res + "/", exist_ok=True)

    # averge
    df = pd.concat([pd.read_csv(path + res + "/" + f, sep=", ", engine="python") for f in files if "results" in f], ignore_index=True)
    df.loc[:, df.dtypes != 'float64'].astype('float')
    df = df.mean()
    df.to_csv("./Results/" + res + "/" + "results.csv", header=False)
    # time
    time, n = 0, 0
    for fname in [f for f in files if "time" in f]:
        with open(path + res + "/" + fname, "r") as f:
            line = f.readline().split(" ")
            time += float(line[1][:-1])
            n += float(line[-1])
    with open("./Results/" + res + "/" + "time.txt", "w") as f:
        f.write(str(time / n))


if __name__  == "__main__":
    result_dirs = [f.name for f in os.scandir("./Outputs") if f.is_dir()]

    for res in result_dirs:
        if "untargeted" in res:
            process_dir_untargeted(res)
        else:
            process_dir_targeted(res)
