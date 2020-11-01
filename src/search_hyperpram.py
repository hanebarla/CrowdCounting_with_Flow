import train


def search():
    Learning_rate = [0.1, 0.01, 1e-3, 1e-4, 1e-5]
    weight_decay = [0.1, 0.01, 1e-3, 1e-4, 1e-5]

    for lr in Learning_rate:

        for wd in weight_decay:
            print("==========lr: {}, wd: {} Start!=========".format(lr, wd))
            try:
                train.train(lr, wd)
            except AssertionError:
                print("lr: {}, wd: {}: Error occursed".format(lr, wd))


if __name__ == "__main__":
    search()
