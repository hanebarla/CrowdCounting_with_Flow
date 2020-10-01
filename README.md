# CrowdCounting_with_Flow

## How to Train
1. Create Python3(3.6>=) Environment<br>

    Anaconda
    ```
    pip install pytorch_memlab (本当は良くない)
    conda install matplotlib
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch (depend on machine)
    conda install -c conda-forge opencv, progress
    ```
1. Clone This Repositry
1. Go to the repo flolder
    ```
    cd CrowdCounting_with_Flow
    ```
1. Run ```make_csv.py``` to create Traindataset Pathset
    ```
    python src/make_csv.py -p [Absolute Path of the Dataset Folder]
    ```
1. make person label(サーバーにはもう作ってます。最初に作ったときGUIを一部用いて作ってしまったのでそこらへんを自動で処理するコードを書いているのでゼロからではまだ動きません。)
1. Run Train
    ```
    python src/train.py -p [TrainDatasetPathcsv path] -e [epock num] -wd [resize size of width] -ht [resize size of height]

    -p default: TrainData_Path.csv
    -e default: 50
    -wd default: 640
    -ht default: 360
    ```