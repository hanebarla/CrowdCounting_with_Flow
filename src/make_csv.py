import random
import csv
from progress.bar import Bar
import argparse


def main():
    """
    csv format
    --------
        index 0: input image(step t), index 1: person label(step t), index 2: input label(step t-1),
        index 3: person label(step t-1), index 4: label flow(step t-1 2 t), index 5: input image(step t+1)
        index 6: preson label(step t+1), index 7: label flw(step t 2 t+1)
    """
    parser = argparse.ArgumentParser(description="""
                                                 Please specify the root folder of the Datasets.
                                                 In default, path is 'E:/Dataset/TUBCrowdFlow/'
                                                 """)
    parser.add_argument('-p', '--path', default='E:/Dataset/TUBCrowdFlow/')
    args = parser.parse_args()

    TrainPathList = []
    TrainPathDict = {}
    TestPathList = []
    TestPathDict = {}

    frame_num_list = [300, 300, 250, 300, 450]
    DatasetFolder = args.path
    ImgFolder = DatasetFolder + "images/"
    GTTrajFolder = DatasetFolder + "gt_trajectories/"
    GTFlowFolder = DatasetFolder + "gt_flow/"
    GTPersonFolder = "PersonTrajectories/"
    SceneFolderNameLis = ["IM01/", "IM01_hDyn/",
                          "IM02/", "IM02_hDyn/",
                          "IM03/", "IM03_hDyn/",
                          "IM04/", "IM04_hDyn/",
                          "IM05/", "IM05_hDyn/"]

    bar = Bar('Makeing csv... ', max=len(SceneFolderNameLis))
    for i, scene in enumerate(SceneFolderNameLis):
        frame_num = frame_num_list[int(i / 2)]
        gtTraj_img_path = GTTrajFolder + scene + GTPersonFolder
        for fr in range(frame_num - 2):
            t_m = fr
            t = fr + 1
            t_p = fr + 2

            t_img_path = ImgFolder + scene + "frame_{:0=4}.png".format(t)
            t_m_img_path = ImgFolder + scene + "frame_{:0=4}.png".format(t_m)
            t_p_img_path = ImgFolder + scene + "frame_{:0=4}.png".format(t_p)

            t_person_img_path = gtTraj_img_path + "PersonTrajectories_frame_{:0=4}.png".format(t)
            t_m_person_img_path = gtTraj_img_path + "PersonTrajectories_frame_{:0=4}.png".format(t_m)
            t_p_person_img_path = gtTraj_img_path + "PersonTrajectories_frame_{:0=4}.png".format(t_p)

            t_m_t_flow_path = GTFlowFolder + scene + "frameGT_{:0=4}.png".format(t_m)
            t_t_p_flow_path = GTFlowFolder + scene + "frameGT_{:0=4}.png".format(t)

            PathList_per_frame = [t_img_path, t_person_img_path,
                                  t_m_img_path, t_m_person_img_path, t_m_t_flow_path,
                                  t_p_img_path, t_p_person_img_path, t_t_p_flow_path]

            if int(i / 2) < 3:
                TrainPathList.append(t_img_path)
                TrainPathDict[t_img_path] = PathList_per_frame
            else:
                TestPathList.append(t_img_path)
                TestPathDict[t_img_path] = PathList_per_frame

        bar.next()
    bar.finish()

    random.shuffle(TrainPathList)
    with open("TrainData_Path.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for path in TrainPathList:
            writer.writerow(TrainPathDict[path])

    with open("TestData_path.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for path in TestPathList:
            writer.writerow(TestPathDict[path])

    print("Done")


if __name__ == "__main__":
    main()
