import numpy as np
from scipy import ndimage
from progress.bar import Bar
import cv2
import os


CrowdFlowPath = "E:/Dataset/TUBCrowdFlow/"  # Your dataset path
ImgFolderPathDict = {
                        "estimate": "estimate/",
                        "gt_flow": "gt_flow/",
                        "gt_traj": "gt_trajectories/",
                        "images": "images/",
                        "masks": "masks/"
                    }
SceneFolderNameLis = ["IM01/", "IM01_hDyn/",
                      "IM02/", "IM02_hDyn/",
                      "IM03/", "IM03_hDyn/",
                      "IM04/", "IM04_hDyn/",
                      "IM05/", "IM05_hDyn/"]
GTTrajTXTFiles = {
                    "bgTraj": "bgTrajectories.csv",
                    "denseTraj": "denseTrajectories.csv",
                    "personTraj": "PersonTrajectories.csv"
                 }

SaveFolder = {
                "person": "PersonTrajectories/",
                "bgtraj": "bgTrajectories/",
                "densetraj": "denseTrajectories/"
              }


def person_annotate_img_generator(txtfile, frame):
    init_img = np.zeros((720, 1280))

    person_pos = np.loadtxt(txtfile, delimiter=",")
    frame_per_pos = np.array(person_pos[:, 2 * frame: 2 * (frame + 1)], dtype=np.uint32)
    shape = frame_per_pos.shape

    for i in range(shape[0]):
        tmp_img = np.zeros((720, 1280))
        pos = frame_per_pos[i, :]
        if pos[0] == 0 and pos[1] == 0:
            continue
        elif pos[0] >= 720 or pos[1] >= 1280:
            continue
        elif pos[0] < 0 or pos[1] < 0:
            continue
        tmp_img[pos[0], pos[1]] = 1.0
        tmp_img = ndimage.filters.gaussian_filter(tmp_img, 10)
        max_num = tmp_img.max()
        tmp_img = np.array(tmp_img * (128 / max_num), dtype=np.uint8)
        init_img += tmp_img

    ret_img = np.where(init_img > 255, 255, init_img)
    ret_img = np.array(ret_img, dtype=np.uint8)

    return ret_img


if __name__ == "__main__":
    frame_num_list = [300, 300, 250, 300, 450]

    for i, scene in enumerate(SceneFolderNameLis):
        path = CrowdFlowPath + \
                    ImgFolderPathDict['gt_traj'] + scene + GTTrajTXTFiles['bgTraj']
        frame_num = frame_num_list[int(i / 2)]
        SaveFolderpath = CrowdFlowPath + ImgFolderPathDict['gt_traj'] + scene + SaveFolder['bgtraj']
        bar = Bar('Makeing Label... : {}'.format(scene), max=frame_num)
        for i in range(frame_num):
            full_save_path = SaveFolderpath + "bgTrajectories_frame_{:0=4}.png".format(i)
            if os.path.isfile(full_save_path):
                bar.next()
                continue
            img = person_annotate_img_generator(path, i)
            cv2.imwrite(full_save_path, img)
            bar.next()

        bar.finish()
