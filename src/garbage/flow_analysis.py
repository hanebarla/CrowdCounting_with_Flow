import numpy as np
import cv2

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


if __name__ == "__main__":
    path = CrowdFlowPath + ImgFolderPathDict['gt_flow'] + SceneFolderNameLis[0] + "frameGT_0000.png"
    img = cv2.imread(path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    print(img_hsv.max())
    h, s, v = cv2.split(img_hsv)
    print(h.max())
    img235 = np.where((h > 212.5 * 0.71) & (h < 275.5 * 0.71), 1, 0)
    imghs_235 = s * img235
    # print(imghs_235.max())
    cv2.imwrite("h_235.png", (img235 * 255))
    cv2.imwrite("hs_235.png", imghs_235)
