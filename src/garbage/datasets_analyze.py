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
                    "bgTraj": "bgTrajectories.txt",
                    "denseTraj": "denseTrajectories.txt",
                    "personTraj": "PersonTrajectories.txt"
                }

for scene in SceneFolderNameLis:
    tmppath = CrowdFlowPath + \
                ImgFolderPathDict['gt_traj'] + scene
    print("======= {} =======".format(scene))

    for txtfile in GTTrajTXTFiles:
        print("  File Name: {}".format(txtfile))
        with open(tmppath + GTTrajTXTFiles[txtfile]) as f:
            count = 0
            not_printed = True

            while True:
                s_line = f.readline()
                linelis = s_line.split(',')
                if not_printed:
                    print("    Colum: {}".format(len(linelis)))
                    not_printed = False
                if not s_line:
                    break
                count += 1

            print("    Count: {}".format(count))
