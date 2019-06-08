import csv

url_withObj = "http://resources.mpi-inf.mpg.de/handtracker/data/GANeratedDataset/data/withObject/"
url_noObj = "http://resources.mpi-inf.mpg.de/handtracker/data/GANeratedDataset/data/noObject/"

with open("dataset_noObj.csv", 'w') as f:
    writer = csv.writer(f)
    for i in range(1, 141):
        for j in range(1, 1025):
            url_image = url_noObj + "{:0>4d}/{:0>4d}_color_composed.png".format(i, j)
            url_pose = url_noObj + "{:0>4d}/{:0>4d}_joint_pos.txt".format(i, j)
            writer.writerow([url_image, url_pose])

    for i in range(1, 897):
        url_image = url_noObj + "0141/{:0>4d}_color_composed.png".format(i)
        url_pose = url_noObj + "0141/{:0>4d}_joint_pos.txt".format(i)
        writer.writerow([url_image, url_pose])


with open("dataset_withObj.csv", 'w') as f:
    writer = csv.writer(f)
    for i in range(1, 185):
        for j in range(1, 1025):
            url_image = url_withObj + "{:0>4d}/{:0>4d}_color_composed.png".format(i, j)
            url_pose = url_withObj + "{:0>4d}/{:0>4d}_joint_pos.txt".format(i, j)
            writer.writerow([url_image, url_pose])
   
    for i in range(1, 183):
        url_image = url_withObj + "0185/{:0>4d}_color_composed.png".format(i)
        url_pose = url_withObj + "0185/{:0>4d}_joint_pos.txt".format(i)
        writer.writerow([url_image, url_pose])
