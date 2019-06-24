import os
import csv
import argparse


def list2str(l):
    x = ""
    for c in l:
        x += c + " "

    return x[:-1]


def txt2dict(txt_path):
    output = {}
    with open(txt_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            ir = row[0].split(" ")
            ro = list2str(ir[1:])
            output[ir[0]] = ro

    return output


def subjects_csv_creator(subjects_path):

    csv_name = "Subject.csv"

    for subject_path in subjects_path:

        actions_paths = [
            os.path.join(subject_path, x) for x in os.listdir(subject_path)
        ]

        intermediate_paths = []

        for action_path in actions_paths:
            for path in os.listdir(action_path):
                intermediate_paths.append(os.path.join(action_path, path))

        with open(csv_name, 'a') as f:
            writer = csv.writer(f)
            for path in intermediate_paths:
                Num_images = len(os.listdir(os.path.join(path, "color")))
                for i in range(Num_images):
                    rgb_path = os.path.join(
                        path, "color/color_{:0>4d}.jpeg".format(i))
                    depth_path = os.path.join(
                        path, "depth/depth_{:0>4d}.png".format(i))
                    joint_data = txt2dict(os.path.join(
                        path, "skeleton.txt"))['{:0>4d}'.format(i)]
                    writer.writerow([rgb_path, depth_path, joint_data])


def main():

    parser = argparse.ArgumentParser(description="FPAB parser")
    parser.add_argument("subjects", help="Subjects path", type=str)
    args = parser.parse_args()

    subject_paths = [
        os.path.join(args.subjects, "Subject_" + str(x)) for x in range(1, 7)
    ]

    subjects_csv_creator(subject_paths)


if __name__ == "__main__":
    main()