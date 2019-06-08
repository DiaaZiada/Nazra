import os
import csv

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
            output[ir[0]] =  ro
    
    return output

def subject_csv_creator(subject_path):
        
    actions_paths = [os.path.join(subject_path, x) for x in os.listdir(subject_path)]

    intermediate_paths = []

    for action_path in actions_paths:
        for path in os.listdir(action_path):
            intermediate_paths.append(os.path.join(action_path, path))

    csv_name = subject_path.split("/")[-1] + ".csv"

    with open(csv_name, 'w') as f:
        writer = csv.writer(f)
        for path in intermediate_paths:
            Num_images = len(os.listdir(os.path.join(path, "color")))
            for i in range(Num_images):
                rgb_path = os.path.join(path, "color/color_{:0>4d}.jpeg".format(i))
                depth_path = os.path.join(path, "depth/depth_{:0>4d}.png".format(i))
                joint_data = txt2dict(os.path.join(path, "skeleton.txt"))['{:0>4d}'.format(i)]
                writer.writerow([rgb_path, depth_path, joint_data])

def main():

    subject6_path = "/media/rashad/01D3F6FB8718F3E0/FPAB/Subject_6"

    subject_csv_creator(subject6_path)

if __name__ == "__main__":
    main()