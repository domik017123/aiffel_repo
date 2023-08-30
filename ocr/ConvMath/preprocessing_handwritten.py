import json 
import os 
from PIL import Image, ImageOps
import splitfolders 

def main():
    root_dir_img = '.\\원천데이터\\손글씨'
    dir_name = '.\\dataset\\output\\'
    file_name = 'math.txt'

    os.makedirs(dir_name, exist_ok=True)

    img_list = []
    for root, dirs, files in os.walk(root_dir_img):
        for idx, file in enumerate(files):
            img_list.append(root + '\\' + file)

    json_list = []
    for i in range(len(img_list)):
        json_path = img_list[i].replace('원천', '라벨링')
        json_path = json_path.replace('png', 'json')
        json_list.append(json_path)

    # img crop with padding
    j = 0
    for i in range(0, len(json_list)):
        with open(json_list[i], 'r', encoding="utf-8") as f:
            json_data = json.load(f)
        for e in json_data['segments']:
            if e['type_detail'] == '수식' and e['equation'] is not None :
                crop_left = int(e['box'][0][0])
                crop_top = int(e['box'][0][1])
                crop_right = int(e['box'][2][0])
                crop_bottom = int(e['box'][2][1])

                del_y = crop_bottom - crop_top
                del_x = crop_right - crop_left 

                if del_x <= 1024 and del_y <= 1024:
                    if del_x <= 32:
                        pad_x = 32 - del_x
                    elif del_x > 32:
                        pad_x = 32 - (del_x % 32)

                    if del_y <= 32:
                        pad_y = 32 - del_y
                    elif del_y > 32:
                        pad_y = 32 - (del_y % 32)

                    padding = (0, 0, pad_x, pad_y) 

                    im = Image.open(img_list[i])
                    crop_image = im.crop((crop_left, crop_top, crop_right, crop_bottom))
                    crop_image = ImageOps.expand(crop_image, padding, fill=(255, 255, 255))
                    crop_image = crop_image.convert('L')
                    fname = (str(j)).zfill(7) + '.png' 
                    crop_image.save(dir_name + fname)

                    # write label file 
                    with open(file_name, 'a', encoding="utf-8") as f:
                        equation = e['equation']
                        equation = equation.replace('\n','\ n')
                        equation = equation.replace('\r','\ r')
                        equation = equation.replace('\t','\ t')
                        equation = equation.replace('\b','\ b')
                        f.write(equation)
                        f.write('\n')
                    j += 1

                    with open('id_info.txt', 'a', encoding="utf-8") as f:
                        f.write(fname)
                        f.write('\t')
                        f.write(img_list[i].split('\\')[-1])
                        f.write('\n')
                    id_dict = {}
                    id_dict[fname] = img_list[i].split('\\')[-1] 


                else:
                    pass



    data_dir = './dataset'
    output = './split_output'
    splitfolders.ratio(data_dir, output=output, seed=1337, ratio=(.8, .1, .1))

    test_dir = './split_output/test/output'
    test_dict = [] 
    for root, dirs, files in os.walk(test_dir):
        for idx, file in enumerate(files):
            test_dict.append(file)
    
    with open('테스트 데이터 목록.txt', 'a', encoding="utf-8") as f:
        for i in range(len(test_dict)):
            f.write(test_dict[i])
            f.write('\t')
            f.write(id_dict[test_dict[i]])
            f.write('\n')

if __name__ == "__main__":
    print('image preprocessing...')
    main()
    print('preprocessing complete')
