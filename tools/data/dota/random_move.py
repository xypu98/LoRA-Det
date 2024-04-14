import os, random, shutil

def moveFile(fileDir):#,save_dir):
    #save_ann_dir = save_dir.replace('images','annfiles')
    ann_dir = fileDir.replace('images','labelTxt')
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)

    # if not os.path.exists(save_ann_dir):
    #     os.mkdir(save_ann_dir)

    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    #filenumber = len(pathDir)
    #rate = 0.34 # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1 5361/15749
    picknumber = 466 #int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片（如果不想设置比例，可以将rate = 0.2注释掉，直接自定义picknumber数值，如picknumber = 10）
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    #print(sample)
    for name in sample:
    # shutil.copy(fileDir + name, tarDir + name) # 复制
        ann_name = name.replace('png','txt')
       # shutil.move(fileDir + name, save_dir + name) # 剪切
        #shutil.move(ann_dir+ann_name,save_ann_dir+ann_name)
        os.remove(fileDir + name)
        os.remove(ann_dir+ann_name)
    return


if __name__ == '__main__':
    fileDir = "/remote-home/pxy/data/DOTA/trainv2/images/"  # 源图片文件夹路径
    #save_dir = "/remote-home/pxy/data/split_ss_dota1_0/lora/val/images/"  # 移动到新的文件夹路径
    moveFile(fileDir)#,save_dir)
