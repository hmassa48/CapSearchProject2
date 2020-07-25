




def main():
    #trainiing settings 

    #Multi GPU strategy
    strategy = tf.distribute.MirroredStrategy() 
    
    #load dataset
    image_paths,mask_paths = load_images("Data/lgg-mri-segmentation/kaggle_3m/")

    #read in the images 
    masks,images = read_in_images(image_paths,mask_paths)
    
    #reshape mask image to (256,256) -- binarize
    for i in range(0,len(masks)):
        m = masks[i]
        m = m[:,:,0]
        m.reshape(m.shape[0],m.shape[1])
        masks[i] = m

    #make images and masks into numpy arrays and normalize the masks
    images = np.asarray(images)
    masks = np.asarray(masks)
    masks = masks / 255
    masks = masks.reshape(masks.shape[0],masks.shape[1],masks.shape[2],1)


    #split the data
    img_overall_train, img_test, mask_overall_train, mask_test = train_test_split(images, masks, test_size=0.16667, random_state=42)
    img_train, img_val, mask_train, mask_val = train_test_split(img_overall_train, mask_overall_train, test_size = 0.166667, random_state = 32)
   
