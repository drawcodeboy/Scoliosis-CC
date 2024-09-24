import cv2
import matplotlib.pyplot as plt
import numpy as np

from dataloader import load_dataset

def image_dataset_test():

    train_ds = load_dataset()
    test_ds = load_dataset(mode='test')

    print(train_ds[0][0][0].shape, train_ds[0][0][1].shape)
    print(train_ds)

    img1 = (train_ds[0][0][0].permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
    img2 = (train_ds[0][0][1].permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
    
    img3 = (test_ds[0][0].permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
    
    print(img1, img2, img3)

    cv2.imshow('aug1', img1)
    cv2.imshow('aug2', img2)
    
    cv2.imshow('test', img3)

    cv2.waitKeyEx()
    cv2.destroyAllWindows()

    cv2.imwrite('./original.jpg', img3)
    
    cv2.imwrite('./aug1.jpg', img1)
    cv2.imwrite('./aug2.jpg', img2)
    '''
    test_ds = load_dataset(mode='test')
    test_ds[0]
    '''

def mask_dataset_test():

    train_ds = load_dataset(dataset='mask')
    test_ds = load_dataset(dataset='mask', mode='test')

    print(train_ds[0][0][0].shape, train_ds[0][0][1].shape)
    print(train_ds)

    img1 = (train_ds[0][0][0].permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
    img2 = (train_ds[0][0][1].permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
    
    img3 = (test_ds[0][0].permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
    
    print(img1, img2, img3)

    cv2.imshow('aug1', img1)
    cv2.imshow('aug2', img2)
    
    cv2.imshow('test', img3)

    cv2.waitKeyEx()
    cv2.destroyAllWindows()
    
    cv2.imwrite('./original.jpg', img3)

    cv2.imwrite('./aug1.jpg', img1)
    cv2.imwrite('./aug2.jpg', img2)

    '''
    test_ds = load_dataset(mode='test')
    test_ds[0]
    '''
    
def seq_dataset_test():
    train_ds = load_dataset(dataset='seq')
    test_ds = load_dataset(dataset='seq', mode='test')

    print(train_ds[0][0][0].shape, train_ds[0][0][1].shape)
    print(train_ds)

    seq1 = train_ds[0][0][0].view(-1).numpy()
    seq2 = train_ds[0][0][1].view(-1).numpy()
    
    seq3 = test_ds[0][0].view(-1).numpy()
    
    print(seq1.shape)
    
    x1 = np.arange(0, len(seq1), 1)
    x2 = np.arange(0, len(seq2), 1)
    x3 = np.arange(0, len(seq3), 1)
    
    fig, axes = plt.subplots(3, 1)
    
    axes[2].plot(x1, seq1)
    axes[1].plot(x2, seq2)
    axes[0].plot(x3, seq3)

    plt.show()
    '''
    test_ds = load_dataset(mode='test')
    test_ds[0]
    '''
    
if __name__ == '__main__':
    # image_dataset_test()
    # mask_dataset_test()
    seq_dataset_test()