import numpy as np
import matplotlib.pyplot as plt
def draw_result(cnt, train_loss, test_loss, acc, title):
    lst_iter = range(cnt)
    _, ax1 = plt.subplots()
    
    
    ax1.set_xlabel("n iteration")
    
    ax2 = ax1.twinx()
    ax1.plot(lst_iter, train_loss, '-b', label='Train Loss')
    ax1.plot(lst_iter, test_loss, '-r', label='Test Loss')
    ax2.plot(lst_iter, acc, '-g', label='Accuracy')
    #ax1.plot(lst_iter, acc, '-g', label='Accuracy')
    ax2.set_ylabel("%")
    ax2.set_ylim(0,100)
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(title)
    title = title.replace(' ', '_').replace('\n', '_')
    # save image
    plt.savefig(title +".svg")

