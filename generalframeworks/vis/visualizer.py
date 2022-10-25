import visdom
import numpy as np
import time
from generalframeworks.utils import label_onehot
import torch

class Visualizer(object):
    def __init__(self, env='main', save_dir=None, **kwargs):
        self.vis = visdom.Visdom(server='http://localhost', port=8097, env=env, log_to_filename=save_dir, **kwargs)
        self.index = {}
        self.log_text = ''
        print('Visdom has been prepared. Please enter $python -m visdom.server$ in the terminal')

    def plot_many(self, d):
        '''
        '''
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, l):
        for i, img in enumerate(l):
            self.img(name=str(i), img_=img)

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                        win=name,
                        opts=dict(title=name),
                        update=None if x == 0 else 'append',
                        **kwargs)
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        self.vis.images(img_, 
                        win=name,
                        opts=dict(title=name),
                        **kwargs)
    
    def label(self, name, label_, **kwargs): # (4, 256, 256)
        label_ = label_onehot(label_, num_class=4) #(4, 4, 256, 256)
        #img_ = torch.zeros(label_.shape[0], 3, label_.shape[2], label_.shape[3])
        img_ = label_[:, 1] * 255
        img_ = torch.stack((img_, label_[:, 2] * 255), dim=1)
        img_ = torch.cat((img_, label_[:, 3].unsqueeze(1) * 255), dim=1)
            
        self.vis.images(img_,
                        win=name,
                        opts=dict(title=name),
                        **kwargs )
                
    def log(self, info, win='log_text'):
        self.log_text += ('[{time}]{info},br.'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

    def save(self):
        self.vis.save()

