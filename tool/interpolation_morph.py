from models.s_ugatit_plus import S_UGATITPlus
from configs.cfgs_s_ugatit_plus import test_cfgs
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def preprocessing(img):
    img = torch.from_numpy(img).float() / 255.0
    img = img * 0.5 + 0.5
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img


def postprocessing(img):
    img = img * 0.5 + 0.5
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
    img = img.astype(np.uint8)  # RGB
    return img

torch.set_grad_enabled(False)
G = S_UGATITPlus(test_cfgs).G

weight = torch.load('/Users/yangjie08/dataset/ugatit_ckpt/morph-99.pt', map_location='cpu')['G']
G.load_state_dict(weight)
G.eval()

# img_path = '/Users/yangjie08/dataset/ugatit_result/morph_ugatit/7211_0.jpg'
img_path = '/Users/yangjie08/dataset/ugatit_result/morph_ugatit/7186_0.jpg'
img_path = '/Users/yangjie08/dataset/jujingwei_0.jpg'
img = Image.open(img_path)
img = np.array(img)[:, :256]
plt.imshow(img)
plt.show()
img_tensor = preprocessing(img)

# wab = G.get_avg_w(dir='AtoB', dev=torch.device('cpu'))
# wba = G.get_avg_w(dir='BtoA', dev=torch.device('cpu'))
# output = G.using_avg_w_forward(img_tensor, wab)

output = G.test_forward(img_tensor, 'AtoB')
output = postprocessing(output)
plt.imshow(output)
plt.show()
exit()



output_sets = []
for i in range(50):
    w = wab + (i+1)/50 * (wba - wab)
    output = G.using_avg_w_forward(img_tensor, w)
    output = postprocessing(output)
    output_sets.append(output)
    plt.imshow(output)
    plt.show()
for i in range(len(output_sets)):
    output = output_sets[i]
    output = Image.fromarray(output)
    output.save('../figs/{}.jpg'.format(i))
