H = 1024
W = 1024
resolution = 8
h, w = int(H/resolution), int(W/resolution)

    
def tex_trans(self, camou):
    # mask=[1, 4096, 4096, 3], camou=[1, 1024, 1024, 3]
    camou_column = []
    for i in range(6):
        camou_row_list = []
        for j in range(6):
            camou1 = T.RandomHorizontalFlip(p=0.5)(camou.permute(0, 3, 1, 2)[0]) # 依概率p水平翻转
            camou2 = T.RandomVerticalFlip(p=0.5)(camou1) # 依概率p垂直翻转
            if np.random.rand(1)>0.5:
                camou3 = TF.rotate(camou2, 90)
            else:
                camou3 = camou2
            # temp = camou3.detach().cpu().permute(1,2,0).numpy()*255
            # cv2.imwrite('./assets/tex1.jpg', cv2.cvtColor(temp, cv2.COLOR_RGB2BGR).astype(np.uint8))
            camou_row_list.append(camou3)
        camou_row = torch.cat(tuple(camou_row_list), 1)
        # print(camou_row.shape)
        camou_column.append(camou_row)
    camou_full = torch.cat(tuple(camou_column), 2).unsqueeze(0)
    # temp = camou_full[0].detach().cpu().permute(1,2,0).numpy()*255
    # cv2.imwrite('./assets/tex2.jpg', cv2.cvtColor(temp, cv2.COLOR_RGB2BGR).astype(np.uint8))
    camou_crop = T.RandomCrop(4096)(camou_full).permute(0, 2, 3, 1) # 随机裁剪
    # temp = camou_crop[0].detach().cpu().numpy()*255
    # cv2.imwrite('./assets/tex3.jpg', cv2.cvtColor(temp, cv2.COLOR_RGB2BGR).astype(np.uint8))
    # print(camou_crop.shape)
    return camou_crop

def tex_trans0(self, camou):
    # mask=[1, 4096, 4096, 3], camou=[1, 1024, 1024, 3]
    camou_column = []
    for i in range(4):
        camou_row_list = []
        for j in range(4):
            camou_row_list.append(camou)
        camou_row = torch.cat(tuple(camou_row_list), 1)
        camou_column.append(camou_row)
    camou_full = torch.cat(tuple(camou_column), 2)
    return camou_full

##################################### SETUP  #############################################
self.verts, self.faces, self.aux = load_obj(
    obj_name,
    load_textures=True,
    create_texture_atlas=False,
    texture_atlas_size=4,
    texture_wrap='repeat',
    path_manager=None,
)
self.camou0 = list(self.aux.texture_images.values())[0].to(self.device)[None]

expand_kernel = torch.nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(device)
expand_kernel.weight.data.fill_(0)
expand_kernel.bias.data.fill_(0)
for i in range(3):
    expand_kernel.weight[i, i, :, :].data.fill_(1)
##########################################################################################

# Initialize camouflage 
# Make the camouflage texture H/8 by W/8 and then the ConvTranspose2d will repeat those patches at a stride of 8 
# making the camouflage into the size of the image
camou_para = torch.rand([1, h, w, 3]).float().to(device)
# Make the parameter learnable
camou_para.requires_grad_(True)
# Make a copy of the beginning point to check against later
begin_para = deepcopy(camou_para)
# Add to the optimizer
optimizer = optim.Adam([camou_para], lr=lr)
# Acts as a repeat operator that tiles the camou pattern at strides of 8 (see figure 3)
camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

# NOT APPLICABLE This is the car mask that defines what type of car it is
self.camou_mask = torch.from_numpy(cv2.imread(camou_mask)/255).to(device).unsqueeze(0).float() 

# Sets the camou para1 texture as the texture to render
if self.tex_trans_flag:
    image = self.camou0
    # This will reference the camou mask so that the car layout is 0 
    # and the outside is 1 and fill in the area with 1s with camou0
    image = image * (1-self.camou_mask) 
    # This will add the formatted area outside the layout (for rendering I assume)
    # to the transposed camouflage texture inside the car mask
    image = image + (self.tex_trans(camou_para1) * self.camou_mask)
else:
    image = self.camou0 * (1-self.camou_mask) + self.tex_trans0(camou_para1) * self.camou_mask
self.mesh.textures = TexturesUV(verts_uvs=[self.verts_uvs], faces_uvs=[self.faces_uvs], maps=image)

# Use to render the cars in the __getitem__ of the dataset

# Acts as a repeat operator that tiles the camou pattern at strides of 8 (see figure 3)
camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
# makes sure to keep the values within 0 and 1
camou_para1 = torch.clamp(camou_para1, 0, 1)
# Sets the camou para1 texture as the texture to render
if self.tex_trans_flag:
    image = self.camou0
    # This will reference the camou mask so that the car layout is 0 
    # and the outside is 1 and fill in the area with 1s with camou0
    image = image * (1-self.camou_mask) 
    # This will add the formatted area outside the layout (for rendering I assume)
    # to the transposed camouflage texture inside the car mask
    image = image + (self.tex_trans(camou_para1) * self.camou_mask)
else:
    image = self.camou0 * (1-self.camou_mask) + self.tex_trans0(camou_para1) * self.camou_mask
self.mesh.textures = TexturesUV(verts_uvs=[self.verts_uvs], faces_uvs=[self.faces_uvs], maps=image)

camou_png = cv2.cvtColor((camou_para1[0].detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)