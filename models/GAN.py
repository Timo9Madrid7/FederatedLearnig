from torch import nn, cat, Tensor
from multipledispatch import dispatch

class SimpleGenerator(nn.Module):
    def __init__(self, noise_dimension=10, image_dimension=784, hidden_dimension=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.n_dim = noise_dimension
        self.im_dim = image_dimension
        self.h_dim = hidden_dimension

        # Generator network
        self.gen = nn.Sequential(
            self.generator_block(self.n_dim, self.h_dim),
            self.generator_block(self.h_dim, self.h_dim * 2),
            self.generator_block(self.h_dim * 2, self.h_dim * 4),
            self.generator_block(self.h_dim * 4, self.h_dim * 8),
            nn.Linear(self.h_dim * 8, self.im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)

    # Simple neural network single block
    def generator_block(self, in_dimension, out_dimension):
        return nn.Sequential(
            nn.Linear(in_dimension, out_dimension),
            nn.BatchNorm1d(out_dimension),
            nn.ReLU(inplace=True),
        )
    

class SimpleDiscriminator(nn.Module):
    def __init__(self, image_dimension=784, hidden_dimension=128, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.im_dim = image_dimension
        self.h_dim = hidden_dimension

        self.disc = nn.Sequential(
            self.discriminator_block(self.im_dim, self.h_dim * 4),
            self.discriminator_block(self.h_dim * 4, self.h_dim * 2),
            self.discriminator_block(self.h_dim * 2, self.h_dim),
            nn.Linear(self.h_dim, 1)
        )

    def forward(self, image):
        return self.disc(image)

    def discriminator_block(self, in_dimension, out_dimension):
      return nn.Sequential(
           nn.Linear(in_dimension, out_dimension),
           nn.LeakyReLU(0.2, inplace=True)
      )
    

class DCDiscriminator(nn.Module):
    def __init__(self, nc, ndf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        
    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class DCGenerator(nn.Module):
    def __init__(self, nc, nz, ngf, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.main = nn.Sequential(
          nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0, bias=False),
          nn.BatchNorm2d(ngf*4),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
          nn.Tanh()
      )
  
    def forward(self, input):
        output = self.main(input)
        return output
 
    
class cDCDiscriminator(DCDiscriminator):
    def __init__(self, nc, ndf, num_class, *args, **kwargs):
        super().__init__(nc, ndf, *args, **kwargs)
        self.image_conv = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True)
        )
        self.label_conv = nn.Sequential(
            nn.Conv2d(num_class, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True)
        )
        self.main = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
       
    def forward(self, inputs, labels):
        x, y = self.image_conv(inputs), self.label_conv(labels)
        x_cat_y = cat([x, y], dim=1)
        outputs = self.main(x_cat_y)
        return outputs.view(-1, 1).squeeze(1)
    

class cDCGenerator(DCGenerator):
    def __init__(self, nc, nz, ngf, num_class, *args, **kwargs):
        super().__init__(nc, nz, ngf, *args, **kwargs)
        self.noise_deconv = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True)
        )
        self.label_deconv = nn.Sequential(
            nn.ConvTranspose2d(num_class, ngf*2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, inputs, labels):
        x, y = self.noise_deconv(inputs), self.label_deconv(labels)
        x_cat_y = cat([x, y], 1)
        outputs = self.main(x_cat_y)
        return outputs


# class DCGenerator(nn.Module):
#     def __init__(self, nz: int, ngf: int, nc: int, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )

#     def forward(self, input):
#         output = self.main(input)
#         return output


# class DCDiscriminator(nn.Module):
#     def __init__(self, ndf: int, nc: int, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         output = self.main(input)
#         return output.view(-1, 1).squeeze(1)