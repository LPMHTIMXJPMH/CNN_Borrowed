import net

real = net.YoLo()

import torch

batch_image = torch.rand(16, 448, 448, 3)

result = real(batch_image)

print(result.shape)
