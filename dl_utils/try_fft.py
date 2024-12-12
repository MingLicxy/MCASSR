import torch
import torch.fft

x = torch.randn(3, 240, 240)
x_complex = torch.complex(x, torch.zeros_like(x))

print("###############", x_complex.real.dtype) # torch.float32
print("###############", x_complex.imag.dtype) 

x_fft1 = torch.fft.fft2(x)
x_fft2 = torch.fft.fft2(x_complex)
#print("###############", x_fft1==x_fft2) 

#  [3, 240, 240]

x_end1 = torch.fft.ifft2(x_fft1)
x_end2 = torch.fft.ifft2(x_fft2)
#print("###############", x_end1==x_end2) 