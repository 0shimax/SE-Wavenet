from torch.autograd import Variable
import torch.onnx
from models.ete_waveform import EteWave



model = EteWave(6).to('cpu')
# (n_batch, 1, seq_len, 3)
x = Variable(torch.randn(1, 1, 500, 6))
torch.onnx.export(model, x, 'EteWave.onnx', verbose=True)
