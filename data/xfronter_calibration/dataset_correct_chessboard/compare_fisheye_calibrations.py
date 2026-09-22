import pathlib
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython.display import display
sys.modules.setdefault('pathlib._local', pathlib)

PROJECT_ROOT = pathlib.Path('/home/dbabiarczyk/studia/semestr10/camera_calibration')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.networks_tool.models import FisheyeCornerGRUSequenceCalibrationNet

MODEL_CHECKPOINTS = {
    'Sieć GRU': PROJECT_ROOT / 'outputs/calibration_net_fisheye/best_model.pth',
    'Sieć LSTM': PROJECT_ROOT / 'outputs/calibration_net_fisheye_lstm/best_model.pth',
    'Sieć SSM': PROJECT_ROOT / 'outputs/calibration_net_fisheye_ssm/best_model.pth',
}

class FrameEncoder(nn.Sequential):
    def __init__(self, points=81):
        super().__init__(
            nn.Flatten(1),
            nn.Linear(points * 4, 96),
            nn.ReLU(),
            nn.LayerNorm(96),
            nn.Linear(96, 96),
            nn.ReLU(),
        )

class FisheyeLSTM(nn.Module):
    def __init__(self, points=81):
        super().__init__()
        self.frame_encoder = FrameEncoder(points)
        self.temporal_encoder = nn.LSTM(96, 128, 2, batch_first=True)
        self.regressor = nn.Sequential(nn.Linear(128, 96), nn.ReLU(), nn.Dropout(.1), nn.Linear(96, 8))
    def forward(self, x):
        b, t, p, f = x.shape
        z = self.frame_encoder(x.reshape(b*t, p, f)).reshape(b, t, -1)
        return self.regressor(self.temporal_encoder(z)[0][:, -1])

class SSMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_decay = nn.Parameter(torch.zeros(128))
        self.input_projection = nn.Linear(96, 128)
        self.output_projection = nn.Linear(128, 96)
        self.skip_projection = nn.Linear(96, 96)
    def forward(self, x):
        state = torch.zeros(x.shape[0], 128, device=x.device, dtype=x.dtype)
        decay = torch.sigmoid(self.logit_decay).to(x.dtype)
        ys = []
        for frame in x.unbind(1):
            state = decay * state + (1 - decay) * self.input_projection(frame)
            ys.append(self.output_projection(state) + self.skip_projection(frame))
        return torch.stack(ys, 1)

class FisheyeSSM(nn.Module):
    def __init__(self, points=81):
        super().__init__()
        self.frame_encoder = FrameEncoder(points)
        self.temporal_encoder = SSMBlock()
        self.temporal_norm = nn.LayerNorm(96)
        self.regressor = nn.Sequential(nn.Linear(96, 96), nn.ReLU(), nn.Dropout(.1), nn.Linear(96, 8))
    def forward(self, x):
        b, t, p, f = x.shape
        z = self.frame_encoder(x.reshape(b*t, p, f)).reshape(b, t, -1)
        return self.regressor(self.temporal_norm(self.temporal_encoder(z))[:, -1])

def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    if config.model_name == 'fisheye_corner_gru_sequence':
        model = FisheyeCornerGRUSequenceCalibrationNet(num_outputs=8, num_points=81)
    elif config.model_name == 'fisheye_corner_lstm_sequence':
        model = FisheyeLSTM()
    elif config.model_name == 'fisheye_corner_ssm_sequence':
        model = FisheyeSSM()
    else: raise ValueError(config.model_name)
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    return model

predictions = {}
for name, path in MODEL_CHECKPOINTS.items():
    model = load_model(path)
    with torch.no_grad(): predictions[name] = model(network_input.cpu()).numpy()[0].astype(float)

def matrices(values):
    values = values.copy(); w, h = image_size_calibration
    values[:4] *= [w, h, w, h]
    K = np.array([[values[0], 0, values[2]], [0, values[1], values[3]], [0, 0, 1]], float)
    return K, values[4:].reshape(4, 1)

methods = {'OpenCV fisheye': (camera_matrix, distortion_coefficients)}
methods.update({name: matrices(value) for name, value in predictions.items()})

def rmse(name, K, D):
    total = 0.; count = 0
    for obj, img in zip(object_points_all, image_points_all):
        # cv2.solvePnP interprets distCoeffs using the Brown-Conrady convention, not the
        # fisheye one, so K/D must never be passed to it directly regardless of the method.
        # Points are always undistorted first with the fisheye model, then solvePnP is run
        # with an identity camera and zero distortion.
        normalized = cv2.fisheye.undistortPoints(img, K, D)
        ok, r, t = cv2.solvePnP(obj.reshape(-1,3), normalized.reshape(-1,2), np.eye(3), None)
        if not ok: raise RuntimeError('Nie udało się wyznaczyć pozy planszy: ' + name)
        projected, _ = cv2.fisheye.projectPoints(obj, r, t, K, D)
        total += np.sum((img.reshape(-1,2) - projected.reshape(-1,2)) ** 2); count += len(img)
    return float(np.sqrt(total / count))

names = ['fx','fy','cx','cy','k1','k2','k3','k4']
rows = []
for name, (K, D) in methods.items():
    values = [K[0,0], K[1,1], K[0,2], K[1,2], *D.reshape(-1)]
    rows.append({'metoda': name, **dict(zip(names, values)), 'RMSE reprojekcji [px]': rmse(name, K, D)})
all_methods_parameter_table = pd.DataFrame(rows)
print('Zestawienie wszystkich czterech metod')
display(all_methods_parameter_table.style.format({**{n:'{:.8f}' for n in names}, 'RMSE reprojekcji [px]':'{:.6f}'}))
parameter_comparison_table = all_methods_parameter_table.set_index('metoda').T
parameter_comparison_table.insert(0, 'jednostka', ['px','px','px','px','-','-','-','-','px'])
print('Tabela parametrów: wiersze = parametry, kolumny = metody')
display(parameter_comparison_table.style.format('{:.8f}', subset=parameter_comparison_table.columns[1:]))

opencv_reference = all_methods_parameter_table.set_index('metoda').loc['OpenCV fisheye', names]
difference_rows = []
for method_name in all_methods_parameter_table['metoda']:
    method_values = all_methods_parameter_table.set_index('metoda').loc[method_name, names]
    difference_rows.append({'metoda': method_name, **dict(zip(names, method_values - opencv_reference))})
difference_table = pd.DataFrame(difference_rows).set_index('metoda')
print('Różnice parametrów względem OpenCV fisheye (metoda - OpenCV)')
display(difference_table.style.format('{:+.8f}'))

example_paths = [p for p in image_paths_for_calibration if p.is_file()][:3]
if len(example_paths) < 3: raise RuntimeError('Potrzeba trzech zdjęć przykładowych.')
fig, axes = plt.subplots(3, len(methods)+1, figsize=(30, 17), squeeze=False)
for row, path in enumerate(example_paths):
    source = cv2.imread(str(path)); axes[row,0].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB)); axes[row,0].set_title('Przed korekcją\n'+path.name)
    for col, (name, (K,D)) in enumerate(methods.items(), 1):
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K,D,image_size_calibration,np.eye(3),balance=0.,new_size=image_size_calibration)
        mx,my = cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),newK,image_size_calibration,cv2.CV_32FC1)
        corrected = cv2.remap(source,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        axes[row,col].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)); axes[row,col].set_title('Po korekcji\n'+name)
    for ax in axes[row]: ax.axis('off')
fig.suptitle('Porównanie przed i po kalibracji — trzy przykładowe zdjęcia', fontsize=18)
plt.tight_layout(rect=(0,0,1,.97)); plt.show()
